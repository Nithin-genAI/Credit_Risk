"""HTTP helpers for the FastAPI service and safe applicant-file preparation."""

from __future__ import annotations

import json
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_METADATA_PATH = PROJECT_ROOT / "models" / "baseline_feature_metadata.json"
ENGINEERED_FIELDS = {
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "CREDIT_TO_ANNUITY",
    "GOODS_TO_CREDIT",
    "INCOME_PER_FAMILY_MEMBER",
    "INCOME_PER_CHILD",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MAX",
}
FEATURE_METADATA = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
MODEL_INPUT_FIELDS = set(
    FEATURE_METADATA["numeric_features"] + FEATURE_METADATA["categorical_features"]
) - ENGINEERED_FIELDS
METADATA_FIELDS = {"application_id", "name", "SK_ID_CURR"}
APPLICANT_FIELDS = MODEL_INPUT_FIELDS | METADATA_FIELDS
REQUIRED_BATCH_FIELDS = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DAYS_BIRTH",
}
API_TIMEOUT_SECONDS = 12
BATCH_TIMEOUT_SECONDS = 120


class APIError(RuntimeError):
    """A backend request failed or returned an invalid response."""


def _json_request(
    method: str,
    url: str,
    *,
    timeout: int = API_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        response = requests.request(
            method,
            url,
            timeout=timeout,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise APIError(f"Could not reach the prediction API: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError(
            f"The prediction API returned invalid JSON (HTTP {response.status_code})."
        ) from exc

    if not response.ok:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise APIError(f"Prediction API returned HTTP {response.status_code}: {detail}")
    if not isinstance(payload, dict):
        raise APIError("The prediction API response must be a JSON object.")
    return payload


def get_health(base_url: str) -> dict[str, Any]:
    return _json_request("GET", f"{base_url.rstrip('/')}/health")


def validate_batch_upload(
    base_url: str,
    content: bytes,
    filename: str,
) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/applications/batch/validate",
        timeout=BATCH_TIMEOUT_SECONDS,
        files={"file": (Path(filename).name, BytesIO(content))},
    )
    required = {
        "valid",
        "file_type",
        "row_count",
        "missing_required_columns",
        "missing_optional_model_columns",
        "duplicate_columns",
        "unsupported_columns",
        "row_issue_count",
        "warnings",
    }
    if not required.issubset(result):
        raise APIError("Batch validation response is missing required fields.")
    return result


def predict_applicant(base_url: str, applicant: dict[str, Any]) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/predict",
        json=applicant,
    )
    required = {
        "application_id",
        "name",
        "risk_probability",
        "predicted_class",
        "provisional_classification",
        "classification_threshold",
        "model_version",
        "top_factors",
    }
    if not required.issubset(result):
        raise APIError("The prediction API response is missing required fields.")
    probability = result["risk_probability"]
    if (
        not isinstance(probability, (int, float))
        or not 0 <= probability <= 1
        or result["predicted_class"] not in {0, 1}
        or not isinstance(result["top_factors"], list)
    ):
        raise APIError("The prediction API returned an invalid assessment result.")
    return result


def predict_batch(base_url: str, content: bytes, filename: str) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/predict/batch",
        timeout=BATCH_TIMEOUT_SECONDS,
        files={"file": (Path(filename).name, BytesIO(content))},
    )
    if (
        not isinstance(result.get("results"), list)
        or any(
            not isinstance(result.get(key), int)
            for key in (
                "total_applications",
                "successful_predictions",
                "failed_predictions",
            )
        )
    ):
        raise APIError("The batch API returned an invalid results payload.")
    if (
        result["total_applications"] != len(result["results"])
        or result["successful_predictions"] + result["failed_predictions"]
        != result["total_applications"]
    ):
        raise APIError("Batch result counts do not align with returned rows.")
    for result_row in result["results"]:
        if (
            not isinstance(result_row, dict)
            or not isinstance(result_row.get("row_number"), int)
            or not isinstance(result_row.get("application_id"), str)
            or result_row.get("status") not in {"scored", "ERROR"}
        ):
            raise APIError("The batch API returned a misaligned or invalid row.")
    return result


def explain_applicant(
    base_url: str,
    application_id: str,
    applicant: dict[str, Any],
    *,
    generate_narrative: bool = False,
    provider_consent: bool = False,
) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/applications/explain",
        json={
            "application_id": application_id,
            "applicant": applicant,
            "generate_narrative": generate_narrative,
            "provider_consent": provider_consent,
        },
        timeout=BATCH_TIMEOUT_SECONDS,
    )
    required = {
        "application_id",
        "risk_probability",
        "predicted_class",
        "classification_threshold",
        "top_factors",
        "narrative",
    }
    if not required.issubset(result) or not isinstance(result["top_factors"], list):
        raise APIError("The applicant explanation response is incomplete.")
    return result


def applicant_chat_context_key(
    application_id: str,
    applicant: dict[str, Any],
) -> str:
    """Create a local-only key so chats stay isolated to one applicant record."""
    payload = json.dumps(
        {
            "application_id": application_id,
            "applicant": applicant_record_from_mapping(applicant),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chat_about_applicant(
    base_url: str,
    applicant: dict[str, Any],
    question: str,
    audience: str,
    *,
    history: list[dict[str, str]] | None = None,
    provider_consent: bool = False,
) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/applications/chat",
        json={
            "applicant": applicant_record_from_mapping(applicant),
            "question": question,
            "audience": audience,
            "history": history or [],
            "provider_consent": provider_consent,
        },
        timeout=BATCH_TIMEOUT_SECONDS,
    )
    if (
        not isinstance(result.get("answer"), str)
        or not isinstance(result.get("answer_status"), str)
        or not isinstance(result.get("evidence"), dict)
    ):
        raise APIError("The applicant chat API returned an invalid response.")
    return result


def ask_regulatory_assistant(
    base_url: str,
    question: str,
    audience: str,
    *,
    provider_consent: bool = False,
) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/assistant/rbi",
        json={
            "question": question,
            "audience": audience,
            "provider_consent": provider_consent,
        },
        timeout=BATCH_TIMEOUT_SECONDS,
    )
    if (
        not isinstance(result.get("answer"), str)
        or not isinstance(result.get("sources"), list)
        or not isinstance(result.get("answer_status"), str)
    ):
        raise APIError("The RBI assistant returned an invalid response.")
    return result


def get_regulatory_sources(base_url: str) -> dict[str, Any]:
    result = _json_request(
        "GET",
        f"{base_url.rstrip('/')}/assistant/rbi/sources",
    )
    if not isinstance(result.get("sources"), list):
        raise APIError("The RBI source catalog response is invalid.")
    return result


def applicant_record_from_mapping(record: dict[str, Any]) -> dict[str, Any]:
    """Return only recognized model fields, excluding labels and identifiers."""
    normalized = {
        _canonical_column(key): value
        for key, value in record.items()
    }
    return {
        key: (
            None
            if pd.isna(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in normalized.items()
        if key in MODEL_INPUT_FIELDS
    }


def _canonical_column(name: Any) -> str:
    raw = str(name).strip()
    aliases = {
        "application id": "application_id",
        "applicant id": "application_id",
        "applicant name": "name",
    }
    alias = aliases.get(raw.casefold())
    if alias:
        return alias
    known = APPLICANT_FIELDS | {"TARGET"}
    by_casefold = {column.casefold(): column for column in known}
    return by_casefold.get(raw.casefold(), raw)


def prepare_batch_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, list[str]]:
    """Keep only model-contract fields and identifiers; strip label/unknowns."""
    canonical = [_canonical_column(column) for column in frame.columns]
    duplicate_columns = pd.Index(canonical)[pd.Index(canonical).duplicated()].unique()
    if len(duplicate_columns):
        raise ValueError(
            "Uploaded columns become duplicates after alias normalization: "
            + ", ".join(map(str, duplicate_columns))
        )
    normalized = frame.copy()
    normalized.columns = canonical
    target_removed = "TARGET" in normalized.columns
    dropped_columns = [
        column for column in normalized.columns if column not in APPLICANT_FIELDS
    ]
    clean = normalized.drop(columns=dropped_columns, errors="ignore")
    missing_required = sorted(REQUIRED_BATCH_FIELDS - set(clean.columns))
    if missing_required:
        raise ValueError(
            "CSV/XLSX is missing required columns: " + ", ".join(missing_required)
        )
    for column in sorted(REQUIRED_BATCH_FIELDS):
        numeric_values = pd.to_numeric(clean[column], errors="coerce")
        invalid = ~np.isfinite(numeric_values)
        if column == "DAYS_BIRTH":
            invalid |= numeric_values >= 0
        else:
            invalid |= numeric_values <= 0
        invalid_rows = numeric_values[invalid].index + 2
        if len(invalid_rows):
            rows = ", ".join(map(str, invalid_rows[:10]))
            suffix = " (showing first 10)" if len(invalid_rows) > 10 else ""
            raise ValueError(
                f"Required field {column} has invalid CSV row(s): {rows}{suffix}."
            )
        clean[column] = numeric_values
    return clean, target_removed, dropped_columns


def encode_batch_frame(frame: pd.DataFrame, filename: str) -> tuple[bytes, str]:
    """Serialize sanitized table data as CSV or XLSX for batch scoring."""
    suffix = Path(filename).suffix.casefold()
    if suffix == ".xlsx":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False)
        return buffer.getvalue(), f"{Path(filename).stem}.xlsx"
    return frame.to_csv(index=False).encode("utf-8"), f"{Path(filename).stem}.csv"


def sanitize_batch_csv(frame: pd.DataFrame) -> tuple[bytes, int, list[str]]:
    """Backward-compatible CSV sanitizer preserving the complete model contract."""
    clean, target_removed, dropped_columns = prepare_batch_frame(frame)
    return clean.to_csv(index=False).encode("utf-8"), int(target_removed), dropped_columns
