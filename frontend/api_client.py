"""HTTP helpers for the FastAPI prediction service."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import requests


API_TIMEOUT_SECONDS = 12
APPLICANT_FIELDS = {
    "application_id",
    "name",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "NAME_CONTRACT_TYPE",
}
REQUIRED_BATCH_FIELDS = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DAYS_BIRTH",
}


class APIError(RuntimeError):
    """A backend request failed or returned an invalid response."""


def _json_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(
            method,
            url,
            timeout=API_TIMEOUT_SECONDS,
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
        "risk_level",
        "model_version",
        "top_factors",
    }
    if not required.issubset(result):
        raise APIError("The prediction API response is missing required fields.")
    probability = result["risk_probability"]
    if (
        not isinstance(probability, (int, float))
        or not 0 <= probability <= 1
        or result["risk_level"] not in {"LOW", "MEDIUM", "HIGH"}
        or not isinstance(result["top_factors"], list)
    ):
        raise APIError("The prediction API returned an invalid assessment result.")
    return result


def predict_batch(base_url: str, csv_content: bytes, filename: str) -> dict[str, Any]:
    result = _json_request(
        "POST",
        f"{base_url.rstrip('/')}/predict/batch",
        files={"file": (filename, BytesIO(csv_content), "text/csv")},
    )
    required_counts = (
        "total_applications",
        "successful_predictions",
        "failed_predictions",
    )
    if (
        not isinstance(result.get("results"), list)
        or any(not isinstance(result.get(key), int) for key in required_counts)
    ):
        raise APIError("The batch API returned an invalid results payload.")
    return result


def sanitize_batch_csv(frame: pd.DataFrame) -> tuple[bytes, int, list[str]]:
    """Drop labels and unsupported columns before forwarding CSV to the API."""
    dropped_target = int("TARGET" in frame.columns)
    dropped_columns = [
        column
        for column in frame.columns
        if column not in APPLICANT_FIELDS
    ]
    clean = frame.drop(columns=dropped_columns, errors="ignore")

    missing_required = sorted(REQUIRED_BATCH_FIELDS - set(clean.columns))
    if missing_required:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing_required)
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
            expected = (
                "a negative finite day count"
                if column == "DAYS_BIRTH"
                else "a positive finite number"
            )
            raise ValueError(
                f"Required field {column} must be {expected}; invalid CSV row(s): "
                f"{rows}{suffix}."
            )
        clean[column] = numeric_values
    return clean.to_csv(index=False).encode("utf-8"), dropped_target, dropped_columns
