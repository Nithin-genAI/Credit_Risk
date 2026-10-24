from __future__ import annotations

import logging
import math
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ApplicantChatRequest,
    ApplicantInput,
    BatchValidationResponse,
)
from backend.services.batch_inference import (
    BatchFileError,
    ENGINEERED_FEATURES,
    model_feature_schema,
    parse_batch_upload,
    score_batch,
    validation_payload,
)
from backend.services.feature_descriptions import describe_model_feature
from backend.services.llm_client import (
    explain_model_evidence,
    generate_applicant_chat_answer,
)
from backend.services.prediction_service import prediction_service
from backend.services.rbi_assistant import answer_question, source_catalog
from backend.settings import (
    CLASSIFICATION_THRESHOLD,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    GROQ_API_KEY_2,
    MAX_UPLOAD_BYTES,
    MAX_RBI_QUERY_CHARS,
)
from backend.schemas import ExplanationRequest, RegulatoryQuestion


logger = logging.getLogger(__name__)
MAX_APPLICANT_CHAT_HISTORY = 8

app = FastAPI(
    title="Drifting Oracle",
    description="Credit-risk prediction and explainability API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _classification(probability: float) -> tuple[int, str]:
    predicted_class = int(probability >= CLASSIFICATION_THRESHOLD)
    label = (
        "elevated_default_risk"
        if predicted_class
        else "lower_default_risk"
    )
    return predicted_class, label


def _validate_model_fields(applicant: dict[str, Any]) -> list[str]:
    model_features = set(model_feature_schema()[0])
    metadata_fields = {"application_id", "name", "SK_ID_CURR"}
    unsupported = sorted(set(applicant) - model_features - metadata_fields)
    return unsupported


def _invalid_model_values(applicant: dict[str, Any]) -> list[str]:
    _, numeric_features, categorical_features = model_feature_schema()
    invalid = []
    for field in numeric_features:
        value = applicant.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            invalid.append(field)
    for field in categorical_features:
        value = applicant.get(field)
        if value is not None and not isinstance(value, str):
            invalid.append(field)
    return sorted(invalid)


def _feature_details(factors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _, _, categorical_features = model_feature_schema()
    return {
        str(factor["feature"]): describe_model_feature(
            str(factor["feature"]),
            categorical_features,
        )
        for factor in factors
    }


def _local_applicant_chat_answer(
    question: str,
    evidence: dict[str, Any],
) -> str:
    factors = evidence["top_factors"]
    question_terms = set(question.casefold().split())
    matching = [
        factor
        for factor in factors
        if question_terms
        & set(
            (
                factor["feature"]
                + " "
                + factor["description"]
            ).casefold().replace("_", " ").split()
        )
    ]
    selected = matching or factors
    lines = [
        (
            f"- {factor['feature']}: {factor['description']} "
            f"({factor['impact'].replace('_', ' ')}, "
            f"SHAP {factor['shap_value']:+.5f})."
        )
        for factor in selected[:5]
    ]
    missing = evidence["missing_model_features"]
    missing_note = (
        f"{missing} model inputs were missing or null and imputed; "
        "the estimate may be less reliable."
        if missing
        else "All raw model fields were supplied for this request."
    )
    return (
        f"The model estimates {evidence['risk_probability']:.1%} default "
        f"probability and assigns the provisional class "
        f"{evidence['provisional_classification'].replace('_', ' ')} at the "
        f"{evidence['classification_threshold']:.0%} threshold. The strongest "
        f"available SHAP factors are:\n"
        + "\n".join(lines)
        + f"\n\n{missing_note} SHAP describes contribution to this model output, "
        "not real-world cause or certainty. This is not a lending decision."
    )


def _local_plain_language_explanation(
    probability: float,
    provisional_classification: str,
    threshold: float,
    factors: list[dict[str, Any]],
    missing_count: int,
) -> str:
    contributions = [
        (
            f"{factor['feature']} pushed the model output "
            f"{'toward' if factor['impact'] == 'increases_risk' else 'away from'} "
            f"default risk (SHAP {factor['shap_value']:+.5f})"
        )
        for factor in factors[:3]
    ]
    missing_note = (
        f"{missing_count} missing or null model inputs were imputed."
        if missing_count
        else "No model inputs were missing or null."
    )
    return (
        f"The model estimates a {probability:.1%} default probability and assigns "
        f"the provisional {provisional_classification.replace('_', ' ')} class "
        f"using a {threshold:.0%} threshold. The leading SHAP contributions are: "
        + "; ".join(contributions)
        + f". {missing_note} SHAP describes contributions to this model output, "
        "not real-world causes or certainty. This is not a lending decision."
    )


def _explanation_payload(
    applicant: dict[str, Any],
    application_id: str,
    generate_narrative: bool,
    provider_consent: bool,
) -> dict[str, Any]:
    unsupported = _validate_model_fields(applicant)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported fields are not accepted for inference.",
                "unsupported_columns": unsupported,
            },
        )
    invalid_values = _invalid_model_values(applicant)
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more model fields have incompatible values.",
                "invalid_columns": invalid_values,
            },
        )

    safe_applicant = {
        key: value
        for key, value in applicant.items()
        if key in set(model_feature_schema()[0]) | {"application_id", "name"}
    }
    try:
        probability, _ = prediction_service.predict(safe_applicant)
        factors = prediction_service.explain(safe_applicant, top_n=5)
    except Exception as exc:
        logger.error("applicant_explanation_failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Could not calculate the saved model explanation.",
        ) from exc

    predicted_class, provisional_classification = _classification(probability)
    model_features = model_feature_schema()[0]
    missing_features = [
        column for column in model_features if safe_applicant.get(column) is None
    ]
    supplied_features = len(model_features) - len(missing_features)
    if generate_narrative:
        if (GROQ_API_KEY or GROQ_API_KEY_2 or GEMINI_API_KEY) and not provider_consent:
            raise HTTPException(
                status_code=403,
                detail="Explicit provider consent is required for an AI narrative.",
            )
        narrative = explain_model_evidence(
            probability,
            predicted_class,
            CLASSIFICATION_THRESHOLD,
            factors,
        )
        if narrative.status == "ready" and narrative.text:
            narrative_payload = {
                "status": narrative.status,
                "text": narrative.text,
                "message": narrative.safe_message,
            }
        else:
            narrative_payload = {
                "status": "local_fallback",
                "text": _local_plain_language_explanation(
                    probability,
                    provisional_classification,
                    CLASSIFICATION_THRESHOLD,
                    factors,
                    len(missing_features),
                ),
                "message": narrative.safe_message
                or "The AI provider did not return an answer; local model evidence is shown.",
            }
    else:
        narrative_payload = {
            "status": "not_requested",
            "text": None,
            "message": (
                "Optional AI narrative not requested. Local model and SHAP "
                "evidence are available."
            ),
        }
    return {
        "application_id": application_id,
        "risk_probability": round(probability, 6),
        "predicted_class": predicted_class,
        "provisional_classification": provisional_classification,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "model_version": "baseline-v1",
        "top_factors": factors,
        "feature_details": _feature_details(factors),
        "feature_coverage": round(supplied_features / len(model_features), 4),
        "missing_model_features": len(missing_features),
        "missing_feature_names": missing_features,
        "narrative": narrative_payload,
        "warning": (
            "This model estimate is not a lending decision. Missing model inputs "
            "are imputed and may reduce reliability."
        ),
    }


def _applicant_chat_payload(request: ApplicantChatRequest) -> dict[str, Any]:
    unsupported = _validate_model_fields(request.applicant)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported fields are not accepted for inference.",
                "unsupported_columns": unsupported,
            },
        )
    invalid_values = _invalid_model_values(request.applicant)
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more model fields have incompatible values.",
                "invalid_columns": invalid_values,
            },
        )

    model_features = model_feature_schema()[0]
    safe_applicant = {
        key: value
        for key, value in request.applicant.items()
        if key in set(model_features)
    }
    try:
        probability, _ = prediction_service.predict(safe_applicant)
        factors = prediction_service.explain(safe_applicant, top_n=10)
    except Exception as exc:
        logger.error("applicant_chat_evidence_failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Could not calculate model evidence for this chat.",
        ) from exc

    predicted_class, provisional_classification = _classification(probability)
    details = _feature_details(factors)
    described_factors = [
        {
            **factor,
            **details[str(factor["feature"])],
        }
        for factor in factors
    ]
    prepared_features = prediction_service.prepare_input(safe_applicant).iloc[0]
    for factor in described_factors:
        raw_feature = factor.get("raw_feature")
        value = prepared_features.get(raw_feature) if raw_feature else None
        if value is None or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        factor["applicant_value"] = value
    missing_features = [
        column
        for column in model_features
        if safe_applicant.get(column) is None
    ]
    evidence: dict[str, Any] = {
        "risk_probability": round(probability, 6),
        "predicted_class": predicted_class,
        "provisional_classification": provisional_classification,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "feature_coverage": round(
            (len(model_features) - len(missing_features)) / len(model_features),
            4,
        ),
        "missing_model_features": len(missing_features),
        "missing_feature_names": missing_features,
        "top_factors": described_factors,
    }
    answer = _local_applicant_chat_answer(request.question, evidence)
    status = "local_evidence_only"
    provider_message: str | None = None

    if request.provider_consent and (
        GROQ_API_KEY or GROQ_API_KEY_2 or GEMINI_API_KEY
    ):
        history = [
            message.model_dump()
            for message in request.history[-MAX_APPLICANT_CHAT_HISTORY:]
        ]
        completion = generate_applicant_chat_answer(
            request.question,
            request.audience,
            evidence,
            history,
        )
        if completion.status == "ready" and completion.text:
            answer = completion.text
            status = "answered_with_selected_applicant_evidence"
            provider_message = completion.safe_message
        else:
            status = "local_fallback_after_provider_failure"
            provider_message = completion.safe_message or (
                "The language provider did not return an answer. Local evidence "
                "is shown instead."
            )

    return {
        "answer": answer,
        "answer_status": status,
        "provider_message": provider_message,
        "evidence": evidence,
        "disclaimer": (
            "SHAP values describe how inputs contributed to this model output; "
            "they do not establish real-world causes or predict a certain outcome. "
            "This is not a lending decision."
        ),
    }


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the maximum size of "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return content


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Drifting Oracle",
        "status": "running",
        "version": app.version,
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    raw_feature_count = len(model_feature_schema()[0])
    return {
        "status": "healthy",
        "model_loaded": prediction_service.model is not None,
        "preprocessor_loaded": prediction_service.preprocessor is not None,
        "expected_model_features": raw_feature_count,
        "engineered_features": len(ENGINEERED_FEATURES),
        "transformed_features": len(prediction_service.feature_names),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "llm_configured": bool(GROQ_API_KEY or GROQ_API_KEY_2 or GEMINI_API_KEY),
        "groq_configured": bool(GROQ_API_KEY or GROQ_API_KEY_2),
        "groq_backup_configured": bool(GROQ_API_KEY_2),
        "gemini_configured": bool(GEMINI_API_KEY),
        "rbi_live_search_configured": bool(GEMINI_API_KEY),
    }


@app.post("/predict")
def predict(applicant: ApplicantInput) -> dict[str, Any]:
    applicant_data = applicant.model_dump()
    unsupported = _validate_model_fields(applicant_data)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported fields are not accepted for inference.",
                "unsupported_columns": unsupported,
            },
        )
    invalid_values = _invalid_model_values(applicant_data)
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more model fields have incompatible values.",
                "invalid_columns": invalid_values,
            },
        )
    try:
        safe_applicant = {
            key: value
            for key, value in applicant_data.items()
            if key in set(model_feature_schema()[0]) | {"application_id", "name"}
        }
        probability, _ = prediction_service.predict(safe_applicant)
        factors = prediction_service.explain(safe_applicant, top_n=5)
    except Exception as exc:
        logger.error("single_prediction_failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed with the saved model.",
        ) from exc
    predicted_class, provisional_classification = _classification(probability)
    model_features = model_feature_schema()[0]
    supplied_features = sum(
        applicant_data.get(column) is not None
        for column in model_features
    )
    return {
        "application_id": applicant.application_id,
        "name": applicant.name,
        "risk_probability": round(probability, 6),
        "predicted_class": predicted_class,
        "provisional_classification": provisional_classification,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "model_version": "baseline-v1",
        "top_factors": factors,
        "feature_coverage": round(supplied_features / len(model_features), 4),
        "missing_model_features": len(model_features) - supplied_features,
        "warning": (
            "This single-applicant contract exposes only a partial feature "
            "set; missing model inputs are imputed and may reduce reliability."
        ),
    }


@app.post("/applications/explain")
def explain_applicant(request: ExplanationRequest) -> dict[str, Any]:
    return _explanation_payload(
        request.applicant,
        request.application_id,
        request.generate_narrative,
        request.provider_consent,
    )


@app.post("/applications/chat")
def applicant_chat(request: ApplicantChatRequest) -> dict[str, Any]:
    return _applicant_chat_payload(request)


@app.post("/assistant/rbi")
def rbi_assistant(question: RegulatoryQuestion) -> dict[str, Any]:
    if len(question.question) > MAX_RBI_QUERY_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Question must be at most {MAX_RBI_QUERY_CHARS} characters.",
        )
    if (
        GROQ_API_KEY or GROQ_API_KEY_2 or GEMINI_API_KEY
    ) and not question.provider_consent:
        raise HTTPException(
            status_code=403,
            detail="Explicit provider consent is required for an RBI assistant answer.",
        )
    return answer_question(question.question, question.audience)


@app.get("/assistant/rbi/sources")
def rbi_sources() -> dict[str, Any]:
    return source_catalog()


@app.post(
    "/applications/batch/validate",
    response_model=BatchValidationResponse,
)
async def validate_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        content = await _read_upload(file)
        parsed = parse_batch_upload(file.filename, content)
    except BatchFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return validation_payload(parsed)


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        content = await _read_upload(file)
        parsed = parse_batch_upload(file.filename, content)
    except BatchFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not parsed.valid_schema:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Uploaded file schema is not valid.",
                "missing_required_columns": parsed.missing_required_columns,
                "duplicate_columns": parsed.duplicate_columns,
            },
        )

    response = score_batch(parsed, CLASSIFICATION_THRESHOLD)
    response["batch_id"] = str(uuid.uuid4())
    return response
