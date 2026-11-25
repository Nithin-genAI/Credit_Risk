from __future__ import annotations

import logging
import math
import uuid
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.schemas import (
    AdminLogin,
    ApplicantChatRequest,
    ApplicantInput,
    BatchValidationResponse,
    DriftAnalyzeRequest,
    DriftOutcomeSubmission,
    ModelVersion,
)
from backend.services.admin_auth import (
    AuthConfigurationError,
    InvalidAccessToken,
    LocalPlaintextAuthDenied,
    admin_usernames,
    clear_login_failures,
    create_access_token,
    ensure_auth_request_allowed,
    login_is_rate_limited,
    record_login_failure,
    verify_access_token,
    verify_admin_credentials,
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
from backend.services.drift_monitor import DriftStorageError, drift_monitor
from backend.services.llm_client import (
    explain_model_evidence,
    generate_applicant_chat_answer,
)
from backend.services.prediction_service import prediction_service
from backend.services.rbi_assistant import answer_question, source_catalog
from backend.settings import (
    ADMIN_TOKEN_TTL_SECONDS,
    CLASSIFICATION_THRESHOLD,
    V2_CLASSIFICATION_THRESHOLD,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    GROQ_API_KEY_2,
    MAX_UPLOAD_BYTES,
    MAX_RBI_QUERY_CHARS,
)
from backend.schemas import ExplanationRequest, RegulatoryQuestion


logger = logging.getLogger(__name__)
MAX_APPLICANT_CHAT_HISTORY = 8
admin_bearer = HTTPBearer(auto_error=False)

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
    allow_headers=["Authorization", "Content-Type"],
)


def _ensure_auth_client_allowed(http_request: Request) -> str | None:
    client_host = (
        http_request.client.host if http_request.client is not None else None
    )
    try:
        ensure_auth_request_allowed(client_host)
    except LocalPlaintextAuthDenied as exc:
        raise HTTPException(
            status_code=403,
            detail="Local prototype authentication is only available from this machine.",
        ) from exc
    except AuthConfigurationError as exc:
        logger.error("admin_auth_configuration_invalid error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured.",
        ) from exc
    return client_host


def require_admin(
    http_request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_bearer),
) -> str:
    _ensure_auth_client_allowed(http_request)
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Admin authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return verify_access_token(credentials.credentials)
    except InvalidAccessToken as exc:
        raise HTTPException(
            status_code=401,
            detail="Admin session is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except AuthConfigurationError as exc:
        logger.error("admin_auth_configuration_invalid error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured.",
        ) from exc


@app.get("/auth/users")
def list_admin_users(http_request: Request) -> dict[str, list[str]]:
    _ensure_auth_client_allowed(http_request)
    try:
        return {"usernames": admin_usernames()}
    except AuthConfigurationError as exc:
        logger.error("admin_auth_configuration_invalid error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured.",
        ) from exc


@app.post("/auth/login")
def login_admin(
    request: AdminLogin,
    http_request: Request,
) -> dict[str, Any]:
    client_host = _ensure_auth_client_allowed(http_request) or "unknown"
    rate_limit_key = f"{client_host}\0{request.username}"
    if login_is_rate_limited(rate_limit_key):
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts. Try again in 15 minutes.",
            headers={"Retry-After": "900"},
        )
    try:
        valid_credentials = verify_admin_credentials(
            request.username,
            request.password.get_secret_value(),
        )
    except AuthConfigurationError as exc:
        logger.error("admin_auth_configuration_invalid error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured.",
        ) from exc
    if not valid_credentials:
        record_login_failure(rate_limit_key)
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    clear_login_failures(rate_limit_key)
    try:
        token = create_access_token(request.username)
    except AuthConfigurationError as exc:
        logger.error("admin_auth_configuration_invalid error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured.",
        ) from exc
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ADMIN_TOKEN_TTL_SECONDS,
    }


def _threshold_for_model(model_version: str) -> float:
    version = prediction_service.normalize_version(model_version)
    return (
        CLASSIFICATION_THRESHOLD
        if version == "v1"
        else V2_CLASSIFICATION_THRESHOLD
    )


def _classification(
    probability: float,
    model_version: str = "v1",
) -> tuple[int, str]:
    predicted_class = int(probability >= _threshold_for_model(model_version))
    label = (
        "elevated_default_risk"
        if predicted_class
        else "lower_default_risk"
    )
    return predicted_class, label


def _validate_model_fields(
    applicant: dict[str, Any],
    model_version: str = "v1",
) -> list[str]:
    model_features = set(model_feature_schema(model_version)[0])
    metadata_fields = {"application_id", "name", "SK_ID_CURR"}
    unsupported = sorted(set(applicant) - model_features - metadata_fields)
    return unsupported


def _invalid_model_values(
    applicant: dict[str, Any],
    model_version: str = "v1",
) -> list[str]:
    _, numeric_features, categorical_features = model_feature_schema(model_version)
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


def _feature_details(
    factors: list[dict[str, Any]],
    model_version: str = "v1",
) -> dict[str, dict[str, Any]]:
    _, _, categorical_features = model_feature_schema(model_version)
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
    model_version: str = "v1",
) -> dict[str, Any]:
    unsupported = _validate_model_fields(applicant, model_version)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported fields are not accepted for inference.",
                "unsupported_columns": unsupported,
            },
        )
    invalid_values = _invalid_model_values(applicant, model_version)
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
        if key
        in set(model_feature_schema(model_version)[0]) | {"application_id", "name"}
    }
    try:
        probability, _ = prediction_service.predict(safe_applicant, model_version)
        factors = prediction_service.explain(
            safe_applicant,
            top_n=5,
            model_version=model_version,
        )
    except Exception as exc:
        logger.error("applicant_explanation_failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Could not calculate the saved model explanation.",
        ) from exc

    predicted_class, provisional_classification = _classification(
        probability,
        model_version,
    )
    threshold = _threshold_for_model(model_version)
    model_features = model_feature_schema(model_version)[0]
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
            threshold,
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
                    threshold,
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
        "classification_threshold": threshold,
        "model_version": prediction_service.model_label(model_version),
        "top_factors": factors,
        "feature_details": _feature_details(factors, model_version),
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
    model_version = request.model_version
    unsupported = _validate_model_fields(request.applicant, model_version)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported fields are not accepted for inference.",
                "unsupported_columns": unsupported,
            },
        )
    invalid_values = _invalid_model_values(request.applicant, model_version)
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more model fields have incompatible values.",
                "invalid_columns": invalid_values,
            },
        )

    model_features = model_feature_schema(model_version)[0]
    safe_applicant = {
        key: value
        for key, value in request.applicant.items()
        if key in set(model_features)
    }
    try:
        probability, _ = prediction_service.predict(safe_applicant, model_version)
        factors = prediction_service.explain(
            safe_applicant,
            top_n=10,
            model_version=model_version,
        )
    except Exception as exc:
        logger.error("applicant_chat_evidence_failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Could not calculate model evidence for this chat.",
        ) from exc

    predicted_class, provisional_classification = _classification(
        probability,
        model_version,
    )
    threshold = _threshold_for_model(model_version)
    details = _feature_details(factors, model_version)
    described_factors = [
        {
            **factor,
            **details[str(factor["feature"])],
        }
        for factor in factors
    ]
    prepared_features = prediction_service.prepare_input(
        safe_applicant,
        model_version,
    ).iloc[0]
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
        "classification_threshold": threshold,
        "model_version": prediction_service.model_label(model_version),
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
        "model_version": prediction_service.model_label(model_version),
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


@app.get("/", dependencies=[Depends(require_admin)])
def root() -> dict[str, str]:
    return {
        "name": "Drifting Oracle",
        "status": "running",
        "version": app.version,
    }


@app.get("/health", dependencies=[Depends(require_admin)])
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
        "model_thresholds": {
            "v1": CLASSIFICATION_THRESHOLD,
            "v2": V2_CLASSIFICATION_THRESHOLD,
        },
        "llm_configured": bool(GROQ_API_KEY or GROQ_API_KEY_2 or GEMINI_API_KEY),
        "groq_configured": bool(GROQ_API_KEY or GROQ_API_KEY_2),
        "groq_backup_configured": bool(GROQ_API_KEY_2),
        "gemini_configured": bool(GEMINI_API_KEY),
        "rbi_live_search_configured": bool(GEMINI_API_KEY),
    }


@app.get("/drift/status", dependencies=[Depends(require_admin)])
def drift_status(model_version: ModelVersion = "v1") -> dict[str, Any]:
    try:
        return drift_monitor.status(model_version)
    except DriftStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Drift monitoring storage is unavailable.",
        ) from exc


@app.get("/drift/report", dependencies=[Depends(require_admin)])
def drift_report(model_version: ModelVersion = "v1") -> dict[str, Any]:
    try:
        return drift_monitor.report(model_version)
    except DriftStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Drift monitoring storage is unavailable.",
        ) from exc


@app.post("/drift/analyze", dependencies=[Depends(require_admin)])
def analyze_drift(request: DriftAnalyzeRequest) -> dict[str, Any]:
    try:
        return drift_monitor.analyze(
            request.model_version,
            request.window_days,
        )
    except DriftStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Drift monitoring storage is unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/drift/outcomes", dependencies=[Depends(require_admin)])
def record_drift_outcomes(
    request: DriftOutcomeSubmission,
) -> dict[str, Any]:
    try:
        result = drift_monitor.record_outcomes(
            [
                (item.prediction_id, item.actual_outcome)
                for item in request.outcomes
            ]
        )
    except DriftStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Drift monitoring storage is unavailable.",
        ) from exc
    return {
        **result,
        "message": (
            "Known prediction outcomes were recorded. Unknown prediction IDs "
            "were not changed."
        ),
    }


@app.post("/predict", dependencies=[Depends(require_admin)])
def predict(
    applicant: ApplicantInput,
    model_version: ModelVersion = "v1",
) -> dict[str, Any]:
    applicant_data = applicant.model_dump()
    unsupported = _validate_model_fields(applicant_data, model_version)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported fields are not accepted for inference.",
                "unsupported_columns": unsupported,
            },
        )
    invalid_values = _invalid_model_values(applicant_data, model_version)
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
            if key
            in set(model_feature_schema(model_version)[0])
            | {"application_id", "name"}
        }
        prediction_id = str(uuid.uuid4())
        probability, _ = prediction_service.predict(
            safe_applicant,
            model_version,
            prediction_id=prediction_id,
        )
        factors = prediction_service.explain(
            safe_applicant,
            top_n=5,
            model_version=model_version,
        )
    except Exception as exc:
        logger.error("single_prediction_failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed with the saved model.",
        ) from exc
    predicted_class, provisional_classification = _classification(
        probability,
        model_version,
    )
    threshold = _threshold_for_model(model_version)
    model_features = model_feature_schema(model_version)[0]
    supplied_features = sum(
        applicant_data.get(column) is not None
        for column in model_features
    )
    return {
        "prediction_id": prediction_id,
        "application_id": applicant.application_id,
        "name": applicant.name,
        "risk_probability": round(probability, 6),
        "predicted_class": predicted_class,
        "provisional_classification": provisional_classification,
        "classification_threshold": threshold,
        "model_version": prediction_service.model_label(model_version),
        "top_factors": factors,
        "feature_coverage": round(supplied_features / len(model_features), 4),
        "missing_model_features": len(model_features) - supplied_features,
        "warning": (
            "This single-applicant contract exposes only a partial feature "
            "set; missing model inputs are imputed and may reduce reliability."
        ),
    }


@app.post("/applications/explain", dependencies=[Depends(require_admin)])
def explain_applicant(request: ExplanationRequest) -> dict[str, Any]:
    return _explanation_payload(
        request.applicant,
        request.application_id,
        request.generate_narrative,
        request.provider_consent,
        request.model_version,
    )


@app.post("/applications/chat", dependencies=[Depends(require_admin)])
def applicant_chat(request: ApplicantChatRequest) -> dict[str, Any]:
    return _applicant_chat_payload(request)


@app.post("/assistant/rbi", dependencies=[Depends(require_admin)])
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


@app.get("/assistant/rbi/sources", dependencies=[Depends(require_admin)])
def rbi_sources() -> dict[str, Any]:
    return source_catalog()


@app.post(
    "/applications/batch/validate",
    response_model=BatchValidationResponse,
    dependencies=[Depends(require_admin)],
)
async def validate_batch(
    file: UploadFile = File(...),
    model_version: ModelVersion = "v1",
) -> dict[str, Any]:
    try:
        content = await _read_upload(file)
        parsed = parse_batch_upload(file.filename, content, model_version)
    except BatchFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return validation_payload(parsed)


@app.post("/predict/batch", dependencies=[Depends(require_admin)])
async def predict_batch(
    file: UploadFile = File(...),
    model_version: ModelVersion = "v1",
) -> dict[str, Any]:
    try:
        content = await _read_upload(file)
        parsed = parse_batch_upload(file.filename, content, model_version)
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

    batch_id = str(uuid.uuid4())
    response = score_batch(
        parsed,
        _threshold_for_model(parsed.model_version),
        batch_id=batch_id,
    )
    return response
