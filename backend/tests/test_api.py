import json
import secrets
import sqlite3
from contextlib import closing
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import bcrypt
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import admin_auth
from backend.services.batch_inference import model_feature_schema
from backend.services.drift_monitor import DriftMonitor
from backend.services.prediction_service import prediction_service


client = TestClient(app)
_TEST_USERNAME = "test-" + secrets.token_hex(4)
_TEST_PASSWORD = secrets.token_urlsafe(24)
_TEST_PASSWORD_HASH = bcrypt.hashpw(
    _TEST_PASSWORD.encode("utf-8"),
    bcrypt.gensalt(rounds=12),
).decode("ascii")
_TEST_TOKEN_SECRET = secrets.token_urlsafe(32)


@pytest.fixture(autouse=True)
def disable_external_providers_for_api_tests(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        admin_auth,
        "ADMIN_CREDENTIALS_JSON",
        json.dumps({_TEST_USERNAME: _TEST_PASSWORD_HASH}),
    )
    monkeypatch.setattr(
        admin_auth,
        "ADMIN_TOKEN_SECRET",
        _TEST_TOKEN_SECRET,
    )
    admin_auth.reset_login_limits()
    client.headers["Authorization"] = (
        f"Bearer {admin_auth.create_access_token(_TEST_USERNAME)}"
    )
    monitor = DriftMonitor(tmp_path / "monitor.sqlite3")
    monkeypatch.setattr("backend.services.drift_monitor.drift_monitor", monitor)
    monkeypatch.setattr("backend.services.batch_inference.drift_monitor", monitor)
    monkeypatch.setattr("backend.main.drift_monitor", monitor)
    monkeypatch.setattr("backend.main.GROQ_API_KEY", "")
    monkeypatch.setattr("backend.main.GROQ_API_KEY_2", "")
    monkeypatch.setattr("backend.main.GEMINI_API_KEY", "")
    monkeypatch.setattr("backend.services.rbi_assistant.GEMINI_API_KEY", "")
    yield monitor
    client.headers.pop("Authorization", None)
    admin_auth.reset_login_limits()


def test_admin_login_and_protected_api_access():
    anonymous_client = TestClient(app)

    users = anonymous_client.get("/auth/users")
    unauthorized = anonymous_client.get("/health")
    invalid_token = anonymous_client.get(
        "/health",
        headers={"Authorization": "Bearer invalid-token"},
    )
    login = anonymous_client.post(
        "/auth/login",
        json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD},
    )

    assert users.status_code == 200
    assert users.json() == {"usernames": [_TEST_USERNAME]}
    assert unauthorized.status_code == 401
    assert invalid_token.status_code == 401
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    protected = anonymous_client.get(
        "/health",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert protected.status_code == 200


def test_admin_login_rate_limits_repeated_failures():
    anonymous_client = TestClient(app)
    wrong_password = secrets.token_urlsafe(24)
    for _ in range(admin_auth.MAX_LOGIN_FAILURES):
        response = anonymous_client.post(
            "/auth/login",
            json={
                "username": _TEST_USERNAME,
                "password": wrong_password,
            },
        )
        assert response.status_code == 401

    limited = anonymous_client.post(
        "/auth/login",
        json={"username": _TEST_USERNAME, "password": _TEST_PASSWORD},
    )
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "900"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Drifting Oracle"


def test_drift_endpoints_and_api_browser_methods(
    disable_external_providers_for_api_tests,
):
    root = client.get("/")
    docs = client.get("/docs")
    openapi = client.get("/openapi.json")
    status = client.get("/drift/status?model_version=v1")
    report = client.get("/drift/report?model_version=v2")
    analysis = client.post(
        "/drift/analyze",
        json={"model_version": "v2", "window_days": 14},
    )

    assert root.status_code == 200
    assert root.json()["name"] == "Drifting Oracle"
    assert docs.status_code == 200
    assert openapi.status_code == 200
    assert "/drift/status" in openapi.json()["paths"]
    assert client.get("/predict").status_code == 405
    assert status.status_code == 200
    assert status.json()["status"] == "Insufficient Data"
    assert status.json()["model_version"] == "v1"
    assert report.status_code == 200
    assert report.json()["model_version"] == "v2"
    assert len(report.json()["feature_metrics"]) == 130
    assert analysis.status_code == 200
    assert analysis.json()["window_days"] == 14
    assert client.get("/drift/status?model_version=v3").status_code == 422


def test_predict_includes_shap_factors():

    response = client.post(
        "/predict",
        json={
            "application_id": "APP-10001",
            "name": "Rahul",
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -12000,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert "top_factors" in result
    assert len(result["top_factors"]) == 5
    assert all(
        set(factor) == {"feature", "impact", "shap_value"}
        for factor in result["top_factors"]
    )
    assert result["classification_threshold"] == 0.15
    assert len(result["prediction_id"]) == 36
    assert result["predicted_class"] in {0, 1}
    assert result["provisional_classification"] in {
        "lower_default_risk",
        "elevated_default_risk",
    }


def test_v2_predict_uses_matching_artifacts_threshold_and_shap():
    response = client.post(
        "/predict?model_version=v2",
        json={
            "application_id": "V2-APP-10001",
            "name": "Applicant",
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -12000,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == "v2"
    assert result["classification_threshold"] == 0.22
    assert result["predicted_class"] == int(result["risk_probability"] >= 0.22)
    assert len(result["top_factors"]) == 5

    bundle = prediction_service.get_model("v2")
    assert prediction_service.get_model("v2") is bundle
    assert bundle.model.n_features_in_ == len(bundle.feature_names)
    assert len(model_feature_schema("v2")[0]) == 120


def test_v2_batch_validation_and_scoring_use_selected_model_and_threshold(
    disable_external_providers_for_api_tests,
):
    monitor = disable_external_providers_for_api_tests
    content = (
        "application_id,AMT_INCOME_TOTAL,AMT_CREDIT,DAYS_BIRTH\n"
        "V2-ROW-A,100000,200000,-12000\n"
        "V2-ROW-B,120000,240000,-13000\n"
    )
    validation = client.post(
        "/applications/batch/validate?model_version=v2",
        files={"file": ("v2-applicants.csv", content, "text/csv")},
    )
    assert validation.status_code == 200
    assert validation.json()["model_version"] == "v2"
    assert validation.json()["expected_model_column_count"] == 120

    response = client.post(
        "/predict/batch?model_version=v2",
        files={"file": ("v2-applicants.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == "v2"
    assert result["classification_threshold"] == 0.22
    assert result["total_applications"] == 2
    assert result["successful_predictions"] == 2
    assert [row["application_id"] for row in result["results"]] == [
        "V2-ROW-A",
        "V2-ROW-B",
    ]
    assert all(row["model_version"] == "v2" for row in result["results"])
    assert len(
        {row["prediction_id"] for row in result["results"]}
    ) == 2
    with closing(sqlite3.connect(monitor.database_path)) as connection:
        rows = connection.execute(
            "SELECT model_version, batch_id, COUNT(*) "
            "FROM predictions GROUP BY model_version, batch_id"
        ).fetchall()
    assert rows == [("v2", result["batch_id"], 2)]


def test_v2_applicant_explanation_and_chat_keep_model_context():
    applicant = {
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 500000,
        "DAYS_BIRTH": -12000,
    }
    explanation = client.post(
        "/applications/explain",
        json={
            "application_id": "V2-APP-10002",
            "applicant": applicant,
            "model_version": "v2",
        },
    )
    assert explanation.status_code == 200
    explanation_result = explanation.json()
    assert explanation_result["model_version"] == "v2"
    assert explanation_result["classification_threshold"] == 0.22
    assert len(explanation_result["top_factors"]) == 5

    chat = client.post(
        "/applications/chat",
        json={
            "applicant": applicant,
            "question": "Explain this model estimate",
            "model_version": "v2",
        },
    )
    assert chat.status_code == 200
    chat_result = chat.json()
    assert chat_result["model_version"] == "v2"
    assert chat_result["evidence"]["classification_threshold"] == 0.22
    assert chat_result["answer_status"] == "local_evidence_only"


def test_predict_rejects_target_and_unsupported_fields():
    base = {
        "application_id": "APP-10002",
        "name": "Applicant",
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 500000,
        "DAYS_BIRTH": -12000,
    }
    for extra in ({"TARGET": 1}, {"UNSUPPORTED_FIELD": 1}):
        response = client.post("/predict", json={**base, **extra})
        assert response.status_code == 422
        assert response.json()["detail"]["unsupported_columns"]


def test_predict_rejects_incompatible_full_schema_values():
    response = client.post(
        "/predict",
        json={
            "application_id": "APP-10006",
            "name": "Applicant",
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -12000,
            "NAME_FAMILY_STATUS": 123,
        },
    )
    assert response.status_code == 422
    assert "NAME_FAMILY_STATUS" in response.json()["detail"]["invalid_columns"]


def test_predictions_log_sanitized_features_and_accept_later_outcomes(
    disable_external_providers_for_api_tests,
):
    monitor = disable_external_providers_for_api_tests
    response = client.post(
        "/predict",
        json={
            "application_id": "SENSITIVE-APP-ID",
            "name": "Sensitive Applicant",
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -12000,
        },
    )
    assert response.status_code == 200
    prediction_id = response.json()["prediction_id"]
    with closing(sqlite3.connect(monitor.database_path)) as connection:
        row = connection.execute(
            "SELECT predicted_at, model_version, probability, feature_values "
            "FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
    assert row is not None
    assert row[0]
    assert row[1] == "v1"
    assert 0 <= row[2] <= 1
    features = json.loads(row[3])
    assert features["AMT_CREDIT"] == 500000
    assert not {"application_id", "name", "SK_ID_CURR", "TARGET"} & set(features)
    assert "Sensitive Applicant" not in row[3]
    assert "SENSITIVE-APP-ID" not in row[3]

    outcome = client.post(
        "/drift/outcomes",
        json={
            "outcomes": [
                {"prediction_id": prediction_id, "actual_outcome": 1}
            ]
        },
    )
    assert outcome.status_code == 200
    assert outcome.json()["updated_count"] == 1
    assert client.get("/drift/report?model_version=v1").json()[
        "performance"
    ]["sample_count"] == 1


def test_prediction_logging_failure_does_not_fail_prediction(
    disable_external_providers_for_api_tests,
    tmp_path,
):
    monitor = disable_external_providers_for_api_tests
    blocked_path = tmp_path / "not_a_database"
    blocked_path.mkdir()
    monitor.database_path = blocked_path

    response = client.post(
        "/predict",
        json={
            "application_id": "APP-LOG-FAIL",
            "name": "Applicant",
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -12000,
        },
    )

    assert response.status_code == 200
    assert "risk_probability" in response.json()


def test_explanation_recomputes_model_evidence_and_returns_narrative(monkeypatch):
    factor = {
        "feature": "num__EXT_SOURCE_3",
        "impact": "increases_risk",
        "shap_value": 0.123,
    }
    monkeypatch.setattr(
        "backend.main.prediction_service.predict",
        Mock(return_value=(0.22, None)),
    )
    monkeypatch.setattr(
        "backend.main.prediction_service.explain",
        Mock(return_value=[factor]),
    )
    monkeypatch.setattr(
        "backend.main.explain_model_evidence",
        Mock(return_value=SimpleNamespace(
            status="not_configured",
            text=None,
            safe_message="LLM is not configured.",
        )),
    )
    response = client.post(
        "/applications/explain",
        json={
            "application_id": "APP-10003",
            "applicant": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
            },
            "generate_narrative": True,
            "provider_consent": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["application_id"] == "APP-10003"
    assert payload["risk_probability"] == 0.22
    assert payload["predicted_class"] == 1
    assert payload["classification_threshold"] == 0.15
    assert payload["top_factors"] == [factor]
    assert payload["narrative"]["status"] == "local_fallback"
    assert "22.0% default probability" in payload["narrative"]["text"]
    assert "num__EXT_SOURCE_3" in payload["narrative"]["text"]
    assert payload["narrative"]["message"] == "LLM is not configured."


def test_ai_requests_require_explicit_provider_consent(monkeypatch):
    monkeypatch.setattr("backend.main.GROQ_API_KEY", "configured")
    monkeypatch.setattr("backend.main.GROQ_API_KEY_2", "")
    explanation = client.post(
        "/applications/explain",
        json={
            "application_id": "APP-10005",
            "applicant": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
            },
            "generate_narrative": True,
        },
    )
    rbi = client.post(
        "/assistant/rbi",
        json={"question": "What is in the KYC direction?"},
    )

    assert explanation.status_code == 403
    assert rbi.status_code == 403


def test_applicant_chat_sends_only_selected_applicant_evidence(monkeypatch):
    factor = {
        "feature": "num__EXT_SOURCE_3",
        "impact": "increases_risk",
        "shap_value": 0.12345,
    }
    monkeypatch.setattr("backend.main.GROQ_API_KEY", "configured")
    monkeypatch.setattr(
        "backend.main.prediction_service.predict",
        Mock(return_value=(0.22, None)),
    )
    monkeypatch.setattr(
        "backend.main.prediction_service.explain",
        Mock(return_value=[factor]),
    )
    captured = {}

    def chat_answer(question, audience, evidence, history):
        captured.update(
            {
                "question": question,
                "audience": audience,
                "evidence": evidence,
                "history": history,
            }
        )
        return SimpleNamespace(
            status="ready",
            text="This answer only uses the supplied model evidence.",
            safe_message=None,
        )

    monkeypatch.setattr("backend.main.generate_applicant_chat_answer", chat_answer)
    response = client.post(
        "/applications/chat",
        json={
            "applicant": {
                "application_id": "PRIVATE-APPLICATION-ID",
                "name": "PRIVATE-APPLICANT-NAME",
                "SK_ID_CURR": 987654,
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
                "EXT_SOURCE_3": 0.42,
            },
            "question": "Why did the model assign this estimate?",
            "audience": "applicant",
            "history": [{"role": "user", "content": "Earlier question"}],
            "provider_consent": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer_status"] == (
        "answered_with_selected_applicant_evidence"
    )
    assert "PRIVATE-APPLICATION-ID" not in str(captured)
    assert "PRIVATE-APPLICANT-NAME" not in str(captured)
    assert "987654" not in str(captured)
    assert "180000" not in str(captured)
    assert captured["evidence"]["risk_probability"] == 0.22
    assert captured["evidence"]["classification_threshold"] == 0.15
    assert captured["evidence"]["top_factors"][0]["shap_value"] == 0.12345
    assert captured["evidence"]["top_factors"][0]["applicant_value"] == 0.42
    assert captured["evidence"]["top_factors"][0]["basis"] == (
        "dataset_field_name_and_project_notes"
    )
    assert "DAYS_EMPLOYED" in captured["evidence"]["missing_feature_names"]
    assert captured["history"] == [{"role": "user", "content": "Earlier question"}]


def test_applicant_chat_without_consent_uses_local_evidence(monkeypatch):
    factor = {
        "feature": "num__CREDIT_TO_INCOME",
        "impact": "increases_risk",
        "shap_value": 0.1,
    }
    monkeypatch.setattr("backend.main.GROQ_API_KEY", "configured")
    monkeypatch.setattr("backend.main.GROQ_API_KEY_2", "")
    monkeypatch.setattr(
        "backend.main.prediction_service.predict",
        Mock(return_value=(0.22, None)),
    )
    monkeypatch.setattr(
        "backend.main.prediction_service.explain",
        Mock(return_value=[factor]),
    )
    monkeypatch.setattr(
        "backend.main.generate_applicant_chat_answer",
        Mock(side_effect=AssertionError("No provider call without consent.")),
    )

    response = client.post(
        "/applications/chat",
        json={
            "applicant": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
            },
            "question": "Explain this risk estimate",
            "provider_consent": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer_status"] == "local_evidence_only"
    assert "SHAP describes" in response.json()["answer"]


def test_applicant_chat_rejects_target_input():
    response = client.post(
        "/applications/chat",
        json={
            "applicant": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
                "TARGET": 1,
            },
            "question": "Explain this risk estimate",
        },
    )

    assert response.status_code == 422
    assert "TARGET" in response.json()["detail"]["unsupported_columns"]


def test_explanation_rejects_target_input():
    response = client.post(
        "/applications/explain",
        json={
            "application_id": "APP-10004",
            "applicant": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
                "TARGET": 1,
            },
        },
    )
    assert response.status_code == 422
    assert "TARGET" in response.json()["detail"]["unsupported_columns"]


def test_batch_validation_and_scoring_keep_row_alignment_and_exclude_labels():
    content = (
        "application_id,AMT_INCOME_TOTAL,AMT_CREDIT,DAYS_BIRTH,TARGET,"
        "SK_ID_CURR,UNSUPPORTED\n"
        "ROW-A,100000,200000,-12000,1,123,ignored\n"
        "ROW-B,not-a-number,200000,-13000,0,456,ignored\n"
    )

    validation = client.post(
        "/applications/batch/validate",
        files={"file": ("applicants.csv", content, "text/csv")},
    )
    assert validation.status_code == 200
    validated = validation.json()
    assert validated["valid"] is False
    assert validated["row_count"] == 2
    assert validated["target_removed"] is True
    assert "UNSUPPORTED" in validated["unsupported_columns"]
    assert validated["row_issue_count"] == 1

    response = client.post(
        "/predict/batch",
        files={"file": ("applicants.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_applications"] == 2
    assert payload["successful_predictions"] == 1
    assert payload["failed_predictions"] == 1
    assert [row["application_id"] for row in payload["results"]] == [
        "ROW-A",
        "ROW-B",
    ]
    assert payload["results"][0]["status"] == "scored"
    assert payload["results"][0]["classification_threshold"] == 0.15
    assert payload["results"][1]["status"] == "ERROR"
    assert "TARGET" not in str(payload["results"])


def test_batch_validation_rejects_nonfinite_numeric_and_wrong_category_type():
    content = (
        "application_id,AMT_INCOME_TOTAL,AMT_CREDIT,DAYS_BIRTH,"
        "AMT_ANNUITY,NAME_FAMILY_STATUS\n"
        "ROW-A,100000,200000,-12000,inf,123\n"
    )
    response = client.post(
        "/applications/batch/validate",
        files={"file": ("invalid.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_issue_count"] == 1
    assert set(payload["row_issues"][0]["columns"]) == {
        "AMT_ANNUITY",
        "NAME_FAMILY_STATUS",
    }


def test_full_feature_batch_preserves_saved_pipeline_order_and_predictions():
    model_columns, _, categorical_columns = model_feature_schema()
    applicant = {
        column: (
            "unseen_category"
            if column in categorical_columns
            else 0.0
        )
        for column in model_columns
    }
    applicant.update(
        {
            "application_id": "FULL-ROW",
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 300000.0,
            "DAYS_BIRTH": -12000.0,
        }
    )
    expected_probability, _ = prediction_service.predict(applicant)
    content = pd.DataFrame([applicant]).to_csv(index=False).encode("utf-8")

    response = client.post(
        "/predict/batch",
        files={"file": ("full-features.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    row = response.json()["results"][0]
    assert row["status"] == "scored"
    assert row["feature_coverage"] == 1.0
    assert row["missing_model_features"] == 0
    assert abs(row["risk_probability"] - expected_probability) < 1e-6


def test_batch_rejects_missing_required_schema():
    response = client.post(
        "/predict/batch",
        files={
            "file": (
                "incomplete.csv",
                "AMT_CREDIT,DAYS_BIRTH\n200000,-12000\n",
                "text/csv",
            )
        },
    )
    assert response.status_code == 422
    assert "AMT_INCOME_TOTAL" in response.json()["detail"]["missing_required_columns"]


def test_batch_rejects_column_limit(monkeypatch):
    monkeypatch.setattr(
        "backend.services.batch_inference.MAX_UPLOAD_COLUMNS",
        3,
    )
    response = client.post(
        "/applications/batch/validate",
        files={
            "file": (
                "wide.csv",
                "AMT_INCOME_TOTAL,AMT_CREDIT,DAYS_BIRTH,EXTRA\n100000,200000,-12000,1\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "maximum of 3 columns" in response.json()["detail"]


def test_batch_rejects_upload_size_limit(monkeypatch):
    monkeypatch.setattr("backend.main.MAX_UPLOAD_BYTES", 4)
    response = client.post(
        "/applications/batch/validate",
        files={"file": ("large.csv", b"123456", "text/csv")},
    )

    assert response.status_code == 413


def test_batch_rejects_unsupported_file_type():
    response = client.post(
        "/applications/batch/validate",
        files={"file": ("applicants.xls", b"not supported", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_batch_accepts_xlsx_and_aliases():
    workbook_bytes = BytesIO()
    with pd.ExcelWriter(workbook_bytes, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Application ID": ["XLSX-01"],
                "AMT_INCOME_TOTAL": [100000],
                "AMT_CREDIT": [200000],
                "DAYS_BIRTH": [-12000],
            }
        ).to_excel(writer, index=False)
    response = client.post(
        "/applications/batch/validate",
        files={
            "file": (
                "applicants.xlsx",
                workbook_bytes.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["file_type"] == "xlsx"
    assert payload["recognized_model_column_count"] == 3