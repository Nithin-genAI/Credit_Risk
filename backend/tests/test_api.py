from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.batch_inference import model_feature_schema
from backend.services.prediction_service import prediction_service


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_external_providers_for_api_tests(monkeypatch):
    monkeypatch.setattr("backend.main.GROQ_API_KEY", "")
    monkeypatch.setattr("backend.main.GROQ_API_KEY_2", "")
    monkeypatch.setattr("backend.main.GEMINI_API_KEY", "")
    monkeypatch.setattr("backend.services.rbi_assistant.GEMINI_API_KEY", "")


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Drifting Oracle"


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
    assert result["predicted_class"] in {0, 1}
    assert result["provisional_classification"] in {
        "lower_default_risk",
        "elevated_default_risk",
    }


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