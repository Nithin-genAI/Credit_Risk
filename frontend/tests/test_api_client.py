from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from frontend.api_client import (
    APIError,
    _json_request,
    applicant_chat_context_key,
    applicant_record_from_mapping,
    chat_about_applicant,
    encode_batch_frame,
    sanitize_batch_csv,
)


def test_batch_sanitizer_removes_target_and_unknown_columns():
    frame = pd.DataFrame({
        "application_id": ["DEMO-001"],
        "name": ["Benchmark Applicant 01"],
        "AMT_INCOME_TOTAL": [100000],
        "AMT_CREDIT": [200000],
        "DAYS_BIRTH": [-12000],
        "TARGET": [1],
        "SK_ID_CURR": [123],
    })

    payload, target_removed, dropped = sanitize_batch_csv(frame)

    assert target_removed == 1
    assert dropped == ["TARGET"]
    assert b"TARGET" not in payload
    assert b"SK_ID_CURR" in payload
    assert b"AMT_CREDIT" in payload


def test_explanation_record_keeps_only_model_fields_and_excludes_target():
    applicant = applicant_record_from_mapping(
        {
            "Application ID": "APP-1",
            "AMT_INCOME_TOTAL": 100000,
            "DAYS_BIRTH": -12000,
            "TARGET": 1,
            "UNSUPPORTED": "ignore",
            "SK_ID_CURR": 123,
        }
    )

    assert applicant == {
        "AMT_INCOME_TOTAL": 100000,
        "DAYS_BIRTH": -12000,
    }


def test_xlsx_serializer_preserves_upload_extension():
    frame = pd.DataFrame(
        {
            "application_id": ["APP-1"],
            "AMT_INCOME_TOTAL": [100000],
            "AMT_CREDIT": [200000],
            "DAYS_BIRTH": [-12000],
        }
    )

    content, filename = encode_batch_frame(frame, "applications.xlsx")

    assert filename == "applications.xlsx"
    assert content[:2] == b"PK"


def test_batch_sanitizer_requires_backend_required_columns():
    with pytest.raises(ValueError, match="DAYS_BIRTH"):
        sanitize_batch_csv(pd.DataFrame({
            "AMT_INCOME_TOTAL": [100000],
            "AMT_CREDIT": [200000],
        }))


def test_batch_sanitizer_rejects_blank_required_values():
    with pytest.raises(ValueError, match="AMT_CREDIT.*row"):
        sanitize_batch_csv(pd.DataFrame({
            "AMT_INCOME_TOTAL": [100000],
            "AMT_CREDIT": [None],
            "DAYS_BIRTH": [-12000],
        }))


def test_json_request_does_not_hide_http_failures(monkeypatch):
    class FakeResponse:
        ok = False
        status_code = 503

        @staticmethod
        def json():
            return {"detail": "unavailable"}

    monkeypatch.setattr(requests, "request", Mock(return_value=FakeResponse()))

    with pytest.raises(APIError, match="HTTP 503"):
        _json_request("GET", "http://api.test/health")


def test_json_request_reports_network_errors(monkeypatch):
    monkeypatch.setattr(
        requests,
        "request",
        Mock(side_effect=requests.Timeout("timed out")),
    )

    with pytest.raises(APIError, match="Could not reach"):
        _json_request("GET", "http://api.test/health")


def test_json_request_rejects_non_object_payload(monkeypatch):
    class FakeResponse:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return ["unexpected"]

    monkeypatch.setattr(requests, "request", Mock(return_value=FakeResponse()))

    with pytest.raises(APIError, match="JSON object"):
        _json_request("GET", "http://api.test/health")


def test_applicant_chat_context_keys_are_record_specific():
    first = applicant_chat_context_key(
        "APP-1",
        {"AMT_INCOME_TOTAL": 100000, "DAYS_BIRTH": -12000},
    )
    second = applicant_chat_context_key(
        "APP-2",
        {"AMT_INCOME_TOTAL": 100000, "DAYS_BIRTH": -12000},
    )
    changed_record = applicant_chat_context_key(
        "APP-1",
        {"AMT_INCOME_TOTAL": 120000, "DAYS_BIRTH": -12000},
    )

    assert first != second
    assert first != changed_record
    assert first == applicant_chat_context_key(
        "APP-1",
        {"DAYS_BIRTH": -12000, "AMT_INCOME_TOTAL": 100000},
    )


def test_applicant_chat_client_excludes_identifiers_and_target(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "answer": "Local evidence answer.",
        "answer_status": "local_evidence_only",
        "evidence": {"top_factors": []},
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("frontend.api_client.requests.request", request)

    result = chat_about_applicant(
        "http://api.test",
        {
            "application_id": "PRIVATE-ID",
            "name": "PRIVATE-NAME",
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "DAYS_BIRTH": -12000,
            "TARGET": 1,
            "UNSUPPORTED": "discard",
        },
        "Why did this happen?",
        "applicant",
        history=[{"role": "user", "content": "Earlier question"}],
    )

    sent = request.call_args.kwargs["json"]
    assert result["answer_status"] == "local_evidence_only"
    assert set(sent["applicant"]) == {
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "DAYS_BIRTH",
    }
    assert "PRIVATE-ID" not in str(sent)
    assert "PRIVATE-NAME" not in str(sent)
    assert "TARGET" not in str(sent)
    assert sent["history"] == [{"role": "user", "content": "Earlier question"}]
