from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from frontend.api_client import APIError, _json_request, sanitize_batch_csv


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
    assert dropped == ["TARGET", "SK_ID_CURR"]
    assert b"TARGET" not in payload
    assert b"SK_ID_CURR" not in payload
    assert b"AMT_CREDIT" in payload


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
