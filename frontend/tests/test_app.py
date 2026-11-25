from pathlib import Path
import secrets
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
_TEST_USERNAME = "test-" + secrets.token_hex(4)
_TEST_PASSWORD = secrets.token_urlsafe(16)


def _authenticated_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["admin_username"] = _TEST_USERNAME
    app.session_state["admin_access_token"] = "test-token"
    app.session_state["admin_expires_at"] = 4_000_000_000
    return app


def _health() -> dict:
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True,
        "expected_model_features": 120,
        "engineered_features": 10,
        "transformed_features": 254,
        "classification_threshold": 0.15,
        "model_thresholds": {"v1": 0.15, "v2": 0.22},
        "groq_configured": False,
        "gemini_configured": False,
        "rbi_live_search_configured": False,
    }


def _status() -> dict:
    return {
        "status": "Insufficient Data",
        "sample_count": 0,
        "window_days": 30,
        "compared_feature_count": 130,
    }


def _report() -> dict:
    return {
        "model_version": "v2",
        "status": "Insufficient Data",
        "sample_count": 0,
        "minimum_sample_size": 100,
        "window_days": 30,
        "window_start": None,
        "window_end": None,
        "reference_feature_count": 129,
        "compared_feature_count": 129,
        "reference_coverage": 1.0,
        "thresholds": {
            "psi_moderate": 0.1,
            "psi_critical": 0.2,
            "missing_rate_moderate_delta": 0.05,
            "missing_rate_critical_delta": 0.1,
        },
        "reasons": ["More predictions are required."],
        "feature_metrics": [
            {
                "feature": "AMT_CREDIT",
                "feature_type": "numeric",
                "status": "Moderate Drift",
                "psi": 0.15,
                "ks_statistic": 0.12,
                "ks_p_value": 0.03,
                "chi_square_p_value": None,
                "reference_distribution": {"low": 0.5, "high": 0.5},
                "current_distribution": {"low": 0.25, "high": 0.75},
                "reference_missing_rate": 0.0,
                "current_missing_rate": 0.01,
                "missing_rate_change": 0.01,
            }
        ],
        "performance": {
            "sample_count": 0,
            "actual_default_rate": None,
            "roc_auc": None,
            "brier_score": None,
            "precision": None,
            "recall": None,
            "note": "No actual outcomes.",
        },
    }


def test_drift_status_opens_versioned_report_dialog_and_close_returns_to_dashboard():
    status_request = Mock(return_value=_status())
    report_request = Mock(return_value=_report())
    with (
        patch("frontend.api_client.get_health", return_value=_health()),
        patch("frontend.api_client.get_drift_status", status_request),
        patch("frontend.api_client.get_drift_report", report_request),
    ):
        app = _authenticated_app().run()
        app.selectbox(key="drift_model_version").set_value("v2").run()

        assert status_request.call_args.args == ("http://127.0.0.1:8000", "v2")
        status_button = next(
            button
            for button in app.button
            if "View V2 report" in button.label
        )
        status_button.click().run(timeout=15)

        assert report_request.call_args.args == ("http://127.0.0.1:8000", "v2")
        assert any(button.label == "Analyze now" for button in app.button)
        assert any(button.label == "× Close" for button in app.button)
        assert any(
            selectbox.label == "Compare reference and current distributions"
            for selectbox in app.selectbox
        )
        assert app.dataframe
        assert not app.exception

        next(button for button in app.button if button.label == "× Close").click().run(
            timeout=15
        )
        assert not any(button.label == "Analyze now" for button in app.button)
        assert not app.exception


def test_admin_login_and_logout_gate_dashboard():
    login_request = Mock(
        return_value={
            "access_token": "authenticated-token",
            "token_type": "bearer",
            "expires_in": 3_600,
        }
    )
    with (
        patch("frontend.api_client.get_health", return_value=_health()),
        patch("frontend.api_client.get_drift_status", return_value=_status()),
        patch(
            "frontend.api_client.get_admin_users",
            return_value=[_TEST_USERNAME],
        ),
        patch("frontend.api_client.login_admin", login_request),
    ):
        app = AppTest.from_file(str(APP_PATH)).run()
        assert any("Admin sign in" in element.value for element in app.markdown)
        assert not app.sidebar.radio

        assert app.selectbox(key="admin_login_username").value == _TEST_USERNAME
        app.text_input(key="admin_login_password").set_value(_TEST_PASSWORD)
        next(button for button in app.button if button.label == "Log in").click().run()

        login_request.assert_called_once_with(
            "http://127.0.0.1:8000",
            _TEST_USERNAME,
            _TEST_PASSWORD,
        )
        assert app.session_state["admin_username"] == _TEST_USERNAME
        assert app.session_state["admin_access_token"] == "authenticated-token"
        assert "admin_login_password" not in app.session_state
        assert any(button.label == "Log out" for button in app.button)
        assert not app.exception

        next(button for button in app.button if button.label == "Log out").click().run()
        assert "admin_access_token" not in app.session_state
        assert any("Admin sign in" in element.value for element in app.markdown)
        assert not app.sidebar.radio
        assert not app.exception
