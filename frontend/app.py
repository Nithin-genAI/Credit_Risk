from __future__ import annotations

import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import (
    APIError,
    analyze_drift,
    applicant_record_from_mapping,
    applicant_chat_context_key,
    ask_regulatory_assistant,
    chat_about_applicant,
    explain_applicant,
    get_drift_report,
    get_drift_status,
    get_health,
    get_regulatory_sources,
    predict_batch,
    validate_batch_upload,
)


API_URL = os.getenv("DRIFTING_ORACLE_API_URL", "http://127.0.0.1:8000").rstrip("/")
VIEWS = [
    "Executive Overview",
    "Application Review",
    "Applicant Explanation",
    "RBI Regulatory Assistant",
]

st.set_page_config(
    page_title="Drifting Oracle | Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --paper:#F5F1E8; --card:#FFFCF5; --sand:#EAE4D7;
                --ink:#203446; --muted:#657681; --teal:#177E78;
                --coral:#C66B55; --line:#D8D0C1; }
        .stApp { background:
            radial-gradient(ellipse at 92% 4%,rgba(226,190,137,.18),transparent 28%),
            linear-gradient(145deg,#FBF8F0 0%,var(--paper) 54%,#F0EBDD 100%);
            color:var(--ink); }
        [data-testid="stHeader"] { background:rgba(245,241,232,.92); }
        [data-testid="stSidebar"] { background:#EAE4D7;
            border-right:1px solid var(--line); }
        [data-testid="stSidebar"] * { color:var(--ink); }
        h1,h2,h3,h4 { letter-spacing:-.025em; color:var(--ink); }
        p,label,.stMarkdown { color:#354B5A; }
        [data-testid="stMetric"] { background:var(--card);
            border:1px solid var(--line); border-radius:12px;
            padding:16px 18px; }
        .oracle-banner { border:1px solid #D8CDB9; border-radius:16px;
            padding:20px 24px; background:
            linear-gradient(112deg,rgba(228,238,224,.94),rgba(255,249,235,.98) 58%,rgba(243,224,208,.92));
            margin-bottom:18px; }
        .oracle-kicker { color:var(--teal); font-size:.77rem; font-weight:700;
            letter-spacing:.16em; text-transform:uppercase; }
        .oracle-subtitle { color:#526771; margin-top:6px; }
        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="stBaseButton-primary"] {
            background:#CBE7D3 !important; border:1px solid #4E8062 !important;
            color:#173B2D !important;
            font-weight:700; border-radius:8px; min-height:2.7rem; }
        div.stButton > button:not([kind="primary"]) {
            background:#FFFDF7; border:1px solid #CEC4B2; color:var(--ink);
            border-radius:8px; }
        [data-testid="stDataFrame"] { border:1px solid var(--line);
            border-radius:10px; background:#FFFDF8; }
        .small-note { color:var(--muted); font-size:.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    st.session_state.setdefault("view", VIEWS[0])
    pending_view = st.session_state.pop("pending_view", None)
    if pending_view in VIEWS:
        st.session_state["view"] = pending_view
    st.session_state.setdefault("session_assessments", [])
    st.session_state.setdefault("recent_batches", [])
    st.session_state.setdefault("batch_source_key", None)
    st.session_state.setdefault("batch_validation", None)
    st.session_state.setdefault("batch_source_bytes", None)
    st.session_state.setdefault("batch_source_name", None)
    st.session_state.setdefault("batch_model_version", None)
    st.session_state.setdefault("batch_results", [])
    st.session_state.setdefault("batch_summary", {})
    st.session_state.setdefault("batch_records", {})
    st.session_state.setdefault("current_explanation", None)
    st.session_state.setdefault("current_explanation_applicant", None)
    st.session_state.setdefault("current_explanation_model_version", "v1")
    st.session_state.setdefault("applicant_chat_histories", {})
    st.session_state.setdefault("rbi_response", None)
    st.session_state.setdefault("drift_model_version", "v1")
    st.session_state.setdefault("show_drift_report", False)


def sidebar() -> str:
    st.sidebar.markdown("### ◈ Drifting Oracle")
    st.sidebar.caption("MODEL RISK · V1 BASELINE")
    selected = st.sidebar.radio(
        "Workspace",
        VIEWS,
        key="view",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Prediction API")
    st.sidebar.code(API_URL, language=None)
    st.sidebar.caption(
        "V1 / V2 provisional thresholds · not lending policy"
    )
    return selected


def header() -> None:
    st.markdown(
        """
        <div class="oracle-banner">
          <div class="oracle-kicker">Drifting Oracle · Risk Intelligence</div>
          <h1 style="margin:5px 0 0">Credit Risk Intelligence Desk</h1>
          <div class="oracle-subtitle">Model-assisted review for analysts and loan officers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def api_health() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return get_health(API_URL), None
    except APIError as exc:
        return None, str(exc)


def render_overview() -> None:
    st.subheader("Executive Overview")
    st.caption(
        "Dashboard counts are session-only. For drift monitoring, the backend "
        "retains model features and probabilities locally for the configured "
        "retention period without names or applicant IDs."
    )
    health, health_error = api_health()
    assessments = st.session_state["session_assessments"]
    recent_batches = st.session_state["recent_batches"]
    total_submitted = sum(batch["total"] for batch in recent_batches)
    cards = st.columns(4)
    cards[0].metric(
        "API / model",
        "Ready" if health and health.get("status") == "healthy" else "Unavailable",
    )
    cards[1].metric("Applications assessed · session", len(assessments))
    cards[2].metric("Applications in batches · session", total_submitted)
    cards[3].metric(
        "Provisional threshold",
        f"{health.get('classification_threshold', 0.15):.2f}"
        if health
        else "0.15",
    )
    if health_error:
        st.error(health_error)
    elif health:
        st.caption(
            f"Model loaded: {health.get('model_loaded')} · "
            f"Preprocessor loaded: {health.get('preprocessor_loaded')} · "
            f"Raw model features: {health.get('expected_model_features', 'unknown')} · "
            f"Transformed inputs: {health.get('transformed_features', 'unknown')} · "
            f"Groq: {health.get('groq_configured', False)} · "
            f"Gemini: {health.get('gemini_configured', False)} · "
            "RBI search key present (availability not checked): "
            f"{health.get('rbi_live_search_configured', False)}"
        )

    st.markdown("#### Model drift monitoring")
    drift_model = st.selectbox(
        "Monitor model",
        ["v1", "v2"],
        format_func=lambda value: (
            "V1 — Baseline" if value == "v1" else "V2 — 129-feature model"
        ),
        key="drift_model_version",
    )
    try:
        drift_status = get_drift_status(API_URL, drift_model)
    except APIError as exc:
        st.error(f"Drift monitoring unavailable: {exc}")
    else:
        status_button = (
            f"{drift_status['status']} · View {drift_model.upper()} report"
        )
        if st.button(status_button, key="open_drift_report"):
            st.session_state["show_drift_report"] = True
            st.rerun()
        st.caption(
            f"{drift_status['sample_count']} predictions in the last "
            f"{drift_status['window_days']} days · "
            f"{drift_status['compared_feature_count']} reference features compared."
        )

    if st.session_state["show_drift_report"]:
        _render_drift_report_dialog()

    st.markdown("#### Saved baseline validation results · historical")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Baseline ROC-AUC", "0.8778")
    metric_cols[1].metric("Baseline Precision", "76.23%")
    st.warning(
        "These displayed baseline values are presentation figures and do not "
        "match the saved evaluation report. Historical validation data "
        "influenced early stopping, so it is not an independent final test. "
        "The 0.15 threshold is provisional and is not lending policy."
    )

    st.markdown("#### Session-only risk distribution")
    if assessments:
        distribution = (
            pd.Series(
                [row["provisional_classification"] for row in assessments]
            )
            .value_counts()
            .rename_axis("Provisional classification")
            .to_frame("Applications")
        )
        st.dataframe(distribution, width="stretch")
    else:
        st.info("No applications have been assessed in this session.")

    st.markdown("#### Recent batch activity · session only")
    if recent_batches:
        st.dataframe(
            pd.DataFrame(recent_batches).head(5),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No batch assessments have been run in this session.")

    actions = st.columns(2)
    if actions[0].button("Review applications", type="primary", width="stretch"):
        st.session_state["pending_view"] = "Application Review"
        st.rerun()
    if actions[1].button("Open RBI assistant", width="stretch"):
        st.session_state["pending_view"] = "RBI Regulatory Assistant"
        st.rerun()


@st.dialog("Detailed drift report", width="large")
def _render_drift_report_dialog() -> None:
    model_version = st.session_state["drift_model_version"]
    st.caption(
        f"Model: {model_version.upper()} · Monitoring uses only this model's "
        "saved reference state."
    )
    report = None
    if st.button("Analyze now", key="analyze_drift_now", type="primary"):
        try:
            with st.spinner("Comparing the monitoring window with reference state…"):
                report = analyze_drift(API_URL, model_version)
        except APIError as exc:
            st.error(str(exc))
    if report is None:
        try:
            report = get_drift_report(API_URL, model_version)
        except APIError as exc:
            st.error(str(exc))
            report = None

    if report is not None:
        summary = st.columns(3)
        summary[0].metric("Drift status", report["status"])
        summary[1].metric("Window predictions", report["sample_count"])
        summary[2].metric(
            "Reference coverage",
            f"{report['compared_feature_count']} / "
            f"{report['reference_feature_count']}",
        )
        if report["reasons"]:
            for reason in report["reasons"]:
                st.info(reason)
        st.caption(
            f"Window: {report['window_start'] or 'no predictions'} → "
            f"{report['window_end'] or 'no predictions'} · "
            f"PSI thresholds: moderate ≥ "
            f"{report['thresholds']['psi_moderate']:.2f}, drifted ≥ "
            f"{report['thresholds']['psi_critical']:.2f}. "
            "KS values are approximate because the saved reference is quantile-based."
        )

        performance = report["performance"]
        st.markdown("#### Prediction and outcome monitoring")
        performance_cols = st.columns(5)
        performance_cols[0].metric(
            "Probability PSI",
            next(
                (
                    f"{metric['psi']:.4f}"
                    for metric in report["feature_metrics"]
                    if metric["feature"] == "prediction_probability"
                    and metric["psi"] is not None
                ),
                "—",
            ),
        )
        performance_cols[1].metric(
            "Labeled outcomes",
            performance["sample_count"],
        )
        performance_cols[2].metric(
            "ROC-AUC",
            f"{performance['roc_auc']:.4f}"
            if performance["roc_auc"] is not None
            else "—",
        )
        performance_cols[3].metric(
            "Brier score",
            f"{performance['brier_score']:.4f}"
            if performance["brier_score"] is not None
            else "—",
        )
        performance_cols[4].metric(
            "Default rate",
            f"{performance['actual_default_rate']:.2%}"
            if performance["actual_default_rate"] is not None
            else "—",
        )
        if performance["note"]:
            st.caption(performance["note"])

        metrics = report["feature_metrics"]
        affected = [
            item
            for item in metrics
            if item["status"] in {"Moderate Drift", "Oracle Drifted"}
        ]
        st.markdown("#### Affected features")
        if affected:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": item["feature"],
                            "Type": item["feature_type"],
                            "Status": item["status"],
                            "PSI": item["psi"],
                            "Reference missing": item["reference_missing_rate"],
                            "Current missing": item["current_missing_rate"],
                            "Missing change": item["missing_rate_change"],
                            "KS p-value": item.get("ks_p_value"),
                            "Chi-square p-value": item.get("chi_square_p_value"),
                        }
                        for item in affected
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        elif report["sample_count"] >= report["minimum_sample_size"]:
            st.success("No features exceed the configured drift thresholds.")
        else:
            st.info(
                "More predictions are required before the drift status can be "
                "classified reliably."
            )

        available_features = [
            item for item in metrics if item["reference_distribution"]
        ]
        if available_features:
            selected_feature = st.selectbox(
                "Compare reference and current distributions",
                [item["feature"] for item in available_features],
                key="drift_feature_detail",
            )
            selected_metric = next(
                item
                for item in available_features
                if item["feature"] == selected_feature
            )
            buckets = sorted(
                set(selected_metric["reference_distribution"])
                | set(selected_metric["current_distribution"])
            )
            distribution = pd.DataFrame(
                {
                    "Reference": [
                        selected_metric["reference_distribution"].get(bucket, 0.0)
                        for bucket in buckets
                    ],
                    "Current": [
                        selected_metric["current_distribution"].get(bucket, 0.0)
                        for bucket in buckets
                    ],
                },
                index=buckets,
            )
            st.caption(
                f"PSI: {selected_metric['psi']:.4f} · "
                f"Reference missing: "
                f"{selected_metric['reference_missing_rate']:.2%} · "
                f"Current missing: "
                f"{selected_metric['current_missing_rate']:.2%}"
                if selected_metric["psi"] is not None
                and selected_metric["current_missing_rate"] is not None
                else f"Feature status: {selected_metric['status']}"
            )
            st.bar_chart(distribution)

    if st.button("× Close", key="close_drift_report"):
        st.session_state["show_drift_report"] = False
        st.rerun()


def _preview_frame(content: bytes, filename: str) -> pd.DataFrame:
    if Path(filename).suffix.casefold() == ".xlsx":
        return pd.read_excel(BytesIO(content), nrows=8, engine="openpyxl")
    return pd.read_csv(BytesIO(content), nrows=8)


def _reset_batch_for_new_file(source_key: str) -> None:
    if source_key == st.session_state["batch_source_key"]:
        return
    st.session_state["batch_source_key"] = source_key
    st.session_state["batch_validation"] = None
    st.session_state["batch_source_bytes"] = None
    st.session_state["batch_source_name"] = None
    st.session_state["batch_model_version"] = None
    st.session_state["batch_results"] = []
    st.session_state["batch_summary"] = {}
    st.session_state["batch_records"] = {}


def _validate_current_upload(
    upload: Any,
    content: bytes,
    model_version: str,
) -> None:
    try:
        with st.spinner("Checking file size, schema, values, and row limit…"):
            response = validate_batch_upload(
                API_URL,
                content,
                upload.name,
                model_version,
            )
        st.session_state["batch_validation"] = response
        st.session_state["batch_source_bytes"] = content
        st.session_state["batch_source_name"] = upload.name
        st.session_state["batch_model_version"] = model_version
    except APIError as exc:
        st.session_state["batch_validation"] = None
        st.error(str(exc))


def _run_current_batch() -> None:
    content = st.session_state["batch_source_bytes"]
    filename = st.session_state["batch_source_name"]
    model_version = st.session_state["batch_model_version"] or "v1"
    if content is None or filename is None:
        st.error("Validate the uploaded file before running the assessment.")
        return
    try:
        with st.spinner("Scoring applicants with the saved model…"):
            result = predict_batch(API_URL, content, filename, model_version)
        frame = pd.read_csv(BytesIO(content)) if Path(filename).suffix.casefold() == ".csv" else pd.read_excel(BytesIO(content), engine="openpyxl")
        results = result["results"]
        records = {
            int(prediction["row_number"]): row.to_dict()
            for prediction, (_, row) in zip(results, frame.iterrows())
        }
        st.session_state["batch_results"] = results
        st.session_state["batch_summary"] = result
        st.session_state["batch_records"] = records
        st.session_state["session_assessments"].extend(
            row
            for row in results
            if row.get("status") == "scored"
        )
        st.session_state["recent_batches"].insert(
            0,
            {
                "batch_id": result.get("batch_id", "session"),
                "total": result["total_applications"],
                "scored": result["successful_predictions"],
                "failed": result["failed_predictions"],
            },
        )
        st.success("Batch assessment completed. Results exist only in this session.")
    except (APIError, OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Batch assessment failed: {exc}")


def _render_batch_results() -> None:
    results = st.session_state["batch_results"]
    if not results:
        return
    summary = st.session_state["batch_summary"]
    st.caption(f"Model used: {summary.get('model_version', 'baseline-v1')}")
    counts = st.columns(3)
    counts[0].metric("Rows", summary.get("total_applications", len(results)))
    counts[1].metric("Scored", summary.get("successful_predictions", 0))
    counts[2].metric("Row errors", summary.get("failed_predictions", 0))

    result_frame = pd.DataFrame(results)
    search = st.text_input("Search results by applicant reference")
    if search:
        result_frame = result_frame[
            result_frame["application_id"].astype(str).str.contains(
                search,
                case=False,
                na=False,
            )
        ]
    display_columns = [
        column
        for column in (
            "row_number",
            "application_id",
            "prediction_id",
            "model_version",
            "risk_probability",
            "predicted_class",
            "provisional_classification",
            "status",
            "error",
        )
        if column in result_frame
    ]
    st.dataframe(
        result_frame[display_columns],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Export visible results as CSV",
        data=result_frame[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="drifting_oracle_batch_results.csv",
        mime="text/csv",
    )
    scored = result_frame[result_frame["status"] == "scored"]
    if scored.empty:
        return

    options = {}
    for row in scored.itertuples():
        row_number = str(row.row_number)
        if row_number.isdecimal():
            options[f"{row.application_id} · row {row_number}"] = int(row_number)
    if not options:
        st.info("No valid scored row numbers are available to inspect.")
        return
    choice = st.selectbox("Choose an application to inspect", list(options))
    if choice is None:
        return
    if st.button("Inspect selected application", type="primary"):
        row_number = options[choice]
        row_data = st.session_state["batch_records"].get(row_number, {})
        applicant = applicant_record_from_mapping(row_data)
        application_id = str(
            row_data.get("application_id")
            or row_data.get("Application ID")
            or row_data.get("SK_ID_CURR")
            or choice.split(" · row ", maxsplit=1)[0]
        )
        try:
            with st.spinner("Recomputing local model and SHAP evidence…"):
                explanation = explain_applicant(
                    API_URL,
                    application_id,
                    applicant,
                    model_version=st.session_state["batch_model_version"] or "v1",
                )
            st.session_state["current_explanation"] = explanation
            st.session_state["current_explanation_applicant"] = applicant
            st.session_state["current_explanation_model_version"] = (
                st.session_state["batch_model_version"] or "v1"
            )
            st.session_state["pending_view"] = "Applicant Explanation"
            st.rerun()
        except APIError as exc:
            st.error(str(exc))


def render_application_review() -> None:
    st.subheader("Application Review & Batch Processing")
    model_version = st.selectbox(
        "Prediction model",
        ["v1", "v2"],
        format_func=lambda value: (
            "V1 — Baseline" if value == "v1" else "V2 — 129-feature model"
        ),
        key="model_version_selector",
    )
    health, _ = api_health()
    if health:
        threshold = health.get("model_thresholds", {}).get(model_version)
        if threshold is not None:
            st.caption(
                f"Selected model's provisional threshold: {threshold:.2f} · "
                "demonstration setting, not lending policy."
            )
    st.write(
        "Upload a CSV or XLSX using the saved model's feature contract. The API "
        "reports missing and unsupported columns before scoring; optional missing "
        "inputs are imputed by the saved preprocessor and clearly disclosed."
    )
    upload = st.file_uploader(
        "Applicant file",
        type=["csv", "xlsx"],
        key="batch_upload",
    )
    if upload is not None:
        content = upload.getvalue()
        source_key = hashlib.sha256(
            f"{upload.name}:{len(content)}:{model_version}:".encode("utf-8")
            + content
        ).hexdigest()
        _reset_batch_for_new_file(source_key)
        st.caption(
            f"{Path(upload.name).name} · {len(content) / 1024:.1f} KB"
        )
        if st.button("Validate file", type="primary"):
            _validate_current_upload(upload, content, model_version)

        validation = st.session_state["batch_validation"]
        if validation:
            st.markdown("#### File validation")
            status_cols = st.columns(3)
            status_cols[0].metric("Rows", validation["row_count"])
            status_cols[1].metric(
                "Recognized model columns",
                f"{validation['recognized_model_column_count']} / "
                f"{validation['expected_model_column_count']}",
            )
            status_cols[2].metric("Invalid rows", validation["row_issue_count"])
            if validation["target_removed"]:
                st.warning(
                    "`TARGET` was detected. The backend excludes it from "
                    "inference and uses it for no prediction."
                )
            if validation["missing_required_columns"]:
                st.error(
                    "Missing required columns: "
                    + ", ".join(validation["missing_required_columns"])
                )
            if validation["missing_optional_model_columns"]:
                st.warning(
                    f"{len(validation['missing_optional_model_columns'])} optional "
                    "model columns are absent. Predictions will use the saved "
                    "imputer and are partial-feature assessments."
                )
            if validation["unsupported_columns"]:
                st.warning(
                    "Unsupported columns will be excluded: "
                    + ", ".join(validation["unsupported_columns"])
                )
            if validation["duplicate_columns"]:
                st.error(
                    "Duplicate columns must be resolved: "
                    + ", ".join(validation["duplicate_columns"])
                )
            for warning in validation["warnings"]:
                st.info(warning)
            if validation["row_issues"]:
                st.warning(
                    f"{validation['row_issue_count']} row(s) contain invalid "
                    "values and will return explicit row-level errors."
                )
                st.dataframe(
                    pd.DataFrame(validation["row_issues"]),
                    width="stretch",
                    hide_index=True,
                )
            if st.button(
                "Run Risk Assessment",
                type="primary",
                disabled=bool(
                    validation["missing_required_columns"]
                    or validation["duplicate_columns"]
                ),
            ):
                _run_current_batch()

        with st.expander("Preview uploaded rows", expanded=False):
            try:
                preview = _preview_frame(content, upload.name)
                preview = preview.drop(columns=["TARGET"], errors="ignore")
                st.dataframe(preview, width="stretch", hide_index=True)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.warning(f"Could not show file preview: {exc}")
    else:
        st.info("Choose a CSV or XLSX file to begin validation.")
    _render_batch_results()
    st.caption(
        "Batch results and counts are session-only. They are not saved as "
        "historical records."
    )


def render_explanation() -> None:
    st.subheader("Applicant Explanation & Insights")
    result = st.session_state["current_explanation"]
    if not result:
        st.info(
            "Choose Inspect on a scored application in Application Review to "
            "open its explanation."
        )
        return
    model_version = st.session_state.get(
        "current_explanation_model_version",
        "v1",
    )
    st.caption(f"Applicant reference: {result['application_id']}")
    cards = st.columns(3)
    cards[0].metric(
        "Predicted default probability",
        f"{result['risk_probability']:.2%}",
    )
    cards[1].metric("Predicted class", str(result["predicted_class"]))
    cards[2].metric(
        "Provisional classification",
        result["provisional_classification"].replace("_", " ").title(),
    )
    st.caption(
        f"Threshold: {result['classification_threshold']:.2f} · "
        f"Model: {result['model_version']} · "
        f"Supplied features: {100 * result['feature_coverage']:.1f}% "
        f"({result['missing_model_features']} missing)"
    )
    missing_feature_names = result.get("missing_feature_names", [])
    if missing_feature_names:
        with st.expander("Missing or imputed model fields", expanded=False):
            st.write(", ".join(missing_feature_names))

    narrative = result["narrative"]
    st.markdown("#### Plain-language explanation")
    if narrative.get("text"):
        st.info(narrative["text"])
        if narrative.get("message"):
            if "tell the developer" in narrative["message"].casefold():
                st.error(narrative["message"])
            else:
                st.caption(narrative["message"])
    elif narrative.get("status") == "not_requested":
        health, _ = api_health()
        if health and health.get("llm_configured"):
            consent_key = "explanation_ai_consent_" + hashlib.sha256(
                json.dumps(
                    {
                        "application_id": result["application_id"],
                        "risk_probability": result["risk_probability"],
                        "top_factors": result["top_factors"],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:12]
            st.caption(
                "Optional AI narrative: only the probability, provisional class, "
                "threshold, and top SHAP factors are sent to the configured "
                "provider. Applicant ID, name, and raw feature row are not sent."
            )
            consent = st.checkbox(
                "I understand and authorize sending this model evidence to the configured provider.",
                key=consent_key,
            )
            if st.button(
                "Generate grounded AI explanation",
                disabled=not consent,
            ):
                try:
                    with st.spinner("Generating explanation from model evidence…"):
                        updated = explain_applicant(
                            API_URL,
                            result["application_id"],
                            st.session_state["current_explanation_applicant"] or {},
                            generate_narrative=True,
                            provider_consent=True,
                            model_version=model_version,
                        )
                    st.session_state["current_explanation"] = updated
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
        else:
            st.info(
                "The optional AI provider is not configured. Local SHAP "
                "evidence remains available."
            )
    else:
        st.warning(
            narrative.get("message")
            or "No generated narrative is available. Review the local SHAP evidence below."
        )

    st.markdown("#### Local SHAP factors")
    factors = pd.DataFrame(result["top_factors"])
    if not factors.empty:
        feature_details = result.get("feature_details", {})
        factors["Meaning (where documented)"] = factors["feature"].map(
            lambda feature: feature_details.get(feature, {}).get(
                "description",
                "No verified description is available in project metadata.",
            )
        )
        factors["Units"] = factors["feature"].map(
            lambda feature: feature_details.get(feature, {}).get("units")
            or "Not documented"
        )
        factors["direction"] = factors["impact"].map(
            {
                "increases_risk": "Pushes toward TARGET = 1",
                "decreases_risk": "Pushes away from TARGET = 1",
            }
        )
        factors = factors.rename(
            columns={
                "feature": "Model feature",
                "direction": "Direction",
                "shap_value": "SHAP value (raw-score scale)",
            }
        )
        st.dataframe(
            factors[
                [
                    "Model feature",
                    "Meaning (where documented)",
                    "Units",
                    "Direction",
                    "SHAP value (raw-score scale)",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    st.warning(
        "SHAP contributions explain the model's raw score, not probability-point "
        "changes or real-world causes. This is not an approval or rejection "
        "recommendation. Missing inputs may reduce reliability."
    )

    st.markdown("#### Ask about this applicant's model result")
    applicant = st.session_state["current_explanation_applicant"] or {}
    context_key = applicant_chat_context_key(
        result["application_id"],
        applicant,
        model_version,
    )
    chat_store = st.session_state["applicant_chat_histories"]
    turns = chat_store.setdefault(context_key, [])
    health, _ = api_health()
    llm_configured = bool(health and health.get("llm_configured"))
    consent = False
    if llm_configured:
        st.caption(
            "With consent, the question, recent chat messages, probability, "
            "provisional class, top SHAP factors, descriptions, and only the "
            "raw values for those top factors may be sent to the configured "
            "backend provider. Applicant identifiers, name, and the complete "
            "raw feature row are excluded. "
            "Do not add private details to your question. Leave consent off to "
            "get a local answer from model evidence only."
        )
        consent = st.checkbox(
            "I authorize sending this question and minimized explanation evidence to the AI provider.",
            key=f"applicant_chat_consent_{context_key[:12]}",
        )
        if health and health.get("groq_backup_configured"):
            st.caption("A backup Groq API key is configured for quota failover.")
    audience = st.radio(
        "Explanation style",
        ["Applicant", "Loan officer"],
        horizontal=True,
        key=f"applicant_chat_audience_{context_key[:12]}",
    )
    for turn in turns:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("provider_message"):
                if "tell the developer" in turn["provider_message"].casefold():
                    st.error(turn["provider_message"])
                elif "output-token limit" in turn["provider_message"].casefold():
                    st.warning(turn["provider_message"])
                else:
                    st.caption(turn["provider_message"])
    with st.form(f"applicant_chat_form_{context_key[:12]}"):
        question = st.text_input(
            "Ask a follow-up question",
            max_chars=1_000,
            placeholder="For example: Which factors pushed the model result upward?",
        )
        submitted = st.form_submit_button(
            "Ask about this result",
            type="primary",
        )
    if submitted:
        if len(question.strip()) < 3:
            st.warning("Enter a question with at least 3 characters.")
        else:
            history = [
                {"role": turn["role"], "content": turn["content"]}
                for turn in turns[-8:]
            ]
            try:
                with st.spinner("Reviewing this applicant's local model evidence…"):
                    answer = chat_about_applicant(
                        API_URL,
                        applicant,
                        question.strip(),
                        "applicant" if audience == "Applicant" else "loan_officer",
                        history=history,
                        provider_consent=consent,
                        model_version=model_version,
                    )
                turns.append({"role": "user", "content": question.strip()})
                turns.append(
                    {
                        "role": "assistant",
                        "content": answer["answer"],
                        "provider_message": answer.get("provider_message"),
                    }
                )
                st.session_state["applicant_chat_histories"][context_key] = turns
                st.rerun()
            except APIError as exc:
                st.error(str(exc))

    report = {
        "application_id": result["application_id"],
        "risk_probability": result["risk_probability"],
        "predicted_class": result["predicted_class"],
        "provisional_classification": result["provisional_classification"],
        "classification_threshold": result["classification_threshold"],
        "model_version": result["model_version"],
        "top_factors": result["top_factors"],
        "feature_details": result.get("feature_details", {}),
        "missing_feature_names": missing_feature_names,
        "narrative": narrative,
        "limitations": result["warning"],
    }
    st.download_button(
        "Download explanation report",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name=f"{result['application_id']}_risk_explanation.json",
        mime="application/json",
    )


SUGGESTED_QUESTIONS = [
    "What identity checks does RBI's KYC direction describe?",
    "What does the RBI digital lending circular say about borrower protection?",
    "What does RBI say about returning loan documents after repayment?",
]


def _set_rbi_question(question: str) -> None:
    st.session_state["rbi_question"] = question


def render_rbi_assistant() -> None:
    st.subheader("RBI Regulatory Assistant")
    st.write(
        "Relevant local RBI excerpts are checked first. When evidence is missing "
        "or currentness needs verification, Gemini Google Search grounding can "
        "look for current official RBI sources if configured."
    )
    health, _ = api_health()
    llm_configured = bool(health and health.get("llm_configured"))
    live_search_configured = bool(
        health and health.get("rbi_live_search_configured")
    )
    st.caption(
        "Gemini search key present: "
        + ("yes" if live_search_configured else "no")
        + " · Provider availability is checked when a question is submitted."
    )
    if llm_configured:
        st.info(
            "When you submit a question, its text and retrieved public RBI "
            "excerpts are sent to the configured AI provider. Do not include "
            "applicant names, IDs, account numbers, or private financial details."
        )
    try:
        catalog = get_regulatory_sources(API_URL)
    except APIError as exc:
        st.error(f"Could not load RBI source references: {exc}")
        catalog = {"sources": []}
    sources = catalog.get("sources", [])
    if not sources or not any(source.get("indexed") for source in sources):
        st.warning(
            "No source pages are indexed yet. Run "
            "`python scripts/ingest_rbi_sources.py` from the repository root "
            "to fetch only the official RBI URLs in the source catalog."
        )
    with st.expander("Curated official RBI references", expanded=False):
        for source in sources:
            status = source.get("status_note", "")
            indexed = "Indexed" if source.get("indexed") else "Not indexed"
            updated = source.get("updated_as_of") or source.get("issued_on") or "Date not listed"
            st.link_button(str(source["title"]), str(source["url"]))
            st.caption(f"{updated} · {indexed}. {status}")
    audience = st.radio(
        "Audience",
        ["Applicant", "Loan officer"],
        horizontal=True,
    )
    st.caption("Suggested questions")
    suggestions = st.columns(3)
    for column, suggestion in zip(suggestions, SUGGESTED_QUESTIONS):
        column.button(
            suggestion,
            width="stretch",
            on_click=_set_rbi_question,
            args=(suggestion,),
        )
    question = st.text_area(
        "Your question",
        max_chars=1_000,
        placeholder="For example: What should a borrower know about identity checks?",
        key="rbi_question",
    )
    provider_consent = True
    if llm_configured:
        consent_key = "rbi_ai_consent_" + hashlib.sha256(
            question.strip().encode("utf-8")
        ).hexdigest()[:12]
        provider_consent = st.checkbox(
            "I understand and authorize sending this question and public source excerpts to the configured provider.",
            key=consent_key,
        )
    if st.button(
        "Ask RBI assistant",
        type="primary",
    ):
        if len(question.strip()) < 3:
            st.warning("Enter a question with at least 3 characters.")
        elif llm_configured and not provider_consent:
            st.warning(
                "Authorize sending this question and public source excerpts "
                "to the configured provider, or disable provider configuration "
                "to use local-only answers."
            )
        else:
            try:
                with st.spinner(
                    "Checking indexed RBI sources and live official sources when configured…"
                ):
                    st.session_state["rbi_response"] = ask_regulatory_assistant(
                        API_URL,
                        question.strip(),
                        "applicant" if audience == "Applicant" else "loan_officer",
                        provider_consent=provider_consent,
                    )
            except APIError as exc:
                st.error(str(exc))
                st.session_state["rbi_response"] = None
    response = st.session_state["rbi_response"]
    if response:
        st.markdown("#### Answer")
        if (
            response.get("answer_status") == "quota_exhausted"
            or "tell the developer" in response["answer"].casefold()
        ):
            st.error(response["answer"])
        else:
            st.write(response["answer"])
        st.caption(f"Evidence status: {response['answer_status']}")
        if response.get("disclaimer"):
            st.info(response["disclaimer"])
        st.markdown("#### Official source references")
        if response["sources"]:
            for source in response["sources"]:
                version = (
                    source.get("updated_as_of")
                    or source.get("issued_on")
                    or "Date not verified in live search"
                )
                st.link_button(str(source["title"]), str(source["url"]))
                st.caption(
                    f"{version} · {source.get('relevant_section', 'Reference')} · "
                    f"{source.get('status_note', '')}"
                )
        else:
            st.caption("No source citations were retrieved.")
    st.warning(
        "Informational only. RBI requirements can depend on the regulated entity, "
        "product, facts, and later amendments. Bank policies may add requirements. "
        "This assistant does not make legal, compliance, approval, or rejection decisions."
    )


def main() -> None:
    load_theme()
    initialize_state()
    page = sidebar()
    header()
    if page == "Executive Overview":
        render_overview()
    elif page == "Application Review":
        render_application_review()
    elif page == "Applicant Explanation":
        render_explanation()
    else:
        render_rbi_assistant()


if __name__ == "__main__":
    main()
