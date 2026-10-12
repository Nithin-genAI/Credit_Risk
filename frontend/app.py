from __future__ import annotations

import os
import random
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIError,
    get_health,
    predict_applicant,
    predict_batch,
    sanitize_batch_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "data" / "demo_applicants.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "baseline_reference_metrics.json"
SHAP_PATH = PROJECT_ROOT / "models" / "baseline_shap_importance.csv"
API_URL = os.getenv("DRIFTING_ORACLE_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEMO_SEED = 42
DEFAULT_FORM = {
    "application_id": "",
    "name": "",
    "AMT_INCOME_TOTAL": None,
    "AMT_CREDIT": None,
    "AMT_ANNUITY": 0.0,
    "AMT_GOODS_PRICE": 0.0,
    "CNT_CHILDREN": None,
    "CNT_FAM_MEMBERS": 1.0,
    "age_years": None,
    "employment_years": None,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "NAME_CONTRACT_TYPE": "Not provided",
}


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
                --coral:#C66B55; --gold:#B58A3A; --lilac:#8272A8;
                --line:#D8D0C1; }
        .stApp { background:
            radial-gradient(ellipse at 92% 4%,rgba(226,190,137,.20),transparent 28%),
            linear-gradient(145deg,#FBF8F0 0%,var(--paper) 54%,#F0EBDD 100%);
            color:var(--ink); }
        [data-testid="stHeader"] { background:rgba(245,241,232,.92); }
        [data-testid="stSidebar"] { background:#EAE4D7;
            border-right:1px solid var(--line); }
        [data-testid="stSidebar"] * { color:var(--ink); }
        h1,h2,h3,h4 { letter-spacing:-.025em; color:var(--ink); }
        p, label, .stMarkdown { color:#354B5A; }
        [data-testid="stMetric"] { background:var(--card);
            border:1px solid var(--line); border-radius:12px;
            padding:17px 20px; box-shadow:0 2px 8px rgba(48,55,47,.035); }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--ink); }
        .oracle-banner { border:1px solid #D8CDB9; border-radius:16px;
            padding:22px 26px; background:
            linear-gradient(112deg,rgba(228,238,224,.94),rgba(255,249,235,.98) 58%,rgba(243,224,208,.92));
            box-shadow:0 8px 24px rgba(57,67,56,.06); margin-bottom:20px; }
        .oracle-kicker { color:var(--teal); font-size:.77rem; font-weight:700;
            letter-spacing:.16em; text-transform:uppercase; }
        .oracle-subtitle { color:#526771; margin-top:6px; }
        .section-card { border:1px solid var(--line); border-radius:12px;
            padding:16px 18px; background:rgba(255,252,245,.84);
            margin:8px 0 16px; }
        .risk-pill { display:inline-block; padding:5px 11px; border-radius:99px;
            background:#E2EEE4; color:#286D61; font-weight:700; }
        div.stButton > button[kind="primary"] {
            background:var(--teal); border:1px solid #126963; color:#FFFDF7;
            font-weight:700; border-radius:8px; min-height:2.8rem; }
        div.stButton > button[kind="primary"]:hover {
            background:#236F6B; border-color:#236F6B; color:white; }
        div.stButton > button:not([kind="primary"]) {
            background:#FFFDF7; border:1px solid #CEC4B2; color:var(--ink);
            border-radius:8px; }
        div.stButton > button:not([kind="primary"]):hover {
            border-color:var(--coral); color:#8D4E40; }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background:#FFFDF8; border-color:#CEC4B2; color:var(--ink); }
        [data-testid="stDataFrame"] { border:1px solid var(--line);
            border-radius:10px; background:#FFFDF8; }
        [data-testid="stAlert"] { border-radius:10px; }
        a { color:#126F69; }
        .small-note { color:var(--muted); font-size:.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    for key, value in DEFAULT_FORM.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("assessment_history", [])
    st.session_state.setdefault("current_assessment", None)
    st.session_state.setdefault("demo_index", -1)
    st.session_state.setdefault("demo_random_count", 0)
    st.session_state.setdefault("employment_years_available", False)


def load_demo_records() -> pd.DataFrame:
    if not DEMO_PATH.exists():
        raise FileNotFoundError(
            "Demo records are not available. Generate them with "
            "`python scripts/create_demo_applicants.py`."
        )
    return pd.read_csv(DEMO_PATH)


def set_form_values(values: dict[str, Any]) -> None:
    for key, default in DEFAULT_FORM.items():
        value = values.get(key, default)
        if pd.isna(value):
            value = default
        st.session_state[key] = value
    for key in ("AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS",
                "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                "NAME_CONTRACT_TYPE"):
        st.session_state[f"{key}_available"] = not pd.isna(values.get(key))


def apply_demo_record(record: pd.Series, index: int) -> None:
    values = {str(key): value for key, value in record.to_dict().items()}
    values["age_years"] = -float(values["DAYS_BIRTH"]) / 365.25
    employed_days = values.get("DAYS_EMPLOYED")
    values["employment_years"] = (
        -float(employed_days) / 365.25
        if pd.notna(employed_days) and float(employed_days) != 365243
        else None
    )
    set_form_values(values)
    st.session_state["employment_years_available"] = (
        pd.notna(employed_days) and float(employed_days) != 365243
    )
    st.session_state["demo_index"] = index
    st.session_state["demo_loaded_notice"] = (
        f"Loaded sampled benchmark applicant {index + 1}."
    )


def demo_buttons() -> None:
    try:
        records = load_demo_records()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return

    first_row = st.columns(2)
    second_row = st.columns(2)
    if first_row[0].button(
        "Load Random Demo Applicant", width="stretch"
    ):
        index = random.Random(
            DEMO_SEED + st.session_state["demo_random_count"]
        ).randrange(len(records))
        st.session_state["demo_random_count"] += 1
        apply_demo_record(records.iloc[index], index)
        st.rerun()
    if first_row[1].button("Load Next Applicant", width="stretch"):
        index = (st.session_state["demo_index"] + 1) % len(records)
        apply_demo_record(records.iloc[index], index)
        st.rerun()
    if second_row[0].button(
        "Randomize Demo Selection", width="stretch"
    ):
        index = random.Random(
            DEMO_SEED + st.session_state["demo_random_count"]
        ).randrange(len(records))
        st.session_state["demo_random_count"] += 1
        apply_demo_record(records.iloc[index], index)
        st.rerun()
    if second_row[1].button("Reset Form", width="stretch"):
        for key in DEFAULT_FORM:
            st.session_state[key] = DEFAULT_FORM[key]
        st.session_state["employment_years_available"] = False
        for key in ("AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS",
                    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                    "NAME_CONTRACT_TYPE"):
            st.session_state[f"{key}_available"] = False
        st.session_state["current_assessment"] = None
        st.rerun()

    notice = st.session_state.pop("demo_loaded_notice", None)
    if notice:
        st.success(notice)
    st.caption(
        "Demo records are reproducible samples from the Home Credit benchmark "
        "dataset, not live or independently verified bank customer applications."
    )


def applicant_form() -> dict[str, Any] | None:
    submitted = False
    with st.form("applicant_assessment", border=True):
        st.subheader("Applicant identity")
        identity = st.columns(2)
        application_id = identity[0].text_input(
            "Application ID", key="application_id"
        )
        name = identity[1].text_input("Applicant name", key="name")

        st.subheader("Income and requested loan")
        financial = st.columns(2)
        income = financial[0].number_input(
            "Annual income", min_value=0.0, value=None, step=5000.0,
            key="AMT_INCOME_TOTAL",
        )
        credit = financial[1].number_input(
            "Requested credit", min_value=0.0, value=None, step=5000.0,
            key="AMT_CREDIT",
        )
        optional = st.columns(2)
        annuity_available = optional[0].checkbox(
            "Annuity provided", key="AMT_ANNUITY_available"
        )
        annuity = optional[0].number_input(
            "Loan annuity (optional)", min_value=0.0, step=500.0,
            key="AMT_ANNUITY", disabled=not annuity_available,
        )
        goods_available = optional[1].checkbox(
            "Goods price provided", key="AMT_GOODS_PRICE_available"
        )
        goods_price = optional[1].number_input(
            "Goods price (optional)", min_value=0.0, step=5000.0,
            key="AMT_GOODS_PRICE", disabled=not goods_available,
        )

        st.subheader("Employment and household")
        household = st.columns(2)
        age_years = household[0].number_input(
            "Age (years)", min_value=18.0, max_value=100.0,
            value=None, step=1.0, key="age_years",
        )
        employment_available = household[1].checkbox(
            "Employment duration available",
            key="employment_years_available",
        )
        employment_years = household[1].number_input(
            "Employment duration (years)", min_value=0.0, max_value=80.0,
            value=None, step=1.0, key="employment_years",
            disabled=not employment_available,
        )
        family = st.columns(2)
        children = family[0].number_input(
            "Number of children", min_value=0, value=None, step=1,
            key="CNT_CHILDREN",
        )
        family_available = st.checkbox(
            "Household size provided", key="CNT_FAM_MEMBERS_available"
        )
        family_members = family[1].number_input(
            "Household members (optional)", min_value=1.0, step=1.0,
            key="CNT_FAM_MEMBERS", disabled=not family_available,
        )

        st.subheader("External credit indicators")
        external = st.columns(3)
        scores = {}
        for col, field in zip(external, ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")):
            available = col.checkbox(
                f"{field} available",
                key=f"{field}_available",
            )
            score = col.number_input(
                field,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=field,
                disabled=not available,
            )
            scores[field] = score if available else None

        contract = st.selectbox(
            "Contract type",
            ["Not provided", "Cash loans", "Revolving loans"],
            key="NAME_CONTRACT_TYPE",
        )
        submitted = st.form_submit_button(
            "Assess Risk",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return None
    if not application_id.strip() or not name.strip():
        st.error("Enter both an application ID and applicant name.")
        return None
    if income is None or credit is None or income <= 0 or credit <= 0:
        st.error("Income and requested credit must both be greater than zero.")
        return None
    if age_years is None:
        st.error("Enter the applicant's age before requesting an assessment.")
        return None
    if children is None:
        st.error("Enter the applicant's number of children before requesting an assessment.")
        return None

    applicant = {
        "application_id": application_id.strip(),
        "name": name.strip(),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity if annuity_available else None,
        "AMT_GOODS_PRICE": goods_price if goods_available else None,
        "CNT_CHILDREN": int(children),
        "CNT_FAM_MEMBERS": family_members if family_available else None,
        "DAYS_BIRTH": -age_years * 365.25,
        "DAYS_EMPLOYED": (
            -employment_years * 365.25
            if employment_available and employment_years is not None
            else None
        ),
        **scores,
        "NAME_CONTRACT_TYPE": (
            None if contract == "Not provided" else contract
        ),
    }
    return applicant


def assess(applicant: dict[str, Any]) -> None:
    st.session_state["current_assessment"] = None
    try:
        with st.spinner("Scoring applicant and calculating explanations…"):
            result = predict_applicant(API_URL, applicant)
    except APIError as exc:
        st.error(str(exc))
        return

    result["input_feature_count"] = sum(
        value is not None
        for key, value in applicant.items()
        if key not in {"application_id", "name"}
    )
    st.session_state["current_assessment"] = result
    st.session_state["assessment_history"].append(result)
    st.success("Assessment completed by the V1 prediction API.")


def header() -> None:
    st.markdown(
        """
        <div class="oracle-banner">
          <div class="oracle-kicker">Drifting Oracle · Risk Intelligence</div>
          <h1 style="margin:5px 0 0">Credit Risk Intelligence Desk</h1>
          <div class="oracle-subtitle">A transparent interface for V1 model assessment and validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    st.sidebar.markdown("### ◈ Drifting Oracle")
    st.sidebar.caption("MODEL RISK · V1 BASELINE")
    selected = st.sidebar.radio(
        "Workspace",
        [
            "Executive Overview",
            "Applicant Assessment",
            "Assessment Details",
            "Batch Assessment",
            "Model Validation",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Prediction API")
    st.sidebar.code(API_URL, language=None)
    st.sidebar.caption("Display-only thresholds · not lending policy")
    return selected


def health_status() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return get_health(API_URL), None
    except APIError as exc:
        return None, str(exc)


def load_baseline_metrics() -> dict[str, float]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    required = {"roc_auc", "pr_auc", "mean", "p25", "p50", "p75"}
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise ValueError("Baseline metric artifact is missing required metrics.")
    if any(
        not isinstance(metrics[key], (int, float))
        for key in required
    ):
        raise ValueError("Baseline metric artifact contains invalid metric values.")
    return metrics


def render_overview() -> None:
    st.subheader("Executive Overview")
    st.caption(
        "Historical validation metrics are fixed evaluation results; "
        "the application counters below reflect only this live UI session."
    )
    health, health_error = health_status()
    try:
        metrics = load_baseline_metrics()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load baseline metric artifact: {exc}")
        return

    assessments = st.session_state["assessment_history"]
    successful = [
        item for item in assessments if item.get("risk_level") != "ERROR"
    ]
    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "API health",
        "Connected" if health and health.get("status") == "healthy" else "Offline",
        help=health_error or "GET /health",
    )
    metric_cols[1].metric("ROC-AUC · validation", f"{metrics['roc_auc']:.4f}")
    metric_cols = st.columns(2)
    metric_cols[0].metric("PR-AUC · validation", f"{metrics['pr_auc']:.4f}")
    metric_cols[1].metric("Assessments · this session", len(successful))
    if health_error:
        st.error(health_error)
    if health:
        st.caption(
            f"Model loaded: {health.get('model_loaded')} · "
            f"Preprocessor loaded: {health.get('preprocessor_loaded')} · "
            f"Expected raw inputs: {health.get('expected_features', 'unknown')}"
        )

    st.markdown("#### Live assessment distribution")
    if successful:
        counts = (
            pd.Series([item.get("risk_level", "UNKNOWN") for item in successful])
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
            .rename_axis("Display risk band")
            .to_frame("Assessments")
        )
        st.bar_chart(counts, color="#36D6C5")
    else:
        st.info("No applicants have been assessed in this UI session yet.")
    st.warning(
        "Historical validation metrics are not live performance estimates. "
        "Display risk bands are demonstration thresholds—not credit approval decisions."
    )


def render_assessment() -> None:
    st.subheader("Applicant Assessment")
    st.write(
        "Submit only the supported V1 fields. Unavailable model features are "
        "left missing and imputed by the saved preprocessor; predictions with "
        "partial data may be unreliable."
    )
    demo_buttons()
    applicant = applicant_form()
    if applicant is not None:
        assess(applicant)


def render_details() -> None:
    st.subheader("Assessment Details")
    result = st.session_state.get("current_assessment")
    if not result:
        st.info("Complete a successful applicant assessment to view its details.")
        return
    details = st.columns(2)
    details[0].metric("Application ID", result.get("application_id", "—"))
    details[1].metric("Applicant", result.get("name", "—"))
    details = st.columns(2)
    probability = result.get("risk_probability")
    details[0].metric(
        "Predicted default probability",
        f"{probability:.2%}" if isinstance(probability, (int, float)) else "—",
    )
    details[1].metric("Display risk band", result.get("risk_level", "—"))
    st.caption(f"Model version: {result.get('model_version', 'unknown')}")
    st.caption(
        f"Supported API fields supplied: "
        f"{result.get('input_feature_count', 'unknown')} of 12; other trained "
        "features are imputed as missing."
    )
    st.markdown("#### Top SHAP contributors")
    factors = result.get("top_factors") or []
    if factors:
        frame = pd.DataFrame(factors)
        frame["Direction"] = frame["impact"].map({
            "increases_risk": "Pushes toward default",
            "decreases_risk": "Pushes away from default",
        }).fillna("Direction not available")
        frame = frame.rename(columns={
            "feature": "Model feature",
            "shap_value": "SHAP value (raw-score scale)",
        })
        st.dataframe(
            frame[["Model feature", "Direction", "SHAP value (raw-score scale)"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No feature contributions were returned by the API.")
    st.warning(
        "SHAP values explain the model's raw score margin: positive values push "
        "toward TARGET=1 (default), negative values push away. They are not "
        "percentage-point changes or causal explanations. A prediction can be "
        "unreliable when many model inputs were unavailable."
    )
    st.caption(
        "LOW / MEDIUM / HIGH are demonstration probability bands only; "
        "they are not approval, rejection, or credit-policy decisions."
    )


def render_batch() -> None:
    st.subheader("Batch Assessment")
    st.write(
        "Upload a CSV with the three required API inputs: "
        "`AMT_INCOME_TOTAL`, `AMT_CREDIT`, and `DAYS_BIRTH`."
    )
    upload = st.file_uploader("Applicant CSV", type=["csv"])
    if upload is None:
        st.info("Choose a CSV file to preview and assess its rows.")
        return
    try:
        frame = pd.read_csv(upload)
        csv_content, target_removed, dropped_columns = sanitize_batch_csv(frame)
    except (
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        st.error(f"Could not prepare CSV: {exc}")
        return
    st.caption(f"{len(frame):,} rows · {len(frame.columns):,} uploaded columns")
    if target_removed:
        st.warning(
            "TARGET was detected and removed before upload to the prediction API. "
            "The benchmark label is never sent as a prediction input."
        )
    if dropped_columns:
        st.caption(
            "Excluded non-API columns: " + ", ".join(map(str, dropped_columns))
        )
    st.dataframe(
        frame.drop(columns=["TARGET"], errors="ignore").head(8),
        width="stretch",
        hide_index=True,
    )
    if st.button("Run Batch Assessment", type="primary"):
        st.session_state["batch_results"] = []
        st.session_state["batch_counts"] = {}
        try:
            with st.spinner("Submitting batch to the prediction API…"):
                response = predict_batch(API_URL, csv_content, upload.name)
        except APIError as exc:
            st.error(str(exc))
            return
        results = response.get("results", [])
        st.session_state["batch_results"] = results
        st.session_state["batch_counts"] = response
        st.session_state["assessment_history"].extend(
            result for result in results if result.get("risk_level") != "ERROR"
        )

    results = st.session_state.get("batch_results", [])
    if not results:
        return
    summary = st.session_state.get("batch_counts", {})
    counts = st.columns(3)
    counts[0].metric("Rows", summary.get("total_applications", len(results)))
    counts[1].metric("Successful", summary.get("successful_predictions", 0))
    counts[2].metric("Failed rows", summary.get("failed_predictions", 0))
    result_frame = pd.DataFrame(results)
    if "TARGET" in result_frame:
        result_frame = result_frame.drop(columns=["TARGET"])
    search = st.text_input("Filter by application ID or name")
    if search:
        text = result_frame.astype(str).apply(
            lambda column: column.str.contains(search, case=False, na=False)
        ).any(axis=1)
        result_frame = result_frame[text]
    st.dataframe(result_frame, width="stretch", hide_index=True)
    st.download_button(
        "Download results CSV",
        data=result_frame.to_csv(index=False).encode("utf-8"),
        file_name="drifting_oracle_batch_results.csv",
        mime="text/csv",
    )
    failures = result_frame[
        result_frame.get("risk_level", pd.Series(dtype=str)) == "ERROR"
    ]
    if not failures.empty:
        st.warning("Failed rows include an `error` field with the backend reason.")


def render_validation() -> None:
    st.subheader("Model Validation")
    st.caption(
        "These are historical held-out validation artifacts for the V1 baseline, "
        "not performance measurements from current applications."
    )
    try:
        metrics = load_baseline_metrics()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load baseline validation metrics: {exc}")
        return
    cards = st.columns(2)
    cards[0].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    cards[1].metric("PR-AUC / Average Precision", f"{metrics['pr_auc']:.4f}")
    cards = st.columns(2)
    cards[0].metric("Validation score mean", f"{metrics['mean']:.4f}")
    cards[1].metric("Validation score median", f"{metrics['p50']:.4f}")
    score_quantiles = pd.DataFrame(
        {
            "Validation score": [
                metrics["p25"],
                metrics["p50"],
                metrics["p75"],
            ]
        },
            index=[
                "01 · 25th percentile",
                "02 · Median",
                "03 · 75th percentile",
            ],
    )
    st.markdown("#### Validation prediction-score quantiles")
    st.bar_chart(score_quantiles, color="#36D6C5")
    st.caption(
        "Saved quantile summary for held-out validation probabilities; "
        "individual validation scores and ROC/PR curve points were not saved."
    )
    st.markdown("#### Global feature importance")
    try:
        importance = pd.read_csv(SHAP_PATH).head(15)
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"SHAP importance artifact unavailable: {exc}")
    else:
        st.bar_chart(
            importance.set_index("feature")[["mean_abs_shap"]],
            color="#36D6C5",
        )
        st.caption(
            "Mean absolute SHAP importance shows magnitude, not whether a factor "
            "increases or decreases an individual applicant's risk."
        )
    st.markdown("#### Validation and limitations")
    st.write(
        "- The V1 notebook uses a stratified 80/20 train/validation split "
        "(random_state=42).\n"
        "- ROC-AUC and PR-AUC summarize held-out ranking quality; they do not "
        "guarantee calibration or performance on future applicants.\n"
        "- The source dataset is imbalanced (about 8% TARGET=1), and many model "
        "features have missing values.\n"
        "- Form inputs expose only a small subset of the 120 raw training fields; "
        "the API fills absent expected columns with missing values.\n"
        "- No live bank feed, threshold approval policy, or automatic retraining "
        "is implemented."
    )
    st.info("Drift monitoring: Not implemented (planned for Phase 4).")
    health, error = health_status()
    if health:
        st.caption(
            f"API reports model_loaded={health.get('model_loaded')} and "
            f"preprocessor_loaded={health.get('preprocessor_loaded')}."
        )
    elif error:
        st.warning(error)


def main() -> None:
    load_theme()
    initialize_state()
    page = sidebar()
    header()
    if page == "Executive Overview":
        render_overview()
    elif page == "Applicant Assessment":
        render_assessment()
    elif page == "Assessment Details":
        render_details()
    elif page == "Batch Assessment":
        render_batch()
    else:
        render_validation()


if __name__ == "__main__":
    main()
