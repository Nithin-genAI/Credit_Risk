# Project Folder Structure

```text
Credit_Risk/
|-- .streamlit/
|   `-- config.toml                         # Streamlit theme and app configuration
|-- backend/
|   |-- main.py                             # Auth, prediction, drift, batch, explain, and RBI routes
|   |-- schemas.py                          # Request and response schema validation
|   |-- settings.py                         # Auth, threshold, monitoring, upload, and LLM settings
|   |-- services/
|   |   |-- admin_auth.py                   # Bcrypt admin authentication and signed sessions
|   |   |-- prediction_service.py           # Cached V1/V2 artifact pairs, inference, and SHAP
|   |   |-- batch_inference.py              # Version-aware CSV/XLSX checks and aligned scoring
|   |   |-- drift_monitor.py                # Local prediction logging and reference-based metrics
|   |   |-- llm_client.py                   # Optional backend-only OpenAI-compatible LLM client
|   |   `-- rbi_assistant.py                # RBI excerpt retrieval, citations, and abstention
|   `-- tests/
|       |-- test_api.py                     # Admin auth, API, and batch behavior tests
|       |-- test_drift_monitor.py            # Drift metrics, privacy filtering, and local storage
|       `-- test_assistants.py              # RBI evidence and citation behavior tests
|-- data/
|   |-- application_train.csv               # Home Credit benchmark training dataset
|   |-- application_test.csv                # Home Credit benchmark inference dataset
|   |-- application_test1.csv               # First 100 source rows for batch inference
|   |-- application_test2.csv               # Next 100 source rows for batch inference
|   |-- demo_applicants.csv                 # Fixed-seed sample applicants for repeatable demos
|   |-- drift_monitoring.sqlite3            # Generated local prediction log; ignored by Git
|   `-- rbi_sources/
|       |-- catalog.json                    # Allowlisted official RBI references and status notes
|       `-- index.jsonl                     # Generated local excerpts; rebuild with the ingest script
|-- frontend/
|   |-- app.py                              # Streamlit views, drift status, and report dialog
|   |-- api_client.py                       # Frontend calls and validation for the FastAPI service
|   `-- tests/
|       |-- test_app.py                     # Login/logout and protected dashboard interactions
|       `-- test_api_client.py              # Authenticated API calls and upload preparation
|-- models/
|   |-- baseline_xgboost.joblib             # Trained V1 XGBoost classifier
|   |-- baseline_preprocessor.joblib        # Saved preprocessing pipeline used during training
|   |-- baseline_feature_metadata.json      # Metadata describing model input features
|   |-- baseline_reference_metrics.json     # Saved baseline evaluation metrics
|   |-- baseline_shap_importance.csv        # Global SHAP feature importance values
|   `-- reference_distribution.json         # V1 numeric/categorical drift reference state
|-- scripts/
|   |-- start_mac.sh                         # Create environment and start API/UI together
|   |-- create_demo_applicants.py           # Rebuilds demo_applicants.csv reproducibly
|   |-- split_application_test.py           # Creates two consecutive 100-row test batches
|   `-- ingest_rbi_sources.py               # Builds the local index from allowlisted RBI pages
|-- training_notebooks/
|   |-- 01_understanding_data.ipynb         # Initial dataset understanding
|   |-- 02_eda.ipynb                        # Exploratory data analysis
|   |-- 03_preprocessing.ipynb              # Data cleaning and preprocessing
|   |-- 04_feature_engineering.ipynb         # Derived model feature development
|   |-- 05_baseline_model.ipynb              # Baseline model training and evaluation
|   `-- 06_reference_state.ipynb             # Reference data and monitoring artifacts
|-- testing/
|   |-- __init__.py                           # Makes diagnostics importable for tests
|   |-- test_start_mac_script.py             # Startup setup and process cleanup behavior
|   |-- evaluate_saved_baseline.py            # Reconstructs validation split and diagnoses saved model
|   |-- test_evaluate_saved_baseline.py       # Verifies generated diagnostic outputs and alignment
|   |-- analyze_thresholds.py                 # Compares fixed cutoffs on diagnostic validation
|   |-- test_analyze_thresholds.py            # Tests threshold metrics and candidate selection
|   `-- reports/                              # Generated evaluation outputs; not model artifacts
|       |-- metrics.json                      # Machine-readable partition metrics and checks
|       |-- evaluation_report.md              # Human-readable results, method, and limitations
|       |-- validation_partition_membership.csv # Original validation row IDs, labels, and subset
|       |-- threshold_analysis_metrics.json   # Threshold metrics and selected candidate
|       |-- threshold_analysis_report.md      # Trade-offs and provisional holdout check
|       |-- threshold_comparison.csv          # Metrics for each requested threshold
|       |-- precision_recall_curve.csv        # Curve coordinates for diagnostic validation
|       `-- precision_recall_curve.svg        # Precision-recall curve with cutoff markers
|-- fine_tuning/
|   |-- 01_data_split.ipynb                  # Reproducible stratified V2 data split
|   |-- 02_feature_engineering_review.ipynb  # Audit derived V2 features
|   |-- 03_preprocessing_and_feature_selection.ipynb # Preprocessing and SHAP feature selection
|   |-- 04_model_training.ipynb               # Compare XGBoost, LightGBM, and CatBoost
|   |-- 05_model_evaluation.ipynb              # Cross-validation, test, and OOF threshold analysis
|   |-- 06_reference_state.ipynb               # Build V2 feature reference distributions
|   |-- feature_engineering.py                 # Shared V2 derived-feature formulas
|   |-- data_splits/
|   |   |-- train.csv                         # 70% labeled training partition
|   |   |-- validation.csv                    # 15% labeled validation partition
|   |   |-- test.csv                          # 15% labeled held-out test partition
|   |   `-- split_metadata.json                # Split proportions, seed, and partition checks
|   `-- artifacts/
|       |-- feature_selection_config.json     # Full 129- and reduced 108-feature sets
|       |-- feature_experiment_results.csv    # Early feature-set experiment results
|       |-- feature_set_cv_comparison.csv     # Five-fold full-vs-reduced comparison
|       |-- cv_fold_assignments.csv            # Reproducible development-fold membership
|       |-- v2_shap_feature_ranking.csv        # Global source-feature SHAP ranking
|       |-- v2_model_metrics.json              # V2 held-out test ROC-AUC and average precision
|       |-- v2_xgboost_model.joblib            # Saved V2 XGBoost model
|       |-- v2_preprocessor.joblib             # Saved V2 preprocessing pipeline
|       |-- v2_reference_state.json            # V2 reference distributions for drift monitoring
|       `-- v2_oof_predictions.csv             # OOF probabilities, created by Notebook 05 threshold analysis
|-- .gitattributes                          # Git file-handling rules
|-- .gitignore                              # Files and generated artifacts excluded from Git
|-- data_summary.txt                        # Dataset notes and summary
|-- README.md                               # Project overview, setup, launch, tests, and limitations
|-- requirements.txt                        # Python dependencies
`-- Folder_Structure.md                     # This project structure guide
```

Admin credentials are bcrypt hashes and a token-signing secret configured
through environment variables. Business API routes require a bearer token;
the login and username-list routes are the only public application endpoints.
See `README.md` for setup and custom-auth limitations.

## How the parts fit together

The Streamlit app in `frontend/` sends applicant requests to the FastAPI service
in `backend/`; the backend uses the saved preprocessor and XGBoost model in
`models/` to produce risk predictions and SHAP explanations. Batch schema
validation and inference run in the backend. The notebooks document model
development, while `data/demo_applicants.csv` provides repeatable benchmark
examples for trying the UI.

The Streamlit entry point adds the repository root to Python's import path so
the `frontend` package can be imported when Streamlit Cloud runs
`frontend/app.py` directly. A deployed frontend also needs
`DRIFTING_ORACLE_API_URL` configured to point to a reachable FastAPI backend;
the local default URL is not reachable from Streamlit Cloud.

The demo records are sampled from the Home Credit benchmark dataset. They are
not verified real-world bank applications. Local drift monitoring compares
model-feature inputs and probabilities to version-specific references; it does
not automatically retrain models. The SQLite log contains sensitive feature
values, excludes names/application IDs, is unencrypted, and must be stored on
protected durable storage for deployments.

To prepare two manageable inference batches from the unlabeled test dataset, run
`python scripts/split_application_test.py`. It writes the first 100 source rows
to `data/application_test1.csv` and the next 100 to
`data/application_test2.csv`, preserving the original columns. The files can
then be uploaded separately in Application Review. These are inference examples;
the unlabeled test dataset cannot provide accuracy or other label-based metrics.

The RBI assistant uses the fixed allowlist in `data/rbi_sources/catalog.json`.
Run `python scripts/ingest_rbi_sources.py` to fetch official pages and create
the local excerpt index. The generated `index.jsonl` is ignored by Git and is
not available until ingestion succeeds; the assistant abstains without it.
Source date and currentness notes are shown with citations. A locally indexed
historical circular is not treated as proof of current applicability.

The `testing/` diagnostic uses the existing saved model and preprocessor. It
reconstructs the notebook split, divides the original validation rows into
stratified diagnostic subsets, and writes metrics and row membership under
`testing/reports/`. These are secondary diagnostics, not an independent final
test estimate, because the original validation partition influenced early
stopping. Running `python testing/evaluate_saved_baseline.py` does not retrain
the model or overwrite its artifacts.

`testing/analyze_thresholds.py` compares the requested fixed cutoffs using
predictions from the same saved model and exact diagnostic partitions. It picks
a candidate only from diagnostic-validation F1 and evaluates that candidate
once on the secondary subset as a provisional check. It does not overwrite the
baseline evaluation outputs or define an approved business threshold.

The separate `fine_tuning/` workflow trains V2 from stratified partitions of
the labeled `data/application_train.csv` dataset. Notebook 05 compares the
129-feature and 108-feature candidates with five-fold cross-validation, fits
the final V2 XGBoost model on the combined training and validation partitions,
and reports performance on the untouched test partition. The saved model uses
the full 129-feature set; its preprocessor produces 321 transformed inputs.
Notebook 05 also generates out-of-fold predictions on development data and
compares thresholds from 0.05 through 0.50 using F2. The threshold ranking and
provisional F2-optimal candidate appear in the notebook output, while OOF
probabilities are saved as `fine_tuning/artifacts/v2_oof_predictions.csv`
when analysis completes. These development results do not replace held-out
test evaluation or establish an approved lending threshold.

The API supports both model versions and defaults to V1 for existing clients.
The selected version is used consistently for batch validation, scoring,
applicant explanations, and applicant chat. V1 artifacts are loaded from
`models/`; the V2 model and matching preprocessor are loaded from
`fine_tuning/artifacts/` and cached separately. Their thresholds are configured
independently. The V2 reference state is used for active monitoring; drift
monitoring is available through FastAPI on port 8000, not as a Streamlit URL.
`/docs` lists the HTTP methods and request shapes. The dashboard exposes
separate V1/V2 statuses and a detailed report dialog. V1 reference distributions
are generated from the same benchmark cohort used by
`training_notebooks/06_reference_state.ipynb`; V2 uses
`fine_tuning/artifacts/v2_reference_state.json`.

Generated environments and caches (for example `.venv/`, `__pycache__/`, and
`.git/`) are intentionally omitted from this guide.
