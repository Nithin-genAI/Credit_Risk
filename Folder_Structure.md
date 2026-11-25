# Project Folder Structure

```text
Credit_Risk/
|-- .streamlit/
|   `-- config.toml                         # Streamlit theme and app configuration
|-- backend/
|   |-- main.py                             # Health, validation, batch, explain, and RBI API routes
|   |-- schemas.py                          # Request and response schema validation
|   |-- settings.py                         # Threshold, upload, LLM, and query configuration
|   |-- services/
|   |   |-- prediction_service.py           # Loads artifacts, prepares inputs, predicts, and explains
|   |   |-- batch_inference.py              # CSV/XLSX contract checks and aligned batch scoring
|   |   |-- llm_client.py                   # Optional backend-only OpenAI-compatible LLM client
|   |   `-- rbi_assistant.py                # RBI excerpt retrieval, citations, and abstention
|   `-- tests/
|       |-- test_api.py                     # API and batch behavior tests
|       `-- test_assistants.py              # RBI evidence and citation behavior tests
|-- data/
|   |-- application_train.csv               # Home Credit benchmark training dataset
|   |-- application_test.csv                # Home Credit benchmark inference dataset
|   |-- application_test1.csv               # First 100 source rows for batch inference
|   |-- application_test2.csv               # Next 100 source rows for batch inference
|   |-- demo_applicants.csv                 # Fixed-seed sample applicants for repeatable demos
|   `-- rbi_sources/
|       |-- catalog.json                    # Allowlisted official RBI references and status notes
|       `-- index.jsonl                     # Generated local excerpts; rebuild with the ingest script
|-- frontend/
|   |-- app.py                              # Streamlit views; adds project root for deployment imports
|   |-- api_client.py                       # Frontend calls and validation for the FastAPI service
|   `-- tests/
|       `-- test_api_client.py              # API client and upload preparation tests
|-- models/
|   |-- baseline_xgboost.joblib             # Trained V1 XGBoost classifier
|   |-- baseline_preprocessor.joblib        # Saved preprocessing pipeline used during training
|   |-- baseline_feature_metadata.json      # Metadata describing model input features
|   |-- baseline_reference_metrics.json     # Saved baseline evaluation metrics
|   |-- baseline_shap_importance.csv        # Global SHAP feature importance values
|   `-- reference_distribution.json         # Reference feature distributions for future monitoring
|-- scripts/
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
|       |-- v2_reference_state.json            # V2 reference distributions for future monitoring
|       `-- v2_oof_predictions.csv             # OOF probabilities, created by Notebook 05 threshold analysis
|-- .gitattributes                          # Git file-handling rules
|-- .gitignore                              # Files and generated artifacts excluded from Git
|-- data_summary.txt                        # Dataset notes and summary
|-- README.md                               # Project overview, setup, launch, tests, and limitations
|-- requirements.txt                        # Python dependencies
`-- Folder_Structure.md                     # This project structure guide
```

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
not verified real-world bank applications. Drift monitoring and automatic
retraining are not implemented.

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

V2's model and preprocessor are separate from the API for now: the backend
still loads V1 artifacts from `models/`. V2's metrics, feature ranking, and
reference state therefore do not describe the model currently serving
predictions. The V2 reference state is prepared for future monitoring; drift
detection is not implemented.

Generated environments and caches (for example `.venv/`, `__pycache__/`, and
`.git/`) are intentionally omitted from this guide.
