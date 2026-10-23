# Project Folder Structure

```text
Credit_Risk/
|-- .streamlit/
|   `-- config.toml                         # Streamlit theme and app configuration
|-- backend/
|   |-- main.py                             # FastAPI routes for health, single, and batch predictions
|   |-- schemas.py                          # Applicant request schema and field validation
|   |-- services/
|   |   `-- prediction_service.py           # Loads model artifacts, prepares inputs, predicts, and explains
|   `-- tests/
|       `-- test_api.py                     # API behavior tests
|-- data/
|   |-- application_train.csv               # Home Credit benchmark training dataset
|   `-- demo_applicants.csv                 # Fixed-seed sample applicants for repeatable UI demos
|-- frontend/
|   |-- app.py                              # Streamlit dashboard and assessment pages
|   |-- api_client.py                       # Frontend calls and validation for the FastAPI service
|   `-- tests/
|       `-- test_api_client.py              # API client and batch-input tests
|-- models/
|   |-- baseline_xgboost.joblib             # Trained V1 XGBoost classifier
|   |-- baseline_preprocessor.joblib        # Saved preprocessing pipeline used during training
|   |-- baseline_feature_metadata.json      # Metadata describing model input features
|   |-- baseline_reference_metrics.json     # Saved baseline evaluation metrics
|   |-- baseline_shap_importance.csv        # Global SHAP feature importance values
|   `-- reference_distribution.json         # Reference feature distributions for future monitoring
|-- scripts/
|   `-- create_demo_applicants.py           # Rebuilds demo_applicants.csv reproducibly
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
|-- .gitattributes                          # Git file-handling rules
|-- .gitignore                              # Files and generated artifacts excluded from Git
|-- data_summary.txt                        # Dataset notes and summary
|-- README.md                               # Project overview, setup, launch, tests, and limitations
|-- requirements.txt                        # Python dependencies
`-- Folder Structure.md                     # This project structure guide
```

## How the parts fit together

The Streamlit app in `frontend/` sends applicant requests to the FastAPI service
in `backend/`; the backend uses the saved preprocessor and XGBoost model in
`models/` to produce risk predictions and SHAP explanations. The notebooks
document the model-development workflow, while `data/demo_applicants.csv` and
its generator script provide repeatable benchmark examples for trying the UI.

The demo records are sampled from the Home Credit benchmark dataset. They are
not verified real-world bank applications. Drift monitoring and automatic
retraining are not implemented.

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

Generated environments and caches (for example `.venv/`, `__pycache__/`, and
`.git/`) are intentionally omitted from this guide.
