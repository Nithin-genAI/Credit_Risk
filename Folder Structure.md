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

Generated environments and caches (for example `.venv/`, `__pycache__/`, and
`.git/`) are intentionally omitted from this guide.
