# Drifting Oracle

Drifting Oracle is a credit-default modeling prototype built from the Home
Credit Default Risk benchmark. V1 predicts the probability of `TARGET = 1`
(payment difficulty/default) using a saved XGBoost model and preprocessing
pipeline. It is an evaluation/demo project, not an underwriting or production
banking system.

## Project layout

- `backend/`: FastAPI health, single-applicant, and batch prediction endpoints.
- `backend/services/prediction_service.py`: loads the saved model/preprocessor,
  prepares inputs, applies the training feature engineering, and calculates
  per-applicant SHAP contributions.
- `frontend/`: Streamlit interface; it calls the FastAPI API and does not run
  inference itself.
- `data/application_train.csv`: Home Credit training benchmark dataset.
- `data/demo_applicants.csv`: 30 reproducibly sampled benchmark records for UI
  walkthroughs. Synthetic display names and IDs are used; `TARGET` is excluded.
- `scripts/create_demo_applicants.py`: reproducible generator for that demo file
  (seed 42).
- `training_notebooks/`: data exploration, preprocessing, feature engineering,
  baseline training, and reference artifact notebooks.
- `models/`: serialized V1 model and preprocessor plus metrics, feature metadata,
  global SHAP importance, and reference input distributions.

## Setup and launch

Use Python 3.13+ and install the repository dependencies:

```powershell
python -m pip install -r requirements.txt
```

To recreate the demo records (requires `data/application_train.csv`):

```powershell
python scripts/create_demo_applicants.py
```

Start the API from the repository root:

```powershell
python -m uvicorn backend.main:app --reload
```

Start the light, retro-chroma Streamlit interface in a second terminal:

```powershell
python -m streamlit run frontend/app.py
```

The UI defaults to `http://127.0.0.1:8000`. Override it with
`DRIFTING_ORACLE_API_URL` if the API is hosted elsewhere, for example:

```powershell
$env:DRIFTING_ORACLE_API_URL = "http://127.0.0.1:8000"
```

The API's interactive docs are at `http://127.0.0.1:8000/docs`.

## UI/API smoke checks

Run the available API tests:

```powershell
python -m pytest backend/tests frontend/tests
```

Then check each page with the API running:

1. **Executive Overview:** API health is connected; ROC-AUC and PR-AUC are
   labeled historical validation metrics; live assessment counts are initially
   zero and only change after UI assessments.
2. **Applicant Assessment:** load a demo record, assess it, load the next record,
   randomize selection, and reset. Verify request failures show an error rather
   than a result.
3. **Assessment Details:** verify the returned probability, display band, model
   version, and direction-labeled top SHAP factors.
4. **Batch Assessment:** upload `data/demo_applicants.csv`, inspect success and
   failed-row counts, filter results, and download the results CSV. If an input
   file includes `TARGET`, confirm the UI warns that it is removed before
   inference.
5. **Model Validation:** confirm metrics and the saved global SHAP-importance
   plot load, and drift status reads **Not implemented**.

## V1 validation facts and limitations

- The training notebook and API use the same ten engineered features. They were
  compared on dataset rows and produced identical feature columns and values;
  inference also replaces the Home Credit `DAYS_EMPLOYED = 365243` sentinel with
  missing data as training does.
- The saved classifier is an `XGBClassifier` with classes `[0, 1]`; `predict_proba`
  column 1 therefore represents `TARGET = 1`. The preprocessor emits 254
  transformed columns, matching the model's 254 input features.
- The saved preprocessor imputes numeric columns and imputes/one-hot encodes
  categorical columns (`handle_unknown="ignore"`). It does not scale numeric
  columns. Inference initializes absent raw columns as missing and transforms
  them using that saved preprocessor.
- SHAP uses the XGBoost raw score margin. Positive contributions push toward
  `TARGET = 1`; negative contributions push away. These values are not
  probability-point changes or causal explanations.
- The applicant form collects only a small subset of the training features.
  Missing model features may be imputed successfully while still making the
  output unreliable. The API health response reports 130 expected raw feature
  slots; the saved transformed model schema has 254 encoded columns.
- The baseline thresholds are demo-only presentation bands: below 10% is LOW,
  below 20% is MEDIUM, otherwise HIGH. They are not loan approval rules.
- Validation artifacts report ROC-AUC `0.7700` and PR-AUC `0.2601`. These are
  historical held-out metrics, not current production or live applicant
  performance. ROC/PR curve point data is not saved, so the dashboard does not
  fabricate those curves.
- Drift monitoring and automatic retraining are **not implemented**. There is
  no live customer feed; demo records are sampled from a public benchmark.

These checks do not establish production readiness, fairness, calibration,
regulatory suitability, or decision-policy approval.
