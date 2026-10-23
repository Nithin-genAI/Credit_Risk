# Drifting Oracle

Drifting Oracle is a credit-default modeling prototype built from the Home
Credit Default Risk benchmark. V1 predicts the probability of `TARGET = 1`
(payment difficulty/default) using a saved XGBoost model and preprocessing
pipeline. It is an evaluation/demo project, not an underwriting or production
banking system.

## Project layout

- `backend/`: FastAPI health, validation, batch prediction, explanation, and RBI
  assistant endpoints.
- `backend/services/prediction_service.py`: loads the saved model/preprocessor,
  prepares inputs, applies the training feature engineering, and calculates
  per-applicant SHAP contributions.
- `backend/services/batch_inference.py`: validates CSV/XLSX feature contracts,
  quarantines unsupported fields and `TARGET`, and preserves row-level result
  alignment.
- `backend/services/rbi_assistant.py`: retrieves relevant excerpts from the
  locally generated, allowlisted RBI source index and abstains when evidence or
  source currency is insufficient.
- `frontend/`: Streamlit dashboard; it calls the FastAPI API and does not run
  inference itself.
- `scripts/ingest_rbi_sources.py`: fetches only catalogued official RBI pages
  and builds the local index at `data/rbi_sources/index.jsonl` (generated,
  ignored by Git).
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

## Prepare small inference batches

To split the first 200 records from `data/application_test.csv` into two
consecutive, 100-row batch-upload files, run:

```powershell
python scripts/split_application_test.py
```

This creates `data/application_test1.csv` (source rows 1–100) and
`data/application_test2.csv` (source rows 101–200). Both keep the source feature
columns, including `SK_ID_CURR` for matching, and do not include a `TARGET`
label. These files demonstrate inference only; they cannot measure prediction
accuracy.

## Configuration

Copy `.env.example` to `.env` in the repository root. Configure
`GROQ_API_KEY` only in this backend environment to enable optional
OpenAI-compatible plain-language explanations. If no key is configured, local
model probability and SHAP factors still work; the API reports the narrative as
not configured. Credentials are never placed in Streamlit configuration.
The default model is `openai/gpt-oss-120b`, and the default Groq-compatible API
base URL is `https://api.groq.com/openai/v1`. Restart the backend after changing
`.env` so the process loads the updated settings.
AI calls require an explicit UI/API consent flag. Applicant explanations send
only the probability, class, threshold, and top SHAP factors—not applicant
identifiers or the raw feature row. RBI questions and retrieved public excerpts
are sent only after the user acknowledges that disclosure; users are warned not
to enter personal applicant data.

`DRIFTING_ORACLE_CLASSIFICATION_THRESHOLD` defaults to `0.15`. It determines
the separate predicted class (class 1 at or above the threshold); it is a
provisional demonstration setting, not an approved lending policy. Upload size,
batch row count, and RBI question length are configurable with the other
`DRIFTING_ORACLE_*` settings.

Build or refresh the local RBI excerpt index from the three allowlisted official
RBI pages:

```powershell
python scripts/ingest_rbi_sources.py
```

The assistant cites source metadata from `data/rbi_sources/catalog.json`, not
LLM-generated URLs. Its corpus marks the 2022 digital-lending and 2023
responsible-lending circulars as currentness-unverified; it does not present
them alone as verified current consolidated requirements. The KYC page records
the update date listed by RBI (14 August 2025); later updates are not verified.
Refresh and verify official sources before compliance use. If the local index
is missing, the assistant abstains instead of inventing an answer.

## API contracts

- `GET /` and `GET /health`: basic API and model/preprocessor readiness.
- `POST /applications/batch/validate`: inspect a CSV/XLSX schema and row values
  without scoring.
- `POST /predict/batch`: score valid rows, return explicit row errors, stable
  identifiers, row numbers, feature coverage, probability, class, and
  provisional classification.
- `POST /applications/explain`: recompute saved-model probability and SHAP
  factors for a selected batch record; optionally generate a grounded narrative.
- `POST /assistant/rbi`: answer an applicant or loan-officer question from
  indexed official excerpts, with dated citations or an abstention status.
- `GET /assistant/rbi/sources`: show catalogued source references and whether
  each is currently indexed.

CSV and XLSX requests use the saved 120-original-feature contract plus the
existing 10 training-time engineered features. `TARGET` and `SK_ID_CURR` are
never model inputs; the latter is only a record reference. Optional absent
features are imputed by the saved preprocessor, but incomplete feature coverage
is disclosed and is not equivalent to complete-feature inference. The saved
model and preprocessor are reused; this workflow does not retrain them.

## UI/API smoke checks

Run the API, frontend-client, and diagnostic tests:

```powershell
python -m pytest backend/tests frontend/tests
```

With the API and Streamlit app running, check:

1. **Executive Overview:** verify API/model readiness, historical validation
   metrics, and session-only counts/activity.
2. **Application Review:** validate an uploaded CSV or XLSX before scoring;
   check missing/unsupported columns, `TARGET` exclusion, row errors,
   result filtering/export, and Inspect navigation.
3. **Applicant Explanation:** inspect local SHAP factors and feature coverage.
   Without a configured LLM key, confirm local evidence remains available and
   the narrative says it is not configured.
4. **RBI Regulatory Assistant:** confirm official sources are clickable, the
   digital-lending query discloses unverified currentness, and unrelated
   questions abstain.

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
- Batch uploads may provide the full set of original model features. Missing
  optional fields are imputed; feature coverage warnings disclose incomplete
  inference. The legacy single-applicant API remains available but is not
  exposed as a manual form in the dashboard.
- The dashboard uses 0.15 as a provisional classification threshold, separate
  from the returned default probability. It is not an approved credit or
  lending decision policy.
- Validation artifacts report ROC-AUC `0.7700` and PR-AUC `0.2601`. These are
  historical held-out metrics, not current production or live applicant
  performance. ROC/PR curve point data is not saved, so the dashboard does not
  fabricate those curves.
- Drift monitoring and automatic retraining are **not implemented**. There is
  no live customer feed; demo records are sampled from a public benchmark.

These checks do not establish production readiness, fairness, calibration,
regulatory suitability, or decision-policy approval.
