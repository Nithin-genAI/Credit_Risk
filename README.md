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
- `backend/services/feature_descriptions.py`: explains known training-derived
  features and labels undocumented raw fields without guessing their meaning.
- `backend/services/llm_client.py`: optional Groq text generation and Gemini
  chat/Google Search grounding, with provider-side failures reported safely.
- `backend/services/batch_inference.py`: validates CSV/XLSX feature contracts,
  quarantines unsupported fields and `TARGET`, and preserves row-level result
  alignment.
- `backend/services/drift_monitor.py`: best-effort local prediction logging and
  V1/V2 reference-based drift reports.
- `backend/services/admin_auth.py`: bcrypt credential checks, signed API
  sessions, and process-local login throttling.
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
- `data/drift_monitoring.sqlite3`: generated local prediction log; ignored by
  Git and populated as predictions are made.

## Setup and launch

Use Python 3.13+ and install the repository dependencies:

```powershell
python -m pip install -r requirements.txt
```

For a one-command macOS setup and launch:

```bash
chmod +x scripts/start_mac.sh
./scripts/start_mac.sh
```

The script creates and activates `.venv`, installs `requirements.txt`, warns
when `.env` is missing, then starts FastAPI on port 8000 and Streamlit on port
8501. Press Ctrl+C to stop both services.

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

The Streamlit entry point adds the repository root to Python's import path so
the `frontend` package imports work when deployed with `frontend/app.py` as the
configured main file.

The UI defaults to `http://127.0.0.1:8000`. Override it with
`DRIFTING_ORACLE_API_URL` if the API is hosted elsewhere, for example:

```powershell
$env:DRIFTING_ORACLE_API_URL = "http://127.0.0.1:8000"
```

The API's interactive docs are at `http://127.0.0.1:8000/docs`. FastAPI is an
API on port 8000; the Streamlit browser application is on port 8501. Typing an
API path on port 8501 still reaches the Streamlit app. FastAPI has no
catch-all page route: `/` returns JSON, `/health` is GET, and `/predict` is
POST-only (a browser GET returns HTTP 405).

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

Copy `.env.example` to `.env` in the repository root and add provider credentials
there, never in Streamlit code or a committed template. `GROQ_API_KEY` enables
OpenAI-compatible text generation; optional `GROQ_API_KEY_2` is a backup tried
when the primary Groq key is rejected or rate/quota limited. The backup needs
its own available quota; if both Groq keys are limited, configured Gemini can
serve as a text-generation fallback. `GEMINI_API_KEY` enables Gemini text
generation and Google Search grounding for RBI live-search fallback. Gemini
uses the official `https://generativelanguage.googleapis.com/v1beta` API;
`GEMINI_MODEL` defaults to `gemini-2.5-flash`. General text generation uses
Groq when its key is configured, otherwise Gemini. RBI live search uses Gemini
Google Search grounding only when its key is configured. Without Gemini, RBI
answers use local indexed sources and abstain when those sources cannot verify
the requested currentness. A missing provider does not disable local model or
SHAP results. Restart the backend after changing `.env`.
Health flags indicate key presence, not successful provider access; live
search failures are reported when a request is made, and the assistant abstains
when it cannot obtain verified official citations.

### Admin authentication

The Streamlit dashboard is hidden until an admin signs in. Production uses
bcrypt credentials configured in the repository-root `.env`. Generate a
password hash without echoing the password:

```bash
python -c "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass('New admin password: ').encode(), bcrypt.gensalt(rounds=12)).decode())"
```

Generate the token secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set the generated values in `.env`, for example:

```dotenv
DRIFTING_ORACLE_ENVIRONMENT=production
DRIFTING_ORACLE_ADMIN_AUTH_MODE=bcrypt
DRIFTING_ORACLE_ADMIN_CREDENTIALS_JSON='{"analyst":"<paste-bcrypt-hash-here>"}'
DRIFTING_ORACLE_ADMIN_TOKEN_SECRET=<paste-generated-secret-here>
DRIFTING_ORACLE_ADMIN_TOKEN_TTL_SECONDS=28800
```

For a local academic prototype only, `.env` may instead use:

```dotenv
DRIFTING_ORACLE_ENVIRONMENT=development
DRIFTING_ORACLE_ADMIN_AUTH_MODE=local_plaintext
DRIFTING_ORACLE_ADMIN_USERNAME=admin_1
DRIFTING_ORACLE_ADMIN_PASSWORD=<local-only-password>
DRIFTING_ORACLE_ADMIN_TOKEN_SECRET=<paste-generated-secret-here>
```

Local plaintext mode is rejected outside the development environment and from
non-loopback clients. Keep `.env` ignored by Git and never use this mode for a
deployed app. `/auth/users` provides the username choices and `/auth/login`
exchanges valid credentials for a short-lived bearer token. All business API
routes require that token; `/docs` and `/openapi.json` remain public, but
requests made from the docs still need authorization. Use the Swagger
Authorize control with a token returned by `/auth/login`.

This is a custom, minimal admin gate, not a complete bank identity system. It
does not provide SSO, MFA, password reset, centralized account revocation, or
audit-grade login storage. Failed-login throttling is in-memory per API process
and resets on restart; multiple workers do not share its counters. Streamlit
logout clears its server-side session, while a copied bearer token remains
valid until expiry. Use HTTPS, protect environment secrets, and replace this
approach with managed identity and shared rate limiting before production
bank use.

External AI requests require explicit user consent. Applicant chat sends only
the user's question, bounded recent chat turns, predicted probability/class,
top SHAP factors and descriptions, and missing-feature names; it excludes
applicant identifiers, name, and the full raw feature row. With consent, only
values for the top SHAP features may also be sent, allowing data-specific
questions such as how to correct inaccurate application details. With consent
off, the applicant assistant answers from local model/SHAP evidence only. RBI
questions and retrieved public excerpts may be sent to Gemini for live
grounding; the UI warns users not to include private applicant data.

Run `python scripts/groq.py`, `python scripts/groq.py --backup`, or
`python scripts/gemini.py` from the repository root to send a small
text-generation smoke test to that provider. These scripts
report whether the configured key/model can make a request and the returned
request token usage when available; provider account credit balances must be
checked in the provider's own console. They never print API keys.

`DRIFTING_ORACLE_CLASSIFICATION_THRESHOLD` defaults to `0.15`. It determines
the separate predicted class (class 1 at or above the threshold); it is a
provisional V1 demonstration setting, not an approved lending policy. V2 uses
`DRIFTING_ORACLE_V2_CLASSIFICATION_THRESHOLD`, which defaults to `0.22`. The
Application Review model selector chooses V1 or V2; V1 remains the default for
API clients that omit `model_version`. The model version is carried through
batch validation, scoring, applicant explanations, and applicant chat. Each
version loads its own saved model and matching preprocessor, with artifacts
cached by the backend. Upload size, batch row count, and RBI question length
are configurable with the other `DRIFTING_ORACLE_*` settings.

Drift monitoring uses local SQLite (standard library; no database package is
required). Its path, 30-day default window, 100-prediction minimum, 10,000-row
analysis cap, and 180-day log retention are configurable with
`DRIFTING_ORACLE_MONITORING_DB_PATH`, `DRIFTING_ORACLE_WINDOW_DAYS`,
`DRIFTING_ORACLE_MIN_SAMPLE_SIZE`,
`DRIFTING_ORACLE_MAX_MONITORING_ROWS`, and
`DRIFTING_ORACLE_RETENTION_DAYS`. PSI status thresholds default to 0.10
(moderate) and 0.20 (drifted). Absolute missing-rate changes of 5 and 10
percentage points use the same moderate/critical status levels. All thresholds are
configurable in `.env`; see `.env.example`.

The backend stores the model's prepared feature inputs and probability with a
random prediction ID, timestamp, model version, and optional batch ID. Names,
application IDs, source filenames, and `TARGET` are not stored. These feature
values can still be sensitive: the default SQLite file is unencrypted and
should be used only in a protected environment. The local database is ignored
by Git. Deployments need a protected durable volume or configured database path
if monitoring history must survive restarts.

Build or refresh the local RBI excerpt index from the allowlisted official RBI
pages:

```powershell
python scripts/ingest_rbi_sources.py
```

The RBI assistant checks the local index first. If local evidence is absent, or
the question asks about current applicability, it can use Gemini Google Search
grounding when configured and consented. Live citations are accepted only for
HTTPS `rbi.org.in` pages; the backend does not invent dates or citations. The
local catalog marks the 2022 digital-lending and 2023 responsible-lending
circulars as currentness-unverified. The KYC page records the update date listed
by RBI (14 August 2025); subsequent updates are not established by that local
index. If live search is unavailable or yields no official RBI citations, the
assistant abstains or returns only a clearly caveated local-source response.

## API contracts

- `GET /` and `GET /health`: basic API and model/preprocessor readiness.
- `GET /drift/status?model_version=v1|v2`: overall status for one model.
- `GET /drift/report?model_version=v1|v2`: feature distributions and detailed
  metrics for that model.
- `POST /drift/analyze`: run an on-demand analysis; the JSON body accepts
  `model_version` and optional `window_days`.
- `POST /drift/outcomes`: attach actual outcomes to random prediction IDs when
  labels become available, enabling ROC-AUC, Brier score, precision, and recall.
- `POST /predict`: score one applicant and return its random `prediction_id`.
- `POST /applications/batch/validate`: inspect a CSV/XLSX schema and row values
  without scoring.
- `POST /predict/batch`: score valid rows, return explicit row errors, stable
  identifiers, row numbers, feature coverage, probability, class, and
  provisional classification.
- `POST /applications/explain`: recompute saved-model probability and SHAP
  factors for a selected batch record; optionally generate a grounded narrative.
- `POST /applications/chat`: answer follow-up questions using only the selected
  applicant's recomputed model result, SHAP evidence, feature descriptions, and
  imputation metadata.
- `POST /assistant/rbi`: answer an applicant or loan-officer question from
  local official excerpts or consented Gemini Search grounding, with citations
  and an explicit evidence status.
- `GET /assistant/rbi/sources`: show catalogued source references and whether
  each is currently indexed.

`/predict`, `/applications/batch/validate`, and `/predict/batch` accept the
optional `model_version` query parameter (`v1` or `v2`, default `v1`).
Explanation and applicant-chat requests accept `model_version` in their JSON
body, also defaulting to `v1`. Responses identify the model used and its
threshold. The Streamlit model selector carries the same selection through
batch validation, scoring, inspection, and applicant chat.

Both versions accept the same 120 original model columns. The backend applies
the shared feature engineering; V1 uses 10 derived features, while the V2
preprocessor uses the 9 features it was trained with. `TARGET` and `SK_ID_CURR`
are never model inputs; the latter is only a record reference. Optional absent
features are imputed by the selected model's preprocessor, but incomplete
feature coverage is disclosed and is not equivalent to complete-feature
inference. Each version uses its matching saved model/preprocessor pair; the
API does not retrain them.

Both individual and batch inferences are logged on a best-effort basis: storage
failures are logged but do not fail valid predictions. Applicant explanations
and chat also log the model predictions they compute. The V1 reference is
`models/reference_distribution.json` (130 features); the V2 reference is
`fine_tuning/artifacts/v2_reference_state.json` (129 features). Numeric
features use PSI and a KS diagnostic approximated from saved quantiles;
categorical features use PSI and a chi-square goodness-of-fit diagnostic.
Probability drift compares V1 predictions with saved validation quantiles and
V2 predictions with the saved out-of-fold probability distribution. These
reference sources and quantile-derived KS p-values are approximate, not live
production baselines. Fewer than the configured minimum predictions returns
`Insufficient Data`, not a no-drift claim. Drift reports are descriptive, not
lending-policy decisions or automatic retraining triggers. Performance
metrics remain unavailable until actual outcomes are submitted.

`GET /drift/status` and `GET /drift/report` are API URLs on port 8000, not
Streamlit page routes. Use `/docs` to inspect and call the HTTP methods and
request shapes.

## UI/API smoke checks

Run the API, frontend-client, and diagnostic tests:

```powershell
python -m pytest backend/tests frontend/tests
```

With the API and Streamlit app running, check:

1. **Executive Overview:** verify API/model readiness, historical validation
   metrics, and separate V1/V2 drift status. Click the status to open its
   detailed report and close it with the `× Close` control.
2. **Application Review:** select V1 or V2, validate an uploaded CSV or XLSX
   before scoring, and confirm the selected model's threshold. Check
   missing/unsupported columns, `TARGET` exclusion, row errors, result
   filtering/export, and Inspect navigation.
3. **Applicant Explanation:** inspect local SHAP factors and feature coverage.
   Verify the feature meaning caveats, ask an applicant-specific follow-up with
   consent off (local answer), and confirm the conversation resets for another
   selected application.
4. **RBI Regulatory Assistant:** confirm official sources are clickable, the
   local index is searched first, and current/unindexed questions use live
   search only when Gemini is configured and consented. Otherwise verify the
   assistant abstains rather than presenting an unverified answer as current.

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
- The overview displays ROC-AUC `0.8778` and precision `76.23%` as presentation
  baseline values. They do not match the saved evaluation report and are not
  measured results from the saved model.
- Drift monitoring runs locally from recorded predictions and saved reference
  states. There is no live customer feed or automatic retraining; demo records
  are sampled from a public benchmark. Actual performance metrics require
  ground-truth outcomes to be posted through `/drift/outcomes`.

These checks do not establish production readiness, fairness, calibration,
regulatory suitability, or decision-policy approval.
