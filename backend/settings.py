"""Runtime configuration for the Drifting Oracle API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_PATH = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_PATH / ".env", override=False)


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


CLASSIFICATION_THRESHOLD = float(
    os.getenv("DRIFTING_ORACLE_CLASSIFICATION_THRESHOLD", "0.15")
)
if not 0 < CLASSIFICATION_THRESHOLD < 1:
    raise ValueError("DRIFTING_ORACLE_CLASSIFICATION_THRESHOLD must be between 0 and 1.")

V2_CLASSIFICATION_THRESHOLD = float(
    os.getenv("DRIFTING_ORACLE_V2_CLASSIFICATION_THRESHOLD", "0.22")
)
if not 0 < V2_CLASSIFICATION_THRESHOLD < 1:
    raise ValueError(
        "DRIFTING_ORACLE_V2_CLASSIFICATION_THRESHOLD must be between 0 and 1."
    )

MAX_UPLOAD_BYTES = _positive_int("DRIFTING_ORACLE_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
MAX_BATCH_ROWS = _positive_int("DRIFTING_ORACLE_MAX_BATCH_ROWS", 5_000)
MAX_UPLOAD_COLUMNS = _positive_int("DRIFTING_ORACLE_MAX_UPLOAD_COLUMNS", 500)
MAX_RBI_QUERY_CHARS = _positive_int("DRIFTING_ORACLE_MAX_RBI_QUERY_CHARS", 1_000)
ADMIN_CREDENTIALS_JSON = os.getenv(
    "DRIFTING_ORACLE_ADMIN_CREDENTIALS_JSON",
    "",
).strip()
ADMIN_AUTH_MODE = os.getenv(
    "DRIFTING_ORACLE_ADMIN_AUTH_MODE",
    "bcrypt",
).strip().casefold()
APP_ENVIRONMENT = os.getenv(
    "DRIFTING_ORACLE_ENVIRONMENT",
    "production",
).strip().casefold()
ADMIN_USERNAME = os.getenv(
    "DRIFTING_ORACLE_ADMIN_USERNAME",
    "",
).strip()
ADMIN_PASSWORD = os.getenv("DRIFTING_ORACLE_ADMIN_PASSWORD", "")
ADMIN_TOKEN_SECRET = os.getenv(
    "DRIFTING_ORACLE_ADMIN_TOKEN_SECRET",
    "",
).strip()
ADMIN_TOKEN_TTL_SECONDS = _positive_int(
    "DRIFTING_ORACLE_ADMIN_TOKEN_TTL_SECONDS",
    8 * 60 * 60,
)

_monitoring_db_path = Path(
    os.getenv(
        "DRIFTING_ORACLE_MONITORING_DB_PATH",
        str(PROJECT_PATH / "data" / "drift_monitoring.sqlite3"),
    )
).expanduser()
MONITORING_DB_PATH = (
    _monitoring_db_path
    if _monitoring_db_path.is_absolute()
    else PROJECT_PATH / _monitoring_db_path
)
DRIFT_WINDOW_DAYS = _positive_int("DRIFTING_ORACLE_WINDOW_DAYS", 30)
DRIFT_MIN_SAMPLE_SIZE = _positive_int("DRIFTING_ORACLE_MIN_SAMPLE_SIZE", 100)
DRIFT_MAX_MONITORING_ROWS = _positive_int(
    "DRIFTING_ORACLE_MAX_MONITORING_ROWS",
    10_000,
)
DRIFT_RETENTION_DAYS = _positive_int(
    "DRIFTING_ORACLE_RETENTION_DAYS",
    180,
)
DRIFT_PSI_MODERATE_THRESHOLD = float(
    os.getenv("DRIFTING_ORACLE_PSI_MODERATE_THRESHOLD", "0.10")
)
DRIFT_PSI_CRITICAL_THRESHOLD = float(
    os.getenv("DRIFTING_ORACLE_PSI_CRITICAL_THRESHOLD", "0.20")
)
DRIFT_MISSING_MODERATE_DELTA = float(
    os.getenv("DRIFTING_ORACLE_MISSING_MODERATE_DELTA", "0.05")
)
DRIFT_MISSING_CRITICAL_DELTA = float(
    os.getenv("DRIFTING_ORACLE_MISSING_CRITICAL_DELTA", "0.10")
)
if not 1 <= DRIFT_WINDOW_DAYS <= 365:
    raise ValueError("DRIFTING_ORACLE_WINDOW_DAYS must be between 1 and 365.")
if DRIFT_RETENTION_DAYS < DRIFT_WINDOW_DAYS:
    raise ValueError(
        "DRIFTING_ORACLE_RETENTION_DAYS must be at least the monitoring window."
    )
if DRIFT_MIN_SAMPLE_SIZE > DRIFT_MAX_MONITORING_ROWS:
    raise ValueError(
        "DRIFTING_ORACLE_MIN_SAMPLE_SIZE cannot exceed the monitoring row cap."
    )
if not 0 < DRIFT_PSI_MODERATE_THRESHOLD < DRIFT_PSI_CRITICAL_THRESHOLD:
    raise ValueError(
        "Drift PSI thresholds must satisfy 0 < moderate < critical."
    )
if not 0 < DRIFT_MISSING_MODERATE_DELTA < DRIFT_MISSING_CRITICAL_DELTA <= 1:
    raise ValueError(
        "Drift missing-rate thresholds must satisfy 0 < moderate < critical <= 1."
    )

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
).rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

PROJECT_ROOT = str(PROJECT_PATH)
RBI_SOURCE_CATALOG = str(PROJECT_PATH / "data" / "rbi_sources" / "catalog.json")
RBI_SOURCE_INDEX = str(PROJECT_PATH / "data" / "rbi_sources" / "index.jsonl")
