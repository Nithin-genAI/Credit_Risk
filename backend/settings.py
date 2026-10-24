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

MAX_UPLOAD_BYTES = _positive_int("DRIFTING_ORACLE_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
MAX_BATCH_ROWS = _positive_int("DRIFTING_ORACLE_MAX_BATCH_ROWS", 5_000)
MAX_UPLOAD_COLUMNS = _positive_int("DRIFTING_ORACLE_MAX_UPLOAD_COLUMNS", 500)
MAX_RBI_QUERY_CHARS = _positive_int("DRIFTING_ORACLE_MAX_RBI_QUERY_CHARS", 1_000)

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
