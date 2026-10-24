"""Smoke-test Gemini text generation using GEMINI_API_KEY from the environment."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.settings import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
)


def main() -> int:
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is not configured.")
        return 1

    try:
        response = requests.post(
            f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "Reply with exactly: Gemini connection successful."
                            }
                        ],
                    }
                ],
                "generationConfig": {"maxOutputTokens": 32},
            },
            timeout=GEMINI_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        answer = "".join(
            part.get("text", "")
            for candidate in payload.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()
        usage = payload.get("usageMetadata", {})
        if not answer:
            print("Gemini accepted the request but returned no text candidate.")
            return 1
        print(f"Gemini API request succeeded (model: {GEMINI_MODEL}).")
        print(f"Response: {answer}")
        if usage:
            print(f"Request token usage: {usage}")
        print(
            "This confirms API access and this request's usage; it does not "
            "report account credit balance."
        )
        return 0
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(
            f"Gemini rejected the request (HTTP {status}). Check key, model, "
            "API access, and quota."
        )
        return 1
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"Gemini request failed ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
