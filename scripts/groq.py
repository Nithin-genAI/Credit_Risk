"""Smoke-test Groq chat completions using GROQ_API_KEY from the environment."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.settings import (
    GROQ_API_KEY,
    GROQ_API_KEY_2,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
)


def main() -> int:
    use_backup = "--backup" in sys.argv[1:]
    api_key = GROQ_API_KEY_2 if use_backup else GROQ_API_KEY
    key_name = "GROQ_API_KEY_2" if use_backup else "GROQ_API_KEY"
    if not api_key:
        print(f"{key_name} is not configured.")
        return 1

    try:
        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly: Groq connection successful.",
                    }
                ],
            },
            timeout=GROQ_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        if not isinstance(answer, str) or not answer.strip():
            print("Groq accepted the request but returned no text.")
            return 1
        print(f"Groq API request succeeded using {key_name} (model: {GROQ_MODEL}).")
        print(f"Response: {answer.strip()}")
        if usage:
            print(f"Request token usage: {usage}")
        print(
            "This confirms API access and this request's usage; it does not "
            "report account credit balance."
        )
        return 0
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"Groq rejected the request (HTTP {status}). Check key, model, and quota.")
        return 1
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"Groq request failed ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
