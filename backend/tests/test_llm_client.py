from types import SimpleNamespace
from unittest.mock import Mock

import requests

from backend.services.llm_client import (
    _gemini_completion,
    generate_applicant_chat_answer,
    generate_completion,
)


def test_gemini_search_returns_only_official_rbi_citations(monkeypatch):
    monkeypatch.setattr("backend.services.llm_client.GEMINI_API_KEY", "test-key")
    redirect_response = Mock()
    redirect_response.status_code = 302
    redirect_response.headers = {
        "Location": "https://www.rbi.org.in/page"
    }
    monkeypatch.setattr(
        "backend.services.llm_client.requests.get",
        Mock(return_value=redirect_response),
    )
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Use the current RBI direction."}]
                },
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": (
                                    "https://vertexaisearch.cloud.google.com/"
                                    "grounding-api-redirect/test"
                                ),
                                "title": "RBI",
                            }
                        },
                        {"web": {"uri": "https://example.com/fake", "title": "Not official"}},
                        {"web": {"uri": "http://rbi.org.in/insecure", "title": "Insecure"}},
                    ],
                    "groundingSupports": [
                        {
                            "segment": {"text": "Use the current RBI direction."},
                            "groundingChunkIndices": [0],
                        }
                    ],
                },
            }
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("backend.services.llm_client.requests.post", post)

    result = _gemini_completion(
        "Answer using official RBI evidence only.",
        "What are current KYC requirements?",
        max_tokens=300,
        google_search=True,
    )

    assert result.status == "ready"
    assert result.text == "Use the current RBI direction."
    assert [source["url"] for source in result.sources] == [
        "https://www.rbi.org.in/page"
    ]
    assert post.call_args.args[0].endswith(
        "/models/gemini-2.5-flash:generateContent"
    )
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "test-key"
    assert post.call_args.kwargs["json"]["tools"] == [{"google_search": {}}]
    assert post.call_args.kwargs["json"]["systemInstruction"]
    assert redirect_response.close.called
    assert result.sources[0]["relevant_section"] == (
        "Use the current RBI direction."
    )


def test_gemini_search_abstains_without_an_official_citation(monkeypatch):
    monkeypatch.setattr("backend.services.llm_client.GEMINI_API_KEY", "test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Here is an uncited answer."}]
                },
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/rules",
                                "title": "Unverified",
                            }
                        }
                    ]
                },
            }
        ]
    }
    monkeypatch.setattr(
        "backend.services.llm_client.requests.post",
        Mock(return_value=response),
    )

    result = _gemini_completion(
        "Only answer with official RBI citations.",
        "Question",
        max_tokens=300,
        google_search=True,
    )

    assert result.status == "no_official_rbi_sources"
    assert result.text is None
    assert result.sources == []


def test_gemini_search_reports_rate_limit_and_timeout(monkeypatch):
    monkeypatch.setattr("backend.services.llm_client.GEMINI_API_KEY", "test-key")
    bad_request_response = SimpleNamespace(status_code=400)
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError(
        response=bad_request_response
    )
    monkeypatch.setattr(
        "backend.services.llm_client.requests.post",
        Mock(return_value=response),
    )
    bad_request = _gemini_completion(
        "System",
        "Question",
        max_tokens=100,
        google_search=True,
    )
    assert bad_request.status == "provider_error"
    assert bad_request.safe_message is not None
    assert "HTTP 400" in bad_request.safe_message

    limited_response = SimpleNamespace(status_code=429)
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError(
        response=limited_response
    )
    monkeypatch.setattr(
        "backend.services.llm_client.requests.post",
        Mock(return_value=response),
    )
    limited = _gemini_completion(
        "System",
        "Question",
        max_tokens=100,
        google_search=True,
    )
    assert limited.status == "rate_limited"

    monkeypatch.setattr(
        "backend.services.llm_client.requests.post",
        Mock(side_effect=requests.Timeout()),
    )
    timed_out = _gemini_completion(
        "System",
        "Question",
        max_tokens=100,
        google_search=True,
    )
    assert timed_out.status == "timeout"


def test_applicant_chat_prompt_is_audience_aware_and_evidence_limited(monkeypatch):
    captured = {}

    def capture(system_prompt, user_prompt, *, max_tokens):
        captured.update(
            {
                "system": system_prompt,
                "user": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        from backend.services.llm_client import LLMResult

        return LLMResult(status="ready", text="Evidence-based answer.")

    monkeypatch.setattr(
        "backend.services.llm_client.generate_completion",
        capture,
    )
    result = generate_applicant_chat_answer(
        "Why did the model assign this estimate?",
        "applicant",
        {
            "risk_probability": 0.4,
            "top_factors": [
                {
                    "feature": "num__EXT_SOURCE_3",
                    "shap_value": 0.12,
                    "impact": "increases_risk",
                    "description": "External-source score 3.",
                }
            ],
            "missing_feature_names": ["DAYS_EMPLOYED"],
        },
        [{"role": "user", "content": "Earlier question"}],
    )

    assert result.status == "ready"
    assert "respectful, simple language" in captured["system"]
    assert "real-world causation" in captured["system"]
    assert "Earlier question" in captured["user"]
    assert "applicant_id" not in captured["user"]


def test_groq_completion_accepts_text_content_parts_and_authenticates(monkeypatch):
    monkeypatch.setattr("backend.services.llm_client.GROQ_API_KEY", "test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Grounded "},
                        {"type": "text", "text": "answer."},
                    ]
                },
                "finish_reason": "stop",
            }
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("backend.services.llm_client.requests.post", post)

    result = generate_completion("System", "Question")

    assert result.status == "ready"
    assert result.text == "Grounded answer."
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
