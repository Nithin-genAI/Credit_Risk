import json

from backend.services.llm_client import GroundedLLMResult, LLMResult
from backend.services.rbi_assistant import answer_question, retrieve_sources


def _write_index(path, *, currentness_status, text, source_id="rbi-kyc-example"):
    record = {
        "source_id": source_id,
        "title": "RBI KYC Direction",
        "url": "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
        "issued_on": "2016-02-25",
        "updated_as_of": "2025-08-14",
        "currentness_status": currentness_status,
        "status_note": "The page was updated on 14 August 2025.",
        "topics": (
            ["digital-lending", "borrower-protection"]
            if "digital-lending" in source_id
            else ["kyc", "customer-identification"]
        ),
        "section": "Excerpt 1",
        "text": text,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_rbi_answer_cites_official_source_metadata(tmp_path, monkeypatch):
    index_path = tmp_path / "index.jsonl"
    _write_index(
        index_path,
        currentness_status="consolidated_as_of_source_update",
        text="Know Your Customer identity verification and customer due diligence.",
    )
    monkeypatch.setattr(
        "backend.services.rbi_assistant.generate_completion",
        lambda *args, **kwargs: LLMResult(
            status="ready",
            text="The direction describes customer due diligence.",
        ),
    )

    result = answer_question(
        "What KYC identity verification does the RBI direction describe?",
        "applicant",
        index_path=index_path,
    )

    assert result["answer_status"] == "answered_from_indexed_sources"
    assert result["sources"][0]["source_id"] == "rbi-kyc-example"
    assert result["sources"][0]["updated_as_of"] == "2025-08-14"
    assert result["sources"][0]["url"].startswith("https://www.rbi.org.in/")


def test_rbi_assistant_abstains_on_unverified_currentness(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.rbi_assistant.GEMINI_API_KEY", "")
    index_path = tmp_path / "index.jsonl"
    _write_index(
        index_path,
        currentness_status="historical_currentness_unverified",
        text="Digital lending borrower protection and loan service provider rules.",
        source_id="rbi-digital-lending-2022",
    )
    monkeypatch.setattr(
        "backend.services.rbi_assistant.generate_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("LLM must not be called for unverified current rules.")
        ),
    )

    result = answer_question(
        "What are the current digital lending compliance requirements?",
        "loan_officer",
        index_path=index_path,
    )

    assert result["answer_status"] == "currentness_unverified"
    assert result["sources"][0]["source_id"] == "rbi-digital-lending-2022"


def test_rbi_assistant_abstains_without_indexed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.rbi_assistant.GEMINI_API_KEY", "")
    result = answer_question(
        "What is the current vehicle insurance rule?",
        "applicant",
        index_path=tmp_path / "missing.jsonl",
    )

    assert result["answer_status"] == "no_indexed_evidence"
    assert result["sources"] == []


def test_live_search_fallback_adds_only_grounded_rbi_citations(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("backend.services.rbi_assistant.GEMINI_API_KEY", "test-key")
    official_source = {
        "source_id": "live-rbi-test",
        "title": "RBI circular",
        "issued_on": None,
        "updated_as_of": None,
        "currentness_status": "live_search_date_unverified",
        "status_note": "Date not verified.",
        "url": "https://www.rbi.org.in/example",
        "relevant_section": "Grounded passage",
    }
    grounded = GroundedLLMResult(
        status="ready",
        text="The official RBI source states the cited requirement.",
        sources=[official_source],
    )
    monkeypatch.setattr(
        "backend.services.rbi_assistant.generate_grounded_rbi_completion",
        lambda question, audience, evidence: grounded,
    )

    result = answer_question(
        "What does RBI say about digital lending disclosure?",
        "applicant",
        index_path=tmp_path / "empty-index.jsonl",
    )

    assert result["answer_status"] == "answered_with_live_search_grounding"
    assert result["sources"] == [official_source]
    assert "Live search" in result["disclaimer"]


def test_currentness_question_uses_live_search_even_with_local_match(
    tmp_path,
    monkeypatch,
):
    index_path = tmp_path / "index.jsonl"
    _write_index(
        index_path,
        currentness_status="consolidated_as_of_source_update",
        text="Know Your Customer identity verification and customer due diligence.",
    )
    monkeypatch.setattr("backend.services.rbi_assistant.GEMINI_API_KEY", "test-key")
    captured = {}

    def grounded_search(question, audience, evidence):
        captured.update({"question": question, "audience": audience, "evidence": evidence})
        return GroundedLLMResult(
            status="ready",
            text="Current information was found in the RBI source.",
            sources=[
                {
                    "source_id": "live-rbi-latest",
                    "title": "Current RBI page",
                    "issued_on": None,
                    "updated_as_of": None,
                    "currentness_status": "live_search_date_unverified",
                    "status_note": "Date not verified.",
                    "url": "https://www.rbi.org.in/current",
                    "relevant_section": "Cited passage",
                }
            ],
        )

    monkeypatch.setattr(
        "backend.services.rbi_assistant.generate_grounded_rbi_completion",
        grounded_search,
    )

    result = answer_question(
        "What are the current RBI KYC requirements?",
        "loan_officer",
        index_path=index_path,
    )

    assert result["answer_status"] == "answered_with_live_search_grounding"
    assert len(result["sources"]) == 2
    assert captured["audience"] == "loan_officer"
    assert "excerpt" in captured["evidence"][0]


def test_retrieval_prefers_high_overlap_local_reference():
    chunks = [
        {
            "source_id": "unrelated",
            "title": "RBI KYC",
            "topics": ["kyc"],
            "keywords": ["identity"],
            "text": "identity checks",
        },
        {
            "source_id": "digital",
            "title": "Digital lending circular",
            "topics": ["digital-lending"],
            "keywords": ["borrower protection", "key facts statement"],
            "text": "The key facts statement must describe the digital loan.",
        },
    ]

    found = retrieve_sources("digital loan key facts statement", chunks)

    assert found[0]["source_id"] == "digital"
