import json

from backend.services.llm_client import LLMResult
from backend.services.rbi_assistant import answer_question


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


def test_rbi_assistant_abstains_without_indexed_evidence(tmp_path):
    result = answer_question(
        "What is the current vehicle insurance rule?",
        "applicant",
        index_path=tmp_path / "missing.jsonl",
    )

    assert result["answer_status"] == "no_indexed_evidence"
    assert result["sources"] == []
