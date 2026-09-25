"""Regression tests for evidence-aware final synthesis."""

from types import SimpleNamespace

from backend.pipeline import process_query
from backend.generation.llm_service import generate_answer


def _source(paper, page, chunk_id, text):
    return SimpleNamespace(
        paper=paper, page=page, chunk_id=chunk_id,
        evidence_text=text,
    )


def test_pipeline_supplies_original_thematic_passages(monkeypatch):
    main = _source("paper2.pdf", 2, "p2-2", "High false positives are reported.")
    supplemental = _source("sample.pdf", 16, "s-16", "Scalability is discussed.")
    comparison = {
        "theme": "Scalability",
        "relationship": "RELATED_FINDINGS",
        "explanation": "Related research concerns; agreement not established.",
        "findings": [
            {"paper": "paper2.pdf", "page": 2, "chunk_id": "p2-2", "claim": "False positives"},
            {"paper": "sample.pdf", "page": 16, "chunk_id": "s-16", "claim": "Scalability"},
        ],
    }
    result = {
        "status": "completed", "cross_paper_evidence": False,
        "evidence": [main], "themes": [{"evidence": [main, supplemental]}],
        "theme_comparisons": [comparison], "analysis_failures": [],
    }
    monkeypatch.setattr("backend.pipeline.classify_query", lambda _: "comparison")
    monkeypatch.setattr("backend.pipeline.run_evidence_pipeline", lambda **_: result)
    monkeypatch.setattr(
        "backend.pipeline.evidence_to_answer_context",
        lambda _: [{"document": "paper2.pdf", "page": 2, "chunk_id": "p2-2", "text": main.evidence_text}],
    )
    captured = {}

    def mock_generate(question, context, **kwargs):
        captured.update(context=context, kwargs=kwargs)
        return "Grounded answer"

    monkeypatch.setattr("backend.pipeline.generate_answer", mock_generate)
    output = process_query("What are the limitations?")
    assert output["answer"] == "Grounded answer"
    assert len(captured["context"]) == 2
    assert captured["context"][1]["text"] == "Scalability is discussed."
    assert captured["kwargs"]["theme_comparisons"] == [comparison]
    assert captured["kwargs"]["analysis_status"] == "completed"


def test_generator_omits_comparisons_with_missing_source_passages(monkeypatch):
    captured = {}

    def mock_generate_content(*, model, contents):
        captured["prompt"] = contents
        return SimpleNamespace(text="The available evidence is limited.")

    monkeypatch.setattr(
        "backend.generation.llm_service.client.models.generate_content",
        mock_generate_content,
    )
    answer = generate_answer(
        "What is established?",
        [{"document": "paper2.pdf", "page": 2, "chunk_id": "p2-2", "text": "A limitation."}],
        theme_comparisons=[{
            "theme": "Scalability", "relationship": "COMMON_CONCERN",
            "findings": [
                {"paper": "paper2.pdf", "page": 2, "chunk_id": "p2-2"},
                {"paper": "missing.pdf", "page": 9, "chunk_id": "missing"},
            ],
        }],
        analysis_status="partial",
    )
    assert answer == "The available evidence is limited."
    assert "No grounded cross-paper comparison is available." in captured["prompt"]
    assert "CLAIM-LEVEL AUDIT STATUS: partial" in captured["prompt"]
