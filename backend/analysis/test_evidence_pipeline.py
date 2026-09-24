
from types import SimpleNamespace

import pytest

from backend.analysis import evidence_pipeline
from backend.analysis.evidence_pipeline import (
    run_evidence_pipeline
)


def test_real_evidence_pipeline(monkeypatch):
    """
    Test real retrieval and reranking with mocked
    Gemini-dependent operations.
    """
    query = "What datasets are used for intrusion detection?"

    def mock_extract_evidence_context_batch(evidence_texts):
        return [
            {
                "claim": (
                    "The research discusses datasets "
                    "used for intrusion detection."
                ),
                "dataset": "KDDCup99 and NSL-KDD",
                "method": "Intrusion detection classification",
                "metric": "Accuracy",
                "conditions": "Research dataset evaluation"
            }
            for _ in evidence_texts
        ]

    monkeypatch.setattr(
        "backend.analysis.evidence_builder."
        "extract_evidence_context_batch",
        mock_extract_evidence_context_batch
    )

    def mock_build_evidence_relationships(claim_group):
        return [
            SimpleNamespace(
                claim=claim_group["claim"],
                paper=evidence.paper,
                page=evidence.page,
                relationship="SUPPORT",
                explanation="The evidence supports the claim.",
                evidence=evidence
            )
            for evidence in claim_group["evidence"]
        ]

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence_relationships",
        mock_build_evidence_relationships
    )

    result = run_evidence_pipeline(
        question=query,
        retrieval_k=5,
        rerank_k=3,
        claim_threshold=0.75
    )

    assert result["question"] == query
    assert result["status"] == "completed"
    assert result["analysis_failures"] == []

    assert len(result["evidence"]) > 0
    assert len(result["claim_groups"]) > 0
    assert len(result["relationships"]) > 0
    assert len(result["audits"]) > 0

    for evidence in result["evidence"]:
        assert evidence.claim is not None
        assert evidence.dataset is not None
        assert evidence.method is not None
        assert evidence.metric is not None
        assert evidence.conditions is not None


def setup_mock_pipeline(monkeypatch, claim_groups):
    """
    Replace retrieval, reranking and evidence
    preparation with deterministic test data.
    """
    monkeypatch.setattr(
        evidence_pipeline,
        "search",
        lambda question, top_k: {"mock": True}
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "format_retrieval_results",
        lambda results: [{"text": "Mock passage"}]
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "rerank",
        lambda question, results, top_k: results
    )

    all_evidence = [
        item
        for group in claim_groups
        for item in group["evidence"]
    ]

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence",
        lambda results: all_evidence
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "group_claims",
        lambda evidence, threshold: claim_groups
    )


def make_group(claim, paper):
    evidence = SimpleNamespace(
        paper=paper,
        page=1,
        chunk_id=1,
        evidence_text="Example evidence",
        claim=claim
    )

    return {
        "claim": claim,
        "evidence": [evidence]
    }


def make_relationship(group):
    evidence = group["evidence"][0]

    return SimpleNamespace(
        claim=group["claim"],
        paper=evidence.paper,
        page=evidence.page,
        relationship="SUPPORT",
        explanation="The passage supports the claim.",
        evidence=evidence
    )


def test_partial_analysis_failure(monkeypatch):
    first = make_group("Claim A", "paperA.pdf")
    second = make_group("Claim B", "paperB.pdf")

    setup_mock_pipeline(
        monkeypatch,
        [first, second]
    )

    def mock_analyzer(group):
        if group["claim"] == "Claim B":
            raise RuntimeError("Simulated provider failure")

        return [make_relationship(group)]

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence_relationships",
        mock_analyzer
    )

    result = run_evidence_pipeline("Compare the claims")

    assert result["status"] == "partial"
    assert len(result["relationships"]) == 1
    assert len(result["audits"]) == 2
    assert len(result["analysis_failures"]) == 1

    successful_audit = result["audits"][0]
    failed_audit = result["audits"][1]

    assert successful_audit.evidence_coverage == 100.0
    assert failed_audit.evidence_coverage == 0.0
    assert failed_audit.unanalyzed_evidence == 1

    assert result["analysis_failures"][0]["claim"] == "Claim B"


def test_complete_analysis_failure(monkeypatch):
    groups = [
        make_group("Claim A", "paperA.pdf"),
        make_group("Claim B", "paperB.pdf")
    ]

    setup_mock_pipeline(monkeypatch, groups)

    def failing_analyzer(group):
        raise RuntimeError("Simulated analysis failure")

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence_relationships",
        failing_analyzer
    )

    result = run_evidence_pipeline("Analyze both claims")

    assert result["status"] == "failed"
    assert result["relationships"] == []
    assert len(result["analysis_failures"]) == 2

    for audit in result["audits"]:
        assert audit.analyzed_evidence == 0
        assert audit.evidence_coverage == 0.0
        assert audit.unanalyzed_evidence == 1


def test_incomplete_relationship_results(monkeypatch):
    group = {
        "claim": "Claim A",
        "evidence": [
            SimpleNamespace(
                paper="paperA.pdf",
                page=1,
                chunk_id=1
            ),
            SimpleNamespace(
                paper="paperA.pdf",
                page=1,
                chunk_id=2
            )
        ]
    }

    setup_mock_pipeline(monkeypatch, [group])

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence_relationships",
        lambda claim_group: [
            SimpleNamespace(
                claim="Claim A",
                paper="paperA.pdf",
                page=1,
                relationship="SUPPORT",
                evidence=group["evidence"][0]
            )
        ]
    )

    result = run_evidence_pipeline("Analyze Claim A")

    assert result["status"] == "failed"
    assert result["relationships"] == []
    assert len(result["analysis_failures"]) == 1

    audit = result["audits"][0]

    assert audit.total_evidence == 2
    assert audit.analyzed_evidence == 0
    assert audit.unanalyzed_evidence == 2
    assert audit.evidence_coverage == 0.0