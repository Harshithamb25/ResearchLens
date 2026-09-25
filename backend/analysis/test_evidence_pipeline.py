
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
    Mock the current source-aware retrieval pipeline.

    Keep retrieval, reranking, and evidence extraction
    deterministic so these tests isolate audit failures.
    """

    all_evidence = [
        item
        for group in claim_groups
        for item in group["evidence"]
    ]

    papers = sorted({
        item.paper
        for item in all_evidence
    })

    candidates = [
        {
            "document": paper,
            "page": 1,
            "chunk_id": index,
            "text": f"Mock passage from {paper}",
        }
        for index, paper in enumerate(papers)
    ]

    monkeypatch.setattr(
        evidence_pipeline,
        "search_across_papers",
        lambda question, per_paper_k, expand_queries: {
            "mock": True
        },
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "format_retrieval_results",
        lambda results: candidates,
    )

    def mock_rerank(question, results, top_k):
        return [
            {
                **item,
                "reranker_score": 1.0,
            }
            for item in results[:top_k]
        ]

    monkeypatch.setattr(
        evidence_pipeline,
        "rerank",
        mock_rerank,
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence",
        lambda results: all_evidence,
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "group_claims",
        lambda evidence, threshold: claim_groups,
    )

    # These tests exercise claim-audit failure handling,
    # not Gemini-based cross-paper theme comparison.
    monkeypatch.setattr(
        evidence_pipeline,
        "compare_cross_paper_themes",
        lambda themes: [],
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
                chunk_id=1,
                claim="Claim A",
                evidence_text="First example passage.",
            ),
            SimpleNamespace(
                paper="paperA.pdf",
                page=1,
                chunk_id=2,
                claim="Claim A",
                evidence_text="Second example passage.",
            ),
        ],
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
                evidence=group["evidence"][0],
            )
        ],
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


def test_additional_evidence_discovers_cross_paper_theme(
    monkeypatch,
):
    """
    A shared theme may be discovered from additional
    candidates even when the main evidence has none.
    """

    initial_evidence = [
        SimpleNamespace(
            paper="sample.pdf",
            page=10,
            chunk_id=1,
            claim="Dataset construction suffers from class imbalance.",
        ),
        SimpleNamespace(
            paper="paper2.pdf",
            page=2,
            chunk_id=2,
            claim="Intrusion detection suffers from false positives.",
        ),
    ]

    additional_evidence = [
        SimpleNamespace(
            paper="paper2.pdf",
            page=8,
            chunk_id=3,
            claim="Limited dataset quality affects generalizability.",
        ),
    ]

    selected = [
        {
            "document": "sample.pdf",
            "page": 10,
            "chunk_id": 1,
            "text": "Initial dataset passage.",
        },
        {
            "document": "paper2.pdf",
            "page": 2,
            "chunk_id": 2,
            "text": "Initial detection-error passage.",
        },
    ]

    extra_candidate = {
        "document": "paper2.pdf",
        "page": 8,
        "chunk_id": 3,
        "text": "Additional dataset passage.",
    }

    candidates = selected + [extra_candidate]

    monkeypatch.setattr(
        evidence_pipeline,
        "_select_evidence",
        lambda question, candidates, limit: candidates[:limit],
    )

    monkeypatch.setattr(
        evidence_pipeline,
        "build_evidence",
        lambda results: additional_evidence,
    )

    themes = evidence_pipeline._discover_additional_themes(
        question="What challenges affect intrusion detection?",
        candidates=candidates,
        selected_results=selected,
        evidence_items=initial_evidence,
    )

    dataset_theme = next(
        theme
        for theme in themes
        if theme["theme_id"] == "DATASET_LIMITATIONS"
    )

    assert dataset_theme["cross_paper"] is True
    assert dataset_theme["source_count"] == 2

    assert {
        item.paper
        for item in dataset_theme["evidence"]
    } == {"sample.pdf", "paper2.pdf"}

    # Additional findings must not silently alter
    # the main answer's evidence.
    assert len(initial_evidence) == 2