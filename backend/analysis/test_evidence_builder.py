from backend.analysis.evidence_builder import build_evidence


def mock_extract_evidence_context(text):
    """
    Simulates the structured output produced by
    the evidence extraction component.
    """

    return {
        "claim": "The model achieved 95% accuracy.",
        "dataset": "NSL-KDD",
        "method": "Deep neural network",
        "metric": "Accuracy",
        "conditions": "Test dataset"
    }


def test_build_evidence(monkeypatch):

    monkeypatch.setattr(
        "backend.analysis.evidence_builder.extract_evidence_context",
        mock_extract_evidence_context
    )

    reranked_results = [
        {
            "text": "The model achieved 95% accuracy.",
            "document": "paper2.pdf",
            "page": 12,
            "chunk_id": 80,
            "reranker_score": 8.39
        }
    ]

    evidence = build_evidence(
        reranked_results
    )

    assert evidence

    assert len(evidence) == 1

    item = evidence[0]

    assert item.paper == "paper2.pdf"

    assert item.page == 12

    assert (
        item.evidence_text
        == "The model achieved 95% accuracy."
    )

    assert (
        item.claim
        == "The model achieved 95% accuracy."
    )

    assert item.dataset == "NSL-KDD"

    assert item.method == "Deep neural network"

    assert item.metric == "Accuracy"

    assert item.conditions == "Test dataset"

    assert item.retrieval_score == 8.39


