from backend.analysis.evidence_builder import build_evidence


def mock_extract_evidence_context_batch(evidence_texts):
    return [
        {
            "claim": f"Claim extracted from evidence {index}",
            "dataset": "KDDCup99",
            "method": "Intrusion detection classification",
            "metric": "Accuracy",
            "conditions": "Research dataset evaluation"
        }
        for index, _ in enumerate(evidence_texts, start=1)
    ]


def test_build_evidence(monkeypatch):

    monkeypatch.setattr(
        "backend.analysis.evidence_builder.extract_evidence_context_batch",
        mock_extract_evidence_context_batch
    )

    reranked_results = [
        {
            "text": "Evidence passage one.",
            "document": "paper1.pdf",
            "page": 3,
            "chunk_id": 1,
            "reranker_score": 8.5
        },
        {
            "text": "Evidence passage two.",
            "document": "paper2.pdf",
            "page": 7,
            "chunk_id": 2,
            "reranker_score": 7.5
        }
    ]

    evidence_items = build_evidence(
        reranked_results
    )

    assert len(evidence_items) == 2

    # First evidence item
    assert evidence_items[0].paper == "paper1.pdf"
    assert evidence_items[0].page == 3
    assert evidence_items[0].chunk_id == 1
    assert evidence_items[0].claim == (
        "Claim extracted from evidence 1"
    )
    assert evidence_items[0].dataset == "KDDCup99"
    assert evidence_items[0].method == (
        "Intrusion detection classification"
    )
    assert evidence_items[0].metric == "Accuracy"
    assert evidence_items[0].conditions == (
        "Research dataset evaluation"
    )
    assert evidence_items[0].retrieval_score == 8.5

    # Second evidence item
    assert evidence_items[1].paper == "paper2.pdf"
    assert evidence_items[1].page == 7
    assert evidence_items[1].chunk_id == 2
    assert evidence_items[1].claim == (
        "Claim extracted from evidence 2"
    )
    assert evidence_items[1].dataset == "KDDCup99"
    assert evidence_items[1].method == (
        "Intrusion detection classification"
    )
    assert evidence_items[1].metric == "Accuracy"
    assert evidence_items[1].conditions == (
        "Research dataset evaluation"
    )
    assert evidence_items[1].retrieval_score == 7.5


def test_evidence_to_answer_context():

    reranked_results = [
        {
            "text": "Evidence passage one.",
            "document": "paper1.pdf",
            "page": 3,
            "chunk_id": 42,
            "reranker_score": 8.5
        }
    ]

    # Use the batch extraction function through the same
    # builder path, with deterministic test data.
    from backend.analysis import evidence_builder

    original_function = (
        evidence_builder.extract_evidence_context_batch
    )

    evidence_builder.extract_evidence_context_batch = (
        mock_extract_evidence_context_batch
    )

    try:
        evidence_items = build_evidence(
            reranked_results
        )

        from backend.analysis.evidence_builder import (
            evidence_to_answer_context
        )

        answer_context = evidence_to_answer_context(
            evidence_items
        )

        assert len(answer_context) == 1
        assert answer_context[0]["document"] == "paper1.pdf"
        assert answer_context[0]["page"] == 3
        assert answer_context[0]["chunk_id"] == 42
        assert answer_context[0]["text"] == (
            "Evidence passage one."
        )

    finally:
        evidence_builder.extract_evidence_context_batch = (
            original_function
        )