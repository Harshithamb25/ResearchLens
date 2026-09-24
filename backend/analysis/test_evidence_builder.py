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

    assert evidence_items[0].paper == "paper1.pdf"
    assert evidence_items[0].page == 3
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

    assert evidence_items[1].paper == "paper2.pdf"
    assert evidence_items[1].page == 7
    assert evidence_items[1].claim == (
        "Claim extracted from evidence 2"
    )