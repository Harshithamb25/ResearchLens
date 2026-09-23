from analysis.evidence_builder import build_evidence


def test_build_evidence():

    reranked_results = [
        {
            "text": "The model achieved 95% accuracy.",
            "document": "paper2.pdf",
            "page": 12,
            "chunk_id": 80,
            "reranker_score": 8.39
        }
    ]

    evidence = build_evidence(reranked_results)

    assert len(evidence) == 1

    assert evidence[0].paper == "paper2.pdf"
    assert evidence[0].page == 12
    assert evidence[0].evidence_text == (
        "The model achieved 95% accuracy."
    )
    assert evidence[0].retrieval_score == 8.39