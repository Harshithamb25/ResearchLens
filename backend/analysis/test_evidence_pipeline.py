from backend.analysis.evidence_pipeline import (
    run_evidence_pipeline
)


def test_real_evidence_pipeline(monkeypatch):

    query = "What datasets are used for intrusion detection?"

    # Mock the batch Gemini-dependent evidence extraction.
    # Retrieval and reranking remain real.
    def mock_extract_evidence_context_batch(
        evidence_texts
    ):
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
        "backend.analysis.evidence_builder.extract_evidence_context_batch",
        mock_extract_evidence_context_batch
    )

    # Prevent real Gemini calls during relationship analysis.
    def mock_build_evidence_relationships(
        claim_group
    ):
        relationships = []

        for evidence in claim_group["evidence"]:
            relationships.append(
                type(
                    "MockRelationship",
                    (),
                    {
                        "claim": claim_group["claim"],
                        "paper": evidence.paper,
                        "page": evidence.page,
                        "relationship": "SUPPORT",
                        "explanation": (
                            "The supplied evidence supports "
                            "the claim."
                        ),
                        "evidence": evidence
                    }
                )()
            )

        return relationships

    monkeypatch.setattr(
        "backend.analysis.evidence_pipeline.build_evidence_relationships",
        mock_build_evidence_relationships
    )

    result = run_evidence_pipeline(
        question=query,
        retrieval_k=5,
        rerank_k=3,
        claim_threshold=0.75
    )

    assert result["question"] == query
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