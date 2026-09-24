from analysis.evidence import Evidence
from analysis.cross_paper_analysis import EvidenceRelationship
from analysis.evidence_pipeline import run_evidence_pipeline


def test_evidence_pipeline(monkeypatch):

    evidence = Evidence(
        paper="paperA.pdf",
        page=5,
        evidence_text="Deep learning improved detection.",
        claim="Deep learning improves intrusion detection.",
        dataset="NSL-KDD",
        method="Deep neural network",
        metric="Accuracy",
        conditions="Test dataset",
        retrieval_score=8.5
    )

    def mock_search(question, top_k=10):
        return {
            "documents": [[
                "Deep learning improved detection."
            ]],
            "metadatas": [[
                {
                    "document": "paperA.pdf",
                    "page": 5,
                    "chunk_id": 10
                }
            ]],
            "distances": [[0.1]]
        }

    def mock_format(results):
        return [
            {
                "text": "Deep learning improved detection.",
                "document": "paperA.pdf",
                "page": 5,
                "chunk_id": 10,
                "retrieval_distance": 0.1
            }
        ]

    def mock_rerank(question, evidence, top_k=5):
        return [
            {
                "text": "Deep learning improved detection.",
                "document": "paperA.pdf",
                "page": 5,
                "chunk_id": 10,
                "reranker_score": 8.5
            }
        ]

    def mock_build_evidence(results):
        return [evidence]

    def mock_group_claims(
        evidence_items,
        threshold=0.75
    ):
        return [
            {
                "claim": (
                    "Deep learning improves "
                    "intrusion detection."
                ),
                "evidence": evidence_items
            }
        ]

    def mock_relationships(claim_group):

        return [
            EvidenceRelationship(
                claim=claim_group["claim"],
                paper="paperA.pdf",
                page=5,
                relationship="SUPPORT",
                explanation=(
                    "The evidence supports "
                    "the claim."
                ),
                evidence=evidence
            )
        ]

    monkeypatch.setattr(
        "analysis.evidence_pipeline.search",
        mock_search
    )

    monkeypatch.setattr(
        "analysis.evidence_pipeline.format_retrieval_results",
        mock_format
    )

    monkeypatch.setattr(
        "analysis.evidence_pipeline.rerank",
        mock_rerank
    )

    monkeypatch.setattr(
        "analysis.evidence_pipeline.build_evidence",
        mock_build_evidence
    )

    monkeypatch.setattr(
        "analysis.evidence_pipeline.group_claims",
        mock_group_claims
    )

    monkeypatch.setattr(
        "analysis.evidence_pipeline.build_evidence_relationships",
        mock_relationships
    )

    result = run_evidence_pipeline(
        "Does deep learning improve intrusion detection?"
    )

    assert result["question"]

    assert len(result["evidence"]) == 1

    assert len(result["claim_groups"]) == 1

    assert len(result["relationships"]) == 1

    assert len(result["audits"]) == 1

    audit = result["audits"][0]

    assert audit.supporting_evidence == 1

    assert audit.evidence_coverage == 100.0

    assert audit.source_count == 1

    assert audit.unresolved_rate == 0.0