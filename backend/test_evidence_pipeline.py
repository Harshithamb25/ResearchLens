from backend.retrieval.retriever import search
from backend.retrieval.result_formatter import format_retrieval_results
from backend.retrieval.reranker import rerank
from backend.analysis.evidence_builder import build_evidence


def test_real_evidence_pipeline(monkeypatch):

    query = "What datasets are used for intrusion detection?"

    # Mock only the Gemini-dependent evidence extraction.
    # Retrieval and reranking remain real.
    def mock_extract_evidence_context(text):
        return {
            "claim": "The research discusses datasets used for intrusion detection.",
            "dataset": "KDDCup99 and NSL-KDD",
            "method": "Intrusion detection classification",
            "metric": "Accuracy",
            "conditions": "Research dataset evaluation"
        }

    monkeypatch.setattr(
        "backend.analysis.evidence_builder.extract_evidence_context",
        mock_extract_evidence_context
    )

    # 1. Semantic retrieval from ChromaDB
    results = search(
        query,
        top_k=10
    )

    # 2. Convert ChromaDB output into normalized records
    formatted_results = format_retrieval_results(
        results
    )

    assert formatted_results, "No retrieval results returned."

    # 3. Cross-encoder reranking
    reranked_results = rerank(
        query,
        formatted_results,
        top_k=5
    )

    assert reranked_results, "Reranker returned no results."

    # 4. Convert to structured Evidence objects
    evidence = build_evidence(
        reranked_results
    )

    assert evidence, "No Evidence objects created."

    # 5. Validate structured evidence
    for item in evidence:
        assert item.paper is not None
        assert item.page is not None
        assert item.evidence_text
        assert item.retrieval_score is not None

        assert item.claim is not None
        assert item.dataset is not None
        assert item.method is not None
        assert item.metric is not None
        assert item.conditions is not None

    # 6. Verify that multiple retrieval stages produced results
    assert len(formatted_results) >= len(reranked_results)
    assert len(reranked_results) == len(evidence)