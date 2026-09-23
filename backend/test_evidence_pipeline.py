from retrieval.retriever import search
from retrieval.result_formatter import format_retrieval_results
from retrieval.reranker import rerank
from analysis.evidence_builder import build_evidence


def test_real_evidence_pipeline():

    query = "What datasets are used for intrusion detection?"

    # 1. Semantic retrieval from ChromaDB
    results = search(query, top_k=10)

    # 2. Convert ChromaDB output into normalized records
    formatted_results = format_retrieval_results(results)

    assert formatted_results, "No retrieval results returned."

    # 3. Cross-encoder reranking
    reranked_results = rerank(
        query,
        formatted_results,
        top_k=5
    )

    assert reranked_results, "Reranker returned no results."

    # 4. Convert to structured Evidence objects
    evidence = build_evidence(reranked_results)

    assert evidence, "No Evidence objects created."

    # 5. Basic validation
    for item in evidence:
        assert item.paper is not None
        assert item.page is not None
        assert item.evidence_text
        assert item.retrieval_score is not None

    # Print results so we can inspect the real evidence
    print("\n--- ResearchLens Evidence ---")

    for index, item in enumerate(evidence, start=1):
        print(f"\nEvidence {index}")
        print(f"Paper: {item.paper}")
        print(f"Page: {item.page}")
        print(f"Score: {item.retrieval_score}")
        print(f"Text: {item.evidence_text[:300]}...")