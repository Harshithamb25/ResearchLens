from pipeline import process_query


def test_process_query_orchestrates_evidence_pipeline(monkeypatch):
    """
    Verify that process_query:

    1. Classifies the question.
    2. Calls the evidence pipeline.
    3. Returns the pipeline result.
    4. Exposes the selected query type and route.
    """

    def mock_classify_query(question):
        return "comparison"

    def mock_evidence_pipeline(
        question,
        retrieval_k,
        rerank_k,
        claim_threshold
    ):
        return {
            "question": question,
            "evidence": [
                {
                    "paper": "paper1.pdf",
                    "page": 5
                }
            ],
            "claim_groups": [
                {
                    "claim": "Deep learning improves intrusion detection."
                }
            ],
            "relationships": [
                {
                    "relationship": "SUPPORT"
                }
            ],
            "audits": [
                {
                    "evidence_coverage": 100.0,
                    "unresolved_rate": 0.0
                }
            ]
        }

    monkeypatch.setattr(
        "pipeline.classify_query",
        mock_classify_query
    )

    monkeypatch.setattr(
        "pipeline.run_evidence_pipeline",
        mock_evidence_pipeline
    )

    result = process_query(
        question="Compare the approaches used in both papers.",
        retrieval_k=10,
        rerank_k=5,
        claim_threshold=0.75
    )

    assert result["question"] == (
        "Compare the approaches used in both papers."
    )

    assert result["query_type"] == "comparison"

    assert result["route"] == "evidence_audit"

    assert "result" in result

    assert result["result"]["evidence"]

    assert result["result"]["claim_groups"]

    assert result["result"]["relationships"]

    assert result["result"]["audits"]

    assert (
        result["result"]["audits"][0]["evidence_coverage"]
        == 100.0
    )