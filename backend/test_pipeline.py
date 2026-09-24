from backend.pipeline import process_query


def test_process_query_orchestrates_evidence_pipeline(monkeypatch):
    """
    Verify that process_query:

    1. Classifies the question.
    2. Calls the evidence pipeline.
    3. Converts evidence into answer context.
    4. Generates a grounded answer.
    5. Returns the final answer and evidence audit.
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
                    "page": 5,
                    "evidence_text": "Deep learning improves intrusion detection."
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

    def mock_evidence_to_answer_context(evidence):
        return [
            {
                "document": "paper1.pdf",
                "page": 5,
                "chunk_id": None,
                "text": "Deep learning improves intrusion detection."
            }
        ]

    def mock_generate_answer(question, evidence):
        return (
            "Deep learning improves intrusion detection "
            "[paper1.pdf, Page 5]."
        )

    monkeypatch.setattr(
        "backend.pipeline.classify_query",
        mock_classify_query
    )

    monkeypatch.setattr(
        "backend.pipeline.run_evidence_pipeline",
        mock_evidence_pipeline
    )

    monkeypatch.setattr(
        "backend.pipeline.evidence_to_answer_context",
        mock_evidence_to_answer_context
    )

    monkeypatch.setattr(
        "backend.pipeline.generate_answer",
        mock_generate_answer
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

    assert result["answer"] is not None

    assert "Deep learning improves intrusion detection" in result["answer"]

    assert "result" in result

    assert result["result"]["evidence"]

    assert result["result"]["claim_groups"]

    assert result["result"]["relationships"]

    assert result["result"]["audits"]

    assert (
        result["result"]["audits"][0]["evidence_coverage"]
        == 100.0
    )
