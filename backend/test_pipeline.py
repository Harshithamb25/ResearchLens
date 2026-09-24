
import pytest

from backend.pipeline import process_query


QUESTION = "Compare the approaches used in both papers."


@pytest.mark.parametrize(
    "status, expected_audit_completed, expected_failures",
    [
        ("completed", True, 0),
        ("partial", False, 1),
        ("failed", False, 1),
        ("no_evidence", False, 0),
    ]
)
def test_process_query_orchestrates_evidence_pipeline(
    monkeypatch,
    status,
    expected_audit_completed,
    expected_failures
):
    """
    Verify answer generation and audit status
    are reported independently.
    """

    def mock_classify_query(question):
        return "comparison"

    evidence = [
        {
            "paper": "paper1.pdf",
            "page": 5,
            "evidence_text": (
                "Deep learning improves intrusion detection."
            )
        }
    ]

    failures = (
        [
            {
                "group_index": 1,
                "claim": "Example claim",
                "reason": "relationship_analysis_failed"
            }
        ]
        if expected_failures
        else []
    )

    def mock_evidence_pipeline(
        question,
        retrieval_k,
        rerank_k,
        claim_threshold
    ):
        has_evidence = status != "no_evidence"

        return {
            "question": question,
            "status": status,
            "evidence": evidence if has_evidence else [],
            "claim_groups": (
                [{"claim": "Example claim"}]
                if has_evidence
                else []
            ),
            "relationships": (
                [{"relationship": "SUPPORT"}]
                if status in ("completed", "partial")
                else []
            ),
            "audits": (
                [{"evidence_coverage": 100.0}]
                if status == "completed"
                else []
            ),
            "analysis_failures": failures
        }

    def mock_evidence_to_answer_context(items):
        assert items == evidence

        return [
            {
                "document": "paper1.pdf",
                "page": 5,
                "chunk_id": None,
                "text": (
                    "Deep learning improves intrusion detection."
                )
            }
        ]

    def mock_generate_answer(question, context):
        assert question == QUESTION
        assert context

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
        question=QUESTION,
        retrieval_k=10,
        rerank_k=5,
        claim_threshold=0.75
    )

    assert result["question"] == QUESTION
    assert result["query_type"] == "comparison"
    assert result["route"] == "evidence_audit"

    assert result["analysis_status"] == status
    assert result["audit_completed"] is expected_audit_completed
    assert result["failed_group_count"] == expected_failures
    assert result["analysis_message"]

    if status == "no_evidence":
        assert result["answer"] is None
        assert result["answer_generated"] is False
        assert result["result"]["evidence"] == []
    else:
        assert result["answer_generated"] is True
        assert "Deep learning improves" in result["answer"]
        assert result["result"]["evidence"]

    if status == "completed":
        assert result["result"]["audits"][0][
            "evidence_coverage"
        ] == 100.0


def test_process_query_rejects_empty_question():
    with pytest.raises(ValueError):
        process_query("   ")