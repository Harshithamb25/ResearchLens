
"""
Central orchestration layer for ResearchLens.

Connects query classification, evidence analysis
and grounded answer generation.

An answer generated from retrieved evidence is not
automatically considered a completed evidence audit.
"""

from backend.analysis.query_router import classify_query
from backend.analysis.evidence_pipeline import (
    run_evidence_pipeline
)
from backend.analysis.evidence_builder import (
    evidence_to_answer_context
)
from backend.generation.llm_service import generate_answer


ANALYSIS_MESSAGES = {
    "completed": (
        "Cross-paper evidence analysis completed."
    ),
    "partial": (
        "Cross-paper evidence analysis is incomplete. "
        "Some claim groups could not be analyzed."
    ),
    "failed": (
        "Cross-paper evidence analysis failed. "
        "Retrieved passages remain available, but "
        "their relationships have not been verified."
    ),
    "no_evidence": (
        "No analyzable evidence was available "
        "for cross-paper analysis."
    )
}


def process_query(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75
):
    """
    Process a research question and return both
    the grounded answer and audit reliability status.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    query_type = classify_query(question)

    evidence_result = run_evidence_pipeline(
        question=question,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        claim_threshold=claim_threshold
    )

    analysis_status = evidence_result["status"]

    if analysis_status not in ANALYSIS_MESSAGES:
        raise ValueError(
            f"Unexpected analysis status: {analysis_status}"
        )

    # An answer may be generated from retrieved passages
    # even when relationship analysis is incomplete.
    # Its audit status must remain explicit.
    answer = None

    if evidence_result["evidence"]:
        answer_context = evidence_to_answer_context(
            evidence_result["evidence"]
        )

        answer = generate_answer(
            question,
            answer_context
        )

    return {
        "question": question,
        "query_type": query_type,
        "route": "evidence_audit",
        "answer": answer,
        "answer_generated": answer is not None,
        "analysis_status": analysis_status,
        "analysis_message": ANALYSIS_MESSAGES[
            analysis_status
        ],
        "audit_completed": (
            analysis_status == "completed"
        ),
        "failed_group_count": len(
            evidence_result["analysis_failures"]
        ),
        "result": evidence_result
    }