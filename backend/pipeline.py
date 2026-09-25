
"""
Central orchestration for ResearchLens.

Connects query classification, evidence analysis
and grounded answer generation.
"""

from backend.analysis.query_router import (
    classify_query,
)
from backend.analysis.evidence_pipeline import (
    run_evidence_pipeline,
)
from backend.analysis.evidence_builder import (
    evidence_to_answer_context,
)
from backend.generation.llm_service import (
    generate_answer,
)


def _analysis_message(status, cross_paper_evidence):
    if status == "no_evidence":
        return (
            "No analyzable evidence was available "
            "for claim-level analysis."
        )

    if status == "failed":
        return (
            "Evidence relationship analysis failed. "
            "Retrieved passages remain available, "
            "but their relationships are unverified."
        )

    if status == "partial":
        return (
            "Evidence analysis is incomplete. "
            "Some claim groups could not be classified."
        )

    if status == "completed":
        if cross_paper_evidence:
            return (
                "Claim-level analysis completed using "
                "evidence from multiple papers. "
                "Shared themes do not establish agreement."
            )

        return (
            "Claim-level analysis completed, but "
            "the selected evidence comes from one paper. "
            "Cross-paper agreement is not established."
        )

    raise ValueError(
        f"Unexpected analysis status: {status}"
    )


def process_query(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75,
):
    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    query_type = classify_query(question)

    evidence_result = run_evidence_pipeline(
        question=question,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        claim_threshold=claim_threshold,
    )

    analysis_status = evidence_result["status"]

    answer = None

    if evidence_result["evidence"]:
        answer_context = evidence_to_answer_context(
            evidence_result["evidence"]
        )

        answer = generate_answer(
            question,
            answer_context,
        )

    return {
        "question": question,
        "query_type": query_type,
        "route": "evidence_audit",
        "answer": answer,
        "answer_generated": answer is not None,
        "analysis_status": analysis_status,
        "analysis_message": _analysis_message(
            analysis_status,
            evidence_result["cross_paper_evidence"],
        ),
        "audit_completed": (
            analysis_status == "completed"
        ),
        "failed_group_count": len(
            evidence_result["analysis_failures"]
        ),
        "result": evidence_result,
    }