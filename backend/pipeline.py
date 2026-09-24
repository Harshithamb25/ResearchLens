"""
Central orchestration layer for ResearchLens.

This module connects query classification to the
evidence-grounded research analysis pipeline and
final grounded answer generation.

Flow:

    Question
        ↓
    Query Classification
        ↓
    Semantic Retrieval
        ↓
    Reranking
        ↓
    Evidence Extraction
        ↓
    Claim Grouping
        ↓
    Cross-Paper Relationship Analysis
        ↓
    Evidence Audit
        ↓
    Grounded Answer Generation
        ↓
    Final Research Response
"""

from backend.analysis.query_router import classify_query
from backend.analysis.evidence_pipeline import (
    run_evidence_pipeline
)
from backend.analysis.evidence_builder import (
    evidence_to_answer_context
)
from backend.generation.llm_service import generate_answer


def process_query(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75
):
    """
    Process a ResearchLens research question.

    The pipeline performs evidence retrieval and auditing
    first, then generates a grounded answer using only
    the retrieved evidence.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    # 1. Classify the research question
    query_type = classify_query(question)

    # 2. Run the evidence-analysis pipeline
    evidence_result = run_evidence_pipeline(
        question=question,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        claim_threshold=claim_threshold
    )

    # 3. Generate a grounded answer only when
    #    evidence has been successfully retrieved
    answer = None

    if evidence_result["evidence"]:

        answer_context = evidence_to_answer_context(
            evidence_result["evidence"]
        )

        answer = generate_answer(
            question,
            answer_context
        )

    # 4. Return both the final answer and
    #    the complete evidence audit
    return {
        "question": question,
        "query_type": query_type,
        "route": "evidence_audit",
        "answer": answer,
        "result": evidence_result
    }