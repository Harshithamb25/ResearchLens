"""
Central orchestration layer for ResearchLens.

This module connects query classification to the
evidence-grounded research analysis pipeline.
"""

from backend.analysis.query_router import classify_query
from backend.analysis.evidence_pipeline import (
    run_evidence_pipeline
)


def process_query(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75
):
    """
    Process a ResearchLens question.

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

    return {
        "question": question,
        "query_type": query_type,
        "route": "evidence_audit",
        "result": evidence_result
    }