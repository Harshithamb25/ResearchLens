"""
Evidence analysis pipeline for ResearchLens.

Pipeline:

    Query
      ↓
    Semantic Retrieval
      ↓
    Result Formatting
      ↓
    Cross-Encoder Reranking
      ↓
    Evidence Context Extraction
      ↓
    Semantic Claim Grouping
      ↓
    Cross-Paper Relationship Analysis
      ↓
    Evidence Audit
"""

from backend.retrieval.retriever import search
from backend.retrieval.result_formatter import format_retrieval_results
from backend.retrieval.reranker import rerank

from backend.analysis.evidence_builder import build_evidence
from backend.analysis.claim_grouper import group_claims
from backend.analysis.relationship_analyzer import (
    build_evidence_relationships
)
from backend.analysis.evidence_audit import (
    audit_evidence_relationships
)


def run_evidence_pipeline(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75
):
    """
    Run the complete evidence analysis pipeline.

    Returns:
        Dictionary containing:
        - question
        - evidence
        - claim_groups
        - relationships
        - audits
    """

    retrieval_results = search(
        question,
        top_k=retrieval_k
    )

    formatted_results = format_retrieval_results(
        retrieval_results
    )

    if not formatted_results:
        return {
            "question": question,
            "evidence": [],
            "claim_groups": [],
            "relationships": [],
            "audits": []
        }

    reranked_results = rerank(
        question,
        formatted_results,
        top_k=rerank_k
    )

    evidence_items = build_evidence(
        reranked_results
    )

    claim_groups = group_claims(
        evidence_items,
        threshold=claim_threshold
    )

    all_relationships = []
    audits = []

    for claim_group in claim_groups:

        relationships = build_evidence_relationships(
            claim_group
        )

        all_relationships.extend(
            relationships
        )

        audit = audit_evidence_relationships(
            claim=claim_group["claim"],
            relationships=relationships,
            total_evidence=len(
                claim_group["evidence"]
            )
        )

        audits.append(audit)

    return {
        "question": question,
        "evidence": evidence_items,
        "claim_groups": claim_groups,
        "relationships": all_relationships,
        "audits": audits
    }