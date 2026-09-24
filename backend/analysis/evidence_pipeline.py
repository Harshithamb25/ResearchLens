
"""
Evidence analysis pipeline for ResearchLens.

Performs retrieval, reranking, evidence extraction,
claim grouping, relationship analysis and auditing.

Relationship-analysis failures are recorded per claim
group rather than silently treated as classifications.
"""

import logging

from backend.retrieval.retriever import search
from backend.retrieval.result_formatter import (
    format_retrieval_results
)
from backend.retrieval.reranker import rerank

from backend.analysis.evidence_builder import build_evidence
from backend.analysis.claim_grouper import group_claims
from backend.analysis.relationship_analyzer import (
    build_evidence_relationships
)
from backend.analysis.evidence_audit import (
    audit_evidence_relationships
)


logger = logging.getLogger(__name__)


def run_evidence_pipeline(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75
):
    """
    Run the evidence-analysis pipeline.

    A failed relationship analysis affects only its
    claim group. Earlier and later groups are retained.

    Status values:
        completed
        partial
        failed
        no_evidence
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
            "status": "no_evidence",
            "evidence": [],
            "claim_groups": [],
            "relationships": [],
            "audits": [],
            "analysis_failures": []
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
    analysis_failures = []

    for group_index, claim_group in enumerate(
        claim_groups,
        start=1
    ):
        claim = claim_group["claim"]
        candidate_count = len(
            claim_group["evidence"]
        )

        try:
            relationships = build_evidence_relationships(
                claim_group
            )

            # The current analyzer requires one validated
            # classification for every evidence item.
            # Reject incomplete results instead of
            # presenting them as a successful analysis.
            if len(relationships) != candidate_count:
                raise ValueError(
                    "Relationship count does not match "
                    "the candidate evidence count."
                )

            audit = audit_evidence_relationships(
                claim=claim,
                relationships=relationships,
                total_evidence=candidate_count
            )

        except Exception:
            logger.exception(
                "Relationship analysis failed for "
                "claim group %s",
                group_index
            )

            # Preserve the failure without exposing raw
            # provider errors or API details to the UI.
            analysis_failures.append({
                "group_index": group_index,
                "claim": claim,
                "candidate_evidence": candidate_count,
                "reason": "relationship_analysis_failed"
            })

            # No classifications were successfully
            # returned for this group.
            relationships = []

            audit = audit_evidence_relationships(
                claim=claim,
                relationships=[],
                total_evidence=candidate_count
            )

        all_relationships.extend(
            relationships
        )

        audits.append(audit)

    if not claim_groups:
        status = "no_evidence"
    elif len(analysis_failures) == len(claim_groups):
        status = "failed"
    elif analysis_failures:
        status = "partial"
    else:
        status = "completed"

    return {
        "question": question,
        "status": status,
        "evidence": evidence_items,
        "claim_groups": claim_groups,
        "relationships": all_relationships,
        "audits": audits,
        "analysis_failures": analysis_failures
    }