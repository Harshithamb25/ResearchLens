"""
Evidence audit metrics for ResearchLens.

The audit summarizes how well a claim is supported by the
analyzed evidence and how broadly that evidence is sourced
across research papers.
"""

from dataclasses import dataclass
from typing import List

from backend.analysis.cross_paper_analysis import (
    EvidenceRelationship
)


@dataclass
class EvidenceAudit:
    """
    Summary of the evidence supporting a research claim.
    """

    claim: str

    total_evidence: int
    analyzed_evidence: int

    supporting_evidence: int
    qualifying_evidence: int
    potential_conflicts: int
    insufficient_evidence: int

    source_count: int

    evidence_coverage: float
    source_diversity: float
    unresolved_rate: float


def audit_evidence_relationships(
    claim: str,
    relationships: List[EvidenceRelationship],
    total_evidence: int | None = None
) -> EvidenceAudit:
    """
    Calculate evidence-audit metrics for a claim.

    Metrics:

    evidence_coverage:
        Percentage of retrieved evidence that received
        a relationship analysis.

    source_count:
        Number of distinct research papers contributing
        analyzed evidence.

    source_diversity:
        Percentage of analyzed evidence items that come
        from distinct papers.

        Formula:
            unique sources / analyzed evidence * 100

        This is a simple provenance-distribution metric,
        not a statistical diversity index.

    unresolved_rate:
        Percentage of analyzed evidence classified as
        POTENTIAL_CONFLICT or INSUFFICIENT_EVIDENCE.
    """

    if total_evidence is None:
        total_evidence = len(relationships)

    analyzed_evidence = len(relationships)

    supporting_evidence = sum(
        1
        for item in relationships
        if item.relationship == "SUPPORT"
    )

    qualifying_evidence = sum(
        1
        for item in relationships
        if item.relationship == "QUALIFY"
    )

    potential_conflicts = sum(
        1
        for item in relationships
        if item.relationship == "POTENTIAL_CONFLICT"
    )

    insufficient_evidence = sum(
        1
        for item in relationships
        if item.relationship == "INSUFFICIENT_EVIDENCE"
    )

    unique_sources = {
        item.paper
        for item in relationships
        if item.paper
    }

    source_count = len(unique_sources)

    if total_evidence > 0:
        evidence_coverage = (
            analyzed_evidence / total_evidence
        ) * 100
    else:
        evidence_coverage = 0.0

    if analyzed_evidence > 0:
        source_diversity = (
            source_count / analyzed_evidence
        ) * 100
    else:
        source_diversity = 0.0

    unresolved_count = (
        potential_conflicts
        + insufficient_evidence
    )

    if analyzed_evidence > 0:
        unresolved_rate = (
            unresolved_count / analyzed_evidence
        ) * 100
    else:
        unresolved_rate = 0.0

    return EvidenceAudit(
        claim=claim,
        total_evidence=total_evidence,
        analyzed_evidence=analyzed_evidence,
        supporting_evidence=supporting_evidence,
        qualifying_evidence=qualifying_evidence,
        potential_conflicts=potential_conflicts,
        insufficient_evidence=insufficient_evidence,
        source_count=source_count,
        evidence_coverage=evidence_coverage,
        source_diversity=source_diversity,
        unresolved_rate=unresolved_rate
    )