
"""
Claim-level evidence audit metrics for ResearchLens.

Distinguishes analysis coverage from evidence sufficiency.
All percentages describe the evidence supplied to the audit,
not the completeness of the entire research literature.
"""

from dataclasses import dataclass
from typing import List

from backend.analysis.cross_paper_analysis import (
    EvidenceRelationship
)


@dataclass
class EvidenceAudit:
    claim: str

    total_evidence: int
    analyzed_evidence: int
    unanalyzed_evidence: int

    supporting_evidence: int
    qualifying_evidence: int
    potential_conflicts: int
    insufficient_evidence: int

    source_count: int

    evidence_coverage: float
    evidence_sufficiency: float
    source_diversity: float
    unresolved_rate: float


def audit_evidence_relationships(
    claim: str,
    relationships: List[EvidenceRelationship],
    total_evidence: int | None = None
) -> EvidenceAudit:
    """
    Audit the relationship results for one claim.

    Coverage:
        Classified items / candidate evidence items.

    Sufficiency:
        Items not classified as INSUFFICIENT_EVIDENCE
        / classified items.

    Source diversity:
        Distinct source papers / classified items.

    Unresolved rate:
        POTENTIAL_CONFLICT and INSUFFICIENT_EVIDENCE
        / classified items.
    """

    analyzed_evidence = len(relationships)

    if total_evidence is None:
        total_evidence = analyzed_evidence

    if total_evidence < 0:
        raise ValueError(
            "Total evidence cannot be negative."
        )

    if analyzed_evidence > total_evidence:
        raise ValueError(
            "Analyzed evidence cannot exceed total evidence."
        )

    allowed_relationships = {
        "SUPPORT",
        "QUALIFY",
        "POTENTIAL_CONFLICT",
        "INSUFFICIENT_EVIDENCE",
    }

    for item in relationships:
        if item.relationship not in allowed_relationships:
            raise ValueError(
                f"Invalid relationship: {item.relationship}"
            )

    supporting_evidence = sum(
        item.relationship == "SUPPORT"
        for item in relationships
    )

    qualifying_evidence = sum(
        item.relationship == "QUALIFY"
        for item in relationships
    )

    potential_conflicts = sum(
        item.relationship == "POTENTIAL_CONFLICT"
        for item in relationships
    )

    insufficient_evidence = sum(
        item.relationship == "INSUFFICIENT_EVIDENCE"
        for item in relationships
    )

    unanalyzed_evidence = (
        total_evidence - analyzed_evidence
    )

    unique_sources = {
        item.paper
        for item in relationships
        if item.paper
    }

    source_count = len(unique_sources)

    evidence_coverage = (
        analyzed_evidence / total_evidence * 100
        if total_evidence
        else 0.0
    )

    evidence_sufficiency = (
        (
            analyzed_evidence - insufficient_evidence
        ) / analyzed_evidence * 100
        if analyzed_evidence
        else 0.0
    )

    source_diversity = (
        source_count / analyzed_evidence * 100
        if analyzed_evidence
        else 0.0
    )

    unresolved_count = (
        potential_conflicts + insufficient_evidence
    )

    unresolved_rate = (
        unresolved_count / analyzed_evidence * 100
        if analyzed_evidence
        else 0.0
    )

    return EvidenceAudit(
        claim=claim,
        total_evidence=total_evidence,
        analyzed_evidence=analyzed_evidence,
        unanalyzed_evidence=unanalyzed_evidence,
        supporting_evidence=supporting_evidence,
        qualifying_evidence=qualifying_evidence,
        potential_conflicts=potential_conflicts,
        insufficient_evidence=insufficient_evidence,
        source_count=source_count,
        evidence_coverage=evidence_coverage,
        evidence_sufficiency=evidence_sufficiency,
        source_diversity=source_diversity,
        unresolved_rate=unresolved_rate
    )