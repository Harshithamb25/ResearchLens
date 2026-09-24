from dataclasses import dataclass
from typing import List

from backend.analysis.cross_paper_analysis import EvidenceRelationship


@dataclass
class EvidenceAudit:
    """
    Summarizes the evidence coverage and relationship
    distribution for a research claim.
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
    unresolved_rate: float


def audit_evidence_relationships(
    claim: str,
    relationships: List[EvidenceRelationship],
    total_evidence: int | None = None
) -> EvidenceAudit:
    """
    Audit the evidence relationships associated with a claim.

    The audit reports evidence coverage, source diversity,
    relationship distribution, and unresolved evidence.

    No claim is declared objectively true or false.
    The audit only summarizes the supplied evidence.
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
        unresolved_rate=unresolved_rate
    )

