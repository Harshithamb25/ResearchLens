from dataclasses import dataclass
from typing import List

from backend.analysis.evidence import Evidence


@dataclass
class EvidenceRelationship:
    """
    Represents the relationship between a claim
    and evidence from a research paper.
    """

    claim: str
    paper: str
    page: int
    relationship: str
    explanation: str
    evidence: Evidence

