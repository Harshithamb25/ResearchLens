from dataclasses import dataclass
from typing import Optional


@dataclass
class Evidence:
    """
    Structured representation of a piece of
    research evidence retrieved from a paper.
    """

    paper: str
    page: int
    evidence_text: str

    claim: Optional[str] = None

    dataset: Optional[str] = None
    method: Optional[str] = None
    metric: Optional[str] = None
    conditions: Optional[str] = None

    retrieval_score: Optional[float] = None
    chunk_id: int | None = None