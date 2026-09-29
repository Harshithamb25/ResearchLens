
from dataclasses import dataclass
from typing import Optional


@dataclass
class Evidence:
    """A source-grounded passage and its extracted research context."""

    paper: str
    page: int
    evidence_text: str

    claim: Optional[str] = None
    dataset: Optional[str] = None
    method: Optional[str] = None
    metric: Optional[str] = None
    conditions: Optional[str] = None

    retrieval_score: Optional[float] = None
    chunk_id: int | str | None = None

    # Attribution of the extracted claim, not merely the passage.
    evidence_scope: str = "UNCERTAIN"

    # The research contribution represented by the claim.
    contribution_type: str = "UNCERTAIN"

    # Concise explanation of the attribution decision.
    attribution_reason: Optional[str] = None

    # A separate background-study claim, when present.
    related_work_claim: Optional[str] = None

    # Retrieval objectives are hints, not verified section labels.
    retrieval_intents: tuple[str, ...] = ()