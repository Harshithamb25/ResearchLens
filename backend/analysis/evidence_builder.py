"""
Evidence builder for ResearchLens.

Converts reranked retrieval results into structured
Evidence objects enriched with research context.
"""

from backend.analysis.evidence import Evidence
from backend.analysis.evidence_extractor import (
    extract_evidence_context
)


def build_evidence(reranked_results):
    evidence_items = []

    for item in reranked_results:

        extracted = extract_evidence_context(
            item["text"]
        )

        evidence_items.append(
            Evidence(
                paper=item["document"],
                page=item["page"],
                evidence_text=item["text"],
                claim=extracted.get("claim"),
                dataset=extracted.get("dataset"),
                method=extracted.get("method"),
                metric=extracted.get("metric"),
                conditions=extracted.get("conditions"),
                retrieval_score=item.get(
                    "reranker_score"
                )
            )
        )

    return evidence_items


def evidence_to_answer_context(evidence_items):
    """
    Convert structured Evidence objects into the
    raw retrieval format expected by generate_answer().
    """

    return [
        {
            "document": evidence.paper,
            "page": evidence.page,
            "chunk_id": None,
            "text": evidence.evidence_text
        }
        for evidence in evidence_items
    ]