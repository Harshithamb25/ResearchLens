"""
Evidence builder for ResearchLens.

Converts reranked retrieval results into structured
Evidence objects enriched with research context.

Evidence extraction is performed in batch to reduce
the number of LLM requests.
"""

from backend.analysis.evidence import Evidence
from backend.analysis.evidence_extractor import (
    extract_evidence_context_batch
)


def build_evidence(reranked_results):
    """
    Convert reranked retrieval results into structured
    Evidence objects.

    All evidence passages are sent to the evidence
    extraction component in a single batch request.

    Args:
        reranked_results:
            List of reranked retrieval results.

    Returns:
        List of Evidence objects.
    """

    if not reranked_results:
        return []

    evidence_texts = [
        item["text"]
        for item in reranked_results
    ]

    extracted_results = extract_evidence_context_batch(
        evidence_texts
    )

    if len(extracted_results) != len(reranked_results):
        raise ValueError(
            "Number of extracted evidence results does not "
            "match number of reranked results."
        )

    evidence_items = []

    for item, extracted in zip(
        reranked_results,
        extracted_results
    ):

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
                retrieval_score=item.get("reranker_score")
            )
        )

    return evidence_items


def evidence_to_answer_context(evidence_items):
    """
    Convert Evidence objects into the context format
    expected by the grounded answer generator.
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