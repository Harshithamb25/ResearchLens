from analysis.evidence import Evidence
from analysis.evidence_extractor import extract_evidence_context


def build_evidence(reranked_results):
    """
    Convert reranked retrieval results into enriched
    Evidence objects containing extracted research context.
    """

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
                retrieval_score=item.get("reranker_score")
            )
        )

    return evidence_items