
"""Build source-attributed Evidence objects for ResearchLens.

Responsibilities:
- Deduplicate retrieved passages.
- Extract claims without merging unrelated sources.
- Exclude empty claims and explicitly identified related work.
- Reject clearly unsupported causal interpretations.
- Preserve original passages and quantitative evidence.
- Retain attribution metadata for downstream synthesis.
"""

import logging
import re

from backend.analysis.evidence import Evidence
from backend.analysis.evidence_extractor import (
    extract_evidence_context_batch,
)


logger = logging.getLogger(__name__)


# Strong indications that the extractor is claiming
# an intervention caused an observed improvement.
CAUSAL_CLAIM_PATTERNS = (
    r"\b(?:system|method|approach|algorithm|"
    r"framework|technology|implementation)\s+"
    r"(?:caused|resulted in|led to|achieved|"
    r"produced|reduced|improved)\b",
    r"\b(?:due to|because of|as a result of)\s+"
    r"(?:the|our|this)\s+"
    r"(?:system|method|approach|algorithm|"
    r"framework|technology|implementation)\b",
)

# Statements like "holiday journeys were shorter than
# normal-day journeys" describe a comparison, not
# necessarily an effect of the proposed system.
OBSERVATIONAL_PATTERNS = (
    r"\bcompared (?:with|to)\b",
    r"\bin comparison (?:with|to)\b",
    r"\bas compared to\b",
    r"\bon (?:public )?holidays?\b",
    r"\bon (?:a )?normal day\b",
    r"\b(?:observed|measured|analysis shows|"
    r"data collected)\b",
)

# Attribution values indicating that the passage
# describes another study rather than the source
# paper's own contribution.
RELATED_SCOPE_VALUES = {
    "RELATED_WORK",
    "PRIOR_WORK",
    "CITED_WORK",
    "OTHER_STUDY",
    "THIRD_PARTY",
}

RELATED_CONTRIBUTION_VALUES = {
    "RELATED_WORK",
    "PRIOR_WORK",
    "CITED_WORK",
    "OTHER_STUDY",
}


def _optional_text(value):
    """Normalize optional extractor text fields."""
    if not isinstance(value, str):
        return None

    cleaned = " ".join(value.split()).strip()

    if cleaned.casefold() in {
        "none",
        "null",
        "n/a",
        "not applicable",
        "unknown",
    }:
        return None

    return cleaned or None


def _normalized_label(value, default="UNCERTAIN"):
    """Normalize attribution labels without changing their meaning."""
    cleaned = _optional_text(value)

    if not cleaned:
        return default

    return re.sub(
        r"[\s-]+",
        "_",
        cleaned.upper(),
    )


def _is_explicit_related_work(extracted):
    """Reject only passages explicitly labeled as prior work."""
    scope = _normalized_label(
        extracted.get("evidence_scope")
    )

    contribution = _normalized_label(
        extracted.get("contribution_type")
    )

    return (
        scope in RELATED_SCOPE_VALUES
        or contribution in RELATED_CONTRIBUTION_VALUES
    )


def _claims_unsupported_causation(claim, source_text):
    """
    Detect a narrow class of causal overstatements.

    A causal claim is rejected when the source passage
    reports an observational comparison but contains
    no matching statement that the proposed system
    caused the reported improvement.

    This is a conservative safeguard, not a complete
    semantic verification system.
    """
    if not claim or not source_text:
        return False

    claim_is_causal = any(
        re.search(pattern, claim, re.IGNORECASE)
        for pattern in CAUSAL_CLAIM_PATTERNS
    )

    if not claim_is_causal:
        return False

    source_is_observational = any(
        re.search(pattern, source_text, re.IGNORECASE)
        for pattern in OBSERVATIONAL_PATTERNS
    )

    if not source_is_observational:
        return False

    source_supports_causation = any(
        re.search(pattern, source_text, re.IGNORECASE)
        for pattern in CAUSAL_CLAIM_PATTERNS
    )

    return not source_supports_causation


def _deduplicate_results(results):
    """Keep one instance of each paper/page/chunk."""
    unique = []
    seen = set()

    for item in results:
        if not isinstance(item, dict):
            continue

        document = (
            item.get("document_id")
            or item.get("document")
        )

        text = _optional_text(item.get("text"))

        if not document or not text:
            continue

        key = (
            str(document),
            str(item.get("page")),
            str(item.get("chunk_id")),
            (
                text[:100]
                if item.get("chunk_id") is None
                else ""
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def build_evidence(reranked_results):
    """
    Extract and validate source-attributed evidence.

    Returns only Evidence objects containing a usable
    claim from the source paper. Rejected passages
    remain available through retrieval but are not
    passed to claim grouping.
    """
    if not reranked_results:
        return []

    unique_results = _deduplicate_results(
        reranked_results
    )

    if not unique_results:
        return []

    extraction_inputs = [
        {
            "text": item["text"],
            "document": item.get("document"),
            "page": item.get("page"),
            "retrieval_intents": (
                item.get("retrieval_intents") or []
            ),
        }
        for item in unique_results
    ]

    extracted_results = extract_evidence_context_batch(
        extraction_inputs
    )

    if len(extracted_results) != len(unique_results):
        raise ValueError(
            "Extraction result count does not match "
            "the number of unique passages."
        )

    evidence_items = []

    rejected_empty = 0
    rejected_related = 0
    rejected_causal = 0

    for item, extracted in zip(
        unique_results,
        extracted_results,
    ):
        if not isinstance(extracted, dict):
            raise ValueError(
                "Evidence extraction returned "
                "an invalid record."
            )

        claim = _optional_text(
            extracted.get("claim")
        )

        # Do not create Evidence objects with
        # empty or placeholder claims.
        if not claim:
            rejected_empty += 1
            continue

        # A source passage can discuss another paper.
        # Such passages must not become claims
        # attributed to the current paper.
        if _is_explicit_related_work(extracted):
            rejected_related += 1
            continue

        source_text = item["text"]

        # Reject a clear observational-to-causal
        # overstatement rather than silently
        # rewriting the extractor's claim.
        if _claims_unsupported_causation(
            claim,
            source_text,
        ):
            logger.warning(
                "Rejected potentially unsupported "
                "causal claim: paper=%s page=%s "
                "chunk=%s claim=%s",
                item.get("document"),
                item.get("page"),
                item.get("chunk_id"),
                claim,
            )

            rejected_causal += 1
            continue

        evidence_items.append(
            Evidence(
                paper=item["document"],
                page=item["page"],
                chunk_id=item.get("chunk_id"),

                # Always preserve the original passage.
                # This lets synthesis and citation
                # validation inspect the actual source.
                evidence_text=source_text,

                claim=claim,

                dataset=_optional_text(
                    extracted.get("dataset")
                ),

                method=_optional_text(
                    extracted.get("method")
                ),

                metric=_optional_text(
                    extracted.get("metric")
                ),

                conditions=_optional_text(
                    extracted.get("conditions")
                ),

                retrieval_score=item.get(
                    "reranker_score"
                ),

                evidence_scope=_normalized_label(
                    extracted.get("evidence_scope")
                ),

                contribution_type=_normalized_label(
                    extracted.get("contribution_type")
                ),

                attribution_reason=_optional_text(
                    extracted.get(
                        "attribution_reason"
                    )
                ),

                related_work_claim=_optional_text(
                    extracted.get(
                        "related_work_claim"
                    )
                ),

                retrieval_intents=tuple(
                    item.get(
                        "retrieval_intents"
                    ) or []
                ),
            )
        )

    logger.info(
        "Evidence extraction: %s passages, "
        "%s valid claims, %s empty, "
        "%s related-work, %s causal rejections",
        len(unique_results),
        len(evidence_items),
        rejected_empty,
        rejected_related,
        rejected_causal,
    )

    return evidence_items


def evidence_to_answer_context(evidence_items):
    """
    Build answer context from validated evidence.

    Include original text, quantitative fields and
    attribution metadata so downstream synthesis
    can verify each claim against its source.
    """
    context = []

    for evidence in evidence_items:
        if not _optional_text(evidence.claim):
            continue

        context.append(
            {
                "document": evidence.paper,
                "page": evidence.page,
                "chunk_id": evidence.chunk_id,

                "text": evidence.evidence_text,
                "claim": evidence.claim,

                "dataset": evidence.dataset,
                "method": evidence.method,
                "metric": evidence.metric,
                "conditions": evidence.conditions,

                "evidence_scope": (
                    evidence.evidence_scope
                ),

                "contribution_type": (
                    evidence.contribution_type
                ),

                "attribution_reason": (
                    evidence.attribution_reason
                ),

                "related_work_claim": (
                    evidence.related_work_claim
                ),

                "retrieval_intents": list(
                    evidence.retrieval_intents
                ),
            }
        )

    return context