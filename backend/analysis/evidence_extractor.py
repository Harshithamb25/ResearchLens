
"""Reliable, source-aware evidence extraction for ResearchLens.

Extracts independently attributed claims from research passages.
Preserves quantitative findings and observational comparisons.
Supports validated JSON, retries, and individual-passage fallback.
"""

import json
import logging
import time

from google.genai import types

from backend.generation.llm_service import (
    _get_client,
    MODEL_NAME,
)


logger = logging.getLogger(__name__)

BATCH_SIZE = 2
MAX_ATTEMPTS = 2
MAX_PASSAGE_CHARS = 6500

OWN_TYPES = {
    "OWN_METHOD",
    "OWN_EVALUATION",
    "OWN_RESULT",
    "OWN_LIMITATION",
}

SCOPES = {
    "OWN_WORK",
    "RELATED_WORK",
    "BACKGROUND",
    "MIXED",
    "UNCERTAIN",
}

CONTRIBUTION_TYPES = OWN_TYPES | {
    "RELATED_WORK",
    "BACKGROUND",
    "UNCERTAIN",
}

TEXT_FIELDS = (
    "claim",
    "dataset",
    "method",
    "metric",
    "conditions",
    "attribution_reason",
    "related_work_claim",
)

CLAIM_FIELDS = (
    "claim",
    "dataset",
    "method",
    "metric",
    "conditions",
)

RULES = """
You are a careful, source-aware academic evidence extractor.

TASK
For each passage, extract the current paper's original,
explicitly supported research contribution. Do not confuse
the current authors' work with cited studies or background.

Treat every passage independently. The supplied filename
identifies the source paper; it does not establish that
every statement in the passage belongs to its authors.

ATTRIBUTION

evidence_scope:
- OWN_WORK: Clearly describes the current authors' work.
- RELATED_WORK: Describes another study's work.
- BACKGROUND: General context, motivation or definitions.
- MIXED: Contains both clearly identifiable original
  contributions and other material.
- UNCERTAIN: Authorship or evidential support is unclear.

contribution_type:
- OWN_METHOD: Original implementation, system or algorithm.
- OWN_EVALUATION: Original experimental setup or field trial.
- OWN_RESULT: Original measured or observed finding.
- OWN_LIMITATION: Limitation of the authors' own work.
- RELATED_WORK: Finding attributed to another study.
- BACKGROUND: General information.
- UNCERTAIN: Contribution cannot be established.

A passage can continue a related-work discussion from a
previous chunk. Expressions such as "this study" or
"the proposed system" alone do not establish authorship.
If the referent is ambiguous, use UNCERTAIN rather than
attributing the claim to the current paper.

QUANTITATIVE EVIDENCE

Preserve important, explicitly reported numerical findings
and their units, conditions and comparison groups.

When one passage contains several closely related findings,
combine them into one concise, factual claim if doing so
preserves their distinct meanings.

For example, if a passage reports:
- signal detection up to 20 m under specified conditions;
- a field trial involving two public bus services;
- holiday journeys 5–10 minutes shorter than normal-day
  journeys;

the claim should preserve all three findings if clearly
attributed to the current authors.

Do not turn detection distance into accuracy.
Do not invent sample sizes, performance improvements,
cost savings or statistical significance.

CAUSALITY

Distinguish:
- A system's measured technical performance.
- An observed difference between groups or conditions.
- An improvement demonstrably caused by an intervention.

A comparison alone does not establish causation.

If a passage says holiday journey times were 5–10 minutes
shorter than normal-day journey times, report that observed
difference. Do NOT claim the proposed tracking system
caused, achieved or produced the reduction.

Do not describe a reported correlation as an intervention
effect unless the passage explicitly supports that claim.

COSTS AND LIMITATIONS

Preserve stated currencies, amounts, units, time periods
and assumptions. Distinguish modeled costs from observed
operational expenditure.

Distinguish a limitation of the current paper from a
limitation reported about a cited earlier study.

EXTRACTION RULES

1. Extract at most one integrated own-work claim per passage.
2. Preserve distinct quantitative findings when relevant.
3. Paraphrase faithfully; never invent missing information.
4. Do not introduce unsupported causal wording.
5. For MIXED passages, isolate only clearly attributed
   original work.
6. For passages containing only related work, background,
   or uncertain attribution, set claim, dataset, method,
   metric and conditions to null.
7. If the passage describes related work, optionally
   summarize it in related_work_claim.
8. Do not extract research objectives as completed results.
9. Do not treat the filename, retrieval intent or an
   adjacent passage as proof of authorship.
10. Do not infer information absent from the passage.
11. Use null for unavailable optional fields.
12. Return valid JSON only, without Markdown.

OUTPUT

Return exactly one result for every input passage.

Each result must contain:
- evidence_index
- claim
- dataset
- method
- metric
- conditions
- evidence_scope
- contribution_type
- attribution_reason
- related_work_claim

Return an object containing a "results" array.
"""


def _clean_text(value):
    """Normalize optional text without inventing content."""
    if not isinstance(value, str):
        return None

    value = " ".join(value.split()).strip()

    if value.casefold() in {
        "",
        "null",
        "none",
        "n/a",
        "not applicable",
        "not specified",
        "not reported",
        "unknown",
    }:
        return None

    return value


def _empty_result(reason=None):
    """Return an explicitly unverified extraction result."""
    return {
        "claim": None,
        "dataset": None,
        "method": None,
        "metric": None,
        "conditions": None,
        "evidence_scope": "UNCERTAIN",
        "contribution_type": "UNCERTAIN",
        "attribution_reason": reason,
        "related_work_claim": None,
    }


def _normalize_label(value, allowed):
    """Validate labels returned by the language model."""
    label = str(
        value or "UNCERTAIN"
    ).strip().upper().replace("-", "_").replace(" ", "_")

    return (
        label
        if label in allowed
        else "UNCERTAIN"
    )


def _normalize_result(result):
    """Validate and sanitize one extracted evidence record."""
    if not isinstance(result, dict):
        raise ValueError(
            "Individual extraction result is not an object."
        )

    normalized = {
        field: _clean_text(result.get(field))
        for field in TEXT_FIELDS
    }

    scope = _normalize_label(
        result.get("evidence_scope"),
        SCOPES,
    )

    contribution = _normalize_label(
        result.get("contribution_type"),
        CONTRIBUTION_TYPES,
    )

    normalized["evidence_scope"] = scope
    normalized["contribution_type"] = contribution

    # Only clearly identified original contributions
    # may produce claims about the source paper.
    is_own_contribution = (
        scope in {"OWN_WORK", "MIXED"}
        and contribution in OWN_TYPES
    )

    if not is_own_contribution:
        for field in CLAIM_FIELDS:
            normalized[field] = None

    # A claimed original contribution without an actual
    # claim is not useful evidence.
    if is_own_contribution and not normalized["claim"]:
        for field in CLAIM_FIELDS:
            normalized[field] = None

        normalized["evidence_scope"] = "UNCERTAIN"
        normalized["contribution_type"] = "UNCERTAIN"

        normalized["attribution_reason"] = (
            normalized["attribution_reason"]
            or "No supported original contribution extracted."
        )

    return normalized


def _normalize_input(item):
    """Accept a passage string or a passage dictionary."""
    if isinstance(item, str):
        item = {"text": item}

    if not isinstance(item, dict):
        raise TypeError(
            "Evidence input must be a string or dictionary."
        )

    return {
        "text": str(
            item.get("text") or ""
        )[:MAX_PASSAGE_CHARS],
        "document": item.get("document"),
        "page": item.get("page"),
        "retrieval_intents": (
            item.get("retrieval_intents") or []
        ),
    }


def _parse_response(response):
    """Accept a JSON object, array or single record."""
    raw = (
        getattr(response, "text", None) or ""
    ).strip()

    if not raw:
        raise ValueError(
            "Gemini returned an empty extraction response."
        )

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    parsed = json.loads(raw)

    if isinstance(parsed, list):
        return {"results": parsed}

    if isinstance(parsed, dict):
        if isinstance(parsed.get("results"), list):
            return parsed

        if "evidence_scope" in parsed:
            return {"results": [parsed]}

    raise ValueError(
        "Gemini returned an unsupported extraction structure."
    )


def _build_prompt(passages):
    """Construct one independent extraction task per passage."""
    payload = [
        {
            "evidence_index": index,
            **passage,
        }
        for index, passage in enumerate(
            passages,
            start=1,
        )
    ]

    return (
        RULES
        + "\nINPUT PASSAGES:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
        )
    )


def _request_batch(passages):
    """Request structured extraction from Gemini."""
    prompt = _build_prompt(passages)

    response = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    parsed = _parse_response(response)
    results = parsed["results"]

    if len(results) != len(passages):
        raise ValueError(
            f"Expected {len(passages)} extraction results; "
            f"received {len(results)}."
        )

    indexed = {}

    for position, item in enumerate(
        results,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "Extraction result is not a JSON object."
            )

        raw_index = item.get(
            "evidence_index",
            position,
        )

        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            raise ValueError(
                "Invalid evidence_index."
            ) from None

        if index in indexed:
            raise ValueError(
                "Duplicate evidence_index."
            )

        indexed[index] = _normalize_result(item)

    expected = set(
        range(1, len(passages) + 1)
    )

    if set(indexed) != expected:
        raise ValueError(
            "Extraction result indices do not match "
            "the input passages."
        )

    return [
        indexed[index]
        for index in sorted(indexed)
    ]


def _extract_resilient(passages):
    """Retry a failed batch, then isolate individual passages."""
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            return _request_batch(passages)

        except Exception as error:
            last_error = error

            logger.warning(
                "Extraction of %s passages failed "
                "(attempt %s): %s: %s",
                len(passages),
                attempt + 1,
                type(error).__name__,
                error,
            )

            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(1)

    if len(passages) > 1:
        midpoint = len(passages) // 2

        return (
            _extract_resilient(
                passages[:midpoint]
            )
            + _extract_resilient(
                passages[midpoint:]
            )
        )

    logger.error(
        "Individual passage extraction failed.",
        exc_info=(
            type(last_error),
            last_error,
            last_error.__traceback__,
        ) if last_error is not None else False,
    )

    return [
        _empty_result(
            "Extraction failed; attribution unverified."
        )
    ]


def extract_evidence_context(evidence_text):
    """Extract one passage through the validated batch path."""
    return extract_evidence_context_batch(
        [evidence_text]
    )[0]


def extract_evidence_context_batch(evidence_texts):
    """Extract independent passages in small, resilient batches."""
    if not evidence_texts:
        return []

    passages = [
        _normalize_input(item)
        for item in evidence_texts
    ]

    results = []

    for start in range(
        0,
        len(passages),
        BATCH_SIZE,
    ):
        batch = passages[
            start:start + BATCH_SIZE
        ]

        # Empty passages must never produce invented claims.
        if all(
            not passage["text"].strip()
            for passage in batch
        ):
            results.extend(
                _empty_result("Empty source passage.")
                for _ in batch
            )
            continue

        # Extract nonempty passages while preserving
        # their original positions in the batch.
        nonempty_indices = [
            index
            for index, passage in enumerate(batch)
            if passage["text"].strip()
        ]

        nonempty_passages = [
            batch[index]
            for index in nonempty_indices
        ]

        extracted = _extract_resilient(
            nonempty_passages
        )

        restored = [
            _empty_result("Empty source passage.")
            for _ in batch
        ]

        for index, result in zip(
            nonempty_indices,
            extracted,
        ):
            restored[index] = result

        results.extend(restored)

    return results