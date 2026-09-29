
"""
ResearchLens: evidence-grounded thematic comparisons.

Compares findings from different papers discussing a
shared research topic. Thematic overlap does not imply
agreement, contradiction or independent corroboration.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any

from google.genai import types

from backend.generation.llm_service import client, MODEL_NAME

logger = logging.getLogger(__name__)

MAX_FINDINGS_PER_PAPER = 4

ALLOWED_RELATIONSHIPS = {
    "COMMON_CONCERN",
    "COMPLEMENTARY_FINDINGS",
    "PROBLEM_AND_PROPOSED_SOLUTION",
    "RELATED_FINDINGS",
    "INSUFFICIENT_EVIDENCE",
}

RELATIONSHIP_LABELS = {
    "COMMON_CONCERN": "Common research concern",
    "COMPLEMENTARY_FINDINGS": "Complementary findings",
    "PROBLEM_AND_PROPOSED_SOLUTION": (
        "Problem and proposed solution"
    ),
    "RELATED_FINDINGS": (
        "Related findings; agreement not established"
    ),
    "INSUFFICIENT_EVIDENCE": (
        "Insufficient evidence for comparison"
    ),
}


def _finding(evidence: Any) -> dict[str, Any]:
    return {
        "paper": evidence.paper,
        "page": evidence.page,
        "chunk_id": evidence.chunk_id,
        "claim": evidence.claim,
        "method": evidence.method,
        "dataset": evidence.dataset,
        "conditions": evidence.conditions,
    }


def _source_groups(
    theme: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_paper = defaultdict(list)
    seen = set()

    for item in theme.get("evidence", []):
        if not item.paper or not item.claim:
            continue

        key = (
            item.paper,
            item.page,
            item.chunk_id,
            item.claim,
        )

        if key in seen:
            continue

        seen.add(key)
        by_paper[item.paper].append(_finding(item))

    return {
        paper: findings[:MAX_FINDINGS_PER_PAPER]
        for paper, findings in sorted(by_paper.items())
    }


def _fallback(
    theme: dict[str, Any],
    by_paper: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    papers = sorted(by_paper)

    source_summaries = [
        {
            "paper": paper,
            "findings": findings,
        }
        for paper, findings in by_paper.items()
    ]

    return {
        "theme_id": theme["theme_id"],
        "theme": theme["theme"],
        "relationship": "RELATED_FINDINGS",
        "relationship_label": (
            RELATIONSHIP_LABELS["RELATED_FINDINGS"]
        ),
        "explanation": (
            f"Findings from {len(papers)} papers concern "
            f"{theme['theme'].lower()}. "
            "Their methods, observations or research "
            "objectives may differ. The available "
            "evidence does not establish agreement "
            "or contradiction."
        ),
        "shared_concern": theme["theme"],
        "important_differences": [],
        "limitations": [
            "This is a topic-level comparison. "
            "Inspect the source passages before "
            "drawing method-specific conclusions."
        ],
        "source_count": len(by_paper),
        "papers": papers,
        "source_summaries": source_summaries,
        "findings": [
            finding
            for summary in source_summaries
            for finding in summary["findings"]
        ],
        "agreement_established": False,
        "conflict_established": False,
        "analysis_method": "Conservative thematic comparison",
    }


def _model_payload(
    theme: dict[str, Any],
    by_paper: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    findings = []

    for paper, items in by_paper.items():
        for item in items:
            findings.append({
                "source_id": f"S{len(findings) + 1}",
                "paper": paper,
                "page": item["page"],
                "claim": item["claim"],
                "method": item["method"],
                "dataset": item["dataset"],
                "conditions": item["conditions"],
            })

    return {
        "theme": theme["theme"],
        "findings": findings,
    }


def _analyze_with_gemini(
    payload: dict[str, Any],
) -> dict[str, Any]:
    instructions = """
You are the ResearchLens thematic comparison analyst.

Compare findings from research papers using ONLY the
supplied evidence.

Your explanation must be concise, specific and written
in your own words. Do not copy entire source sentences.

Allowed relationships:

COMMON_CONCERN:
Different papers explicitly identify the same broad
research concern. This does not prove agreement on
a method-specific claim.

COMPLEMENTARY_FINDINGS:
Different papers provide distinct findings that
illuminate different aspects of a shared topic.

PROBLEM_AND_PROPOSED_SOLUTION:
One paper identifies a problem while another proposes
an approach intended to address that type of problem.
Do not claim that the solution is experimentally proven
unless the supplied findings establish that.

RELATED_FINDINGS:
The papers discuss a related topic, but a more specific
relationship is not established.

INSUFFICIENT_EVIDENCE:
The supplied findings do not permit a meaningful
comparison.

Strict requirements:

1. Compare the paper's own findings where possible.
2. If a finding describes another study, do not
   attribute that study's result to the current paper.
3. Never invent methods, datasets, measurements,
   experimental results, advantages or limitations.
4. Do not convert detection range, journey time,
   cost or other measurements into accuracy.
5. Do not infer agreement from thematic overlap.
6. Distinguish proposed methods from evaluated results.
7. Use only source IDs provided in the input.
8. Support the explanation with findings from at least
   two different papers.
9. Write two or three concise explanatory sentences.
10. Express differences and qualifications in your
    own words, retaining exact technical terms and
    numerical measurements where necessary.
11. If the evidence is ambiguous, select
    RELATED_FINDINGS or INSUFFICIENT_EVIDENCE.

Return one JSON object with EXACTLY these fields:

{
  "relationship": "one allowed relationship",
  "explanation": "two or three concise sentences",
  "shared_concern": "short description or null",
  "important_differences": [
    "a concise evidence-grounded difference"
  ],
  "limitations": [
    "a concise qualification"
  ],
  "source_ids": ["S1", "S2"]
}

INPUT:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            instructions
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    raw = (response.text or "").strip()

    if not raw:
        raise ValueError(
            "Gemini returned an empty theme comparison."
        )

    parsed = json.loads(raw)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Theme comparison must be a JSON object."
        )

    return parsed


def _validate_analysis(
    analysis: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    relationship = analysis.get("relationship")

    if relationship not in ALLOWED_RELATIONSHIPS:
        raise ValueError(
            "Invalid thematic relationship category."
        )

    explanation = analysis.get("explanation")

    if (
        not isinstance(explanation, str)
        or not explanation.strip()
        or len(explanation) > 1800
    ):
        raise ValueError(
            "Invalid thematic explanation."
        )

    allowed_sources = {
        item["source_id"]: item["paper"]
        for item in payload["findings"]
    }

    source_ids = analysis.get("source_ids")

    if (
        not isinstance(source_ids, list)
        or not source_ids
        or any(
            not isinstance(source_id, str)
            or source_id not in allowed_sources
            for source_id in source_ids
        )
    ):
        raise ValueError(
            "Theme comparison contains invalid sources."
        )

    represented_papers = {
        allowed_sources[source_id]
        for source_id in source_ids
    }

    if len(represented_papers) < 2:
        raise ValueError(
            "Comparison must reference at least "
            "two different papers."
        )

    differences = analysis.get("important_differences")
    limitations = analysis.get("limitations")

    if not isinstance(differences, list):
        raise ValueError(
            "Invalid important_differences field."
        )

    if not isinstance(limitations, list):
        raise ValueError(
            "Invalid limitations field."
        )

    for item in differences + limitations:
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item) > 500
        ):
            raise ValueError(
                "Invalid comparison detail."
            )

    shared_concern = analysis.get("shared_concern")

    if (
        shared_concern is not None
        and (
            not isinstance(shared_concern, str)
            or len(shared_concern) > 500
        )
    ):
        raise ValueError(
            "Invalid shared_concern field."
        )

    return {
        "relationship": relationship,
        "relationship_label": (
            RELATIONSHIP_LABELS[relationship]
        ),
        "explanation": explanation.strip(),
        "shared_concern": (
            shared_concern.strip()
            if isinstance(shared_concern, str)
            else None
        ),
        "important_differences": [
            item.strip() for item in differences
        ],
        "limitations": [
            item.strip() for item in limitations
        ],
        "source_ids": list(dict.fromkeys(source_ids)),
    }


def compare_cross_paper_themes(
    themes: list[dict[str, Any]],
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """
    Compare themes supported by at least two papers.

    API failures return conservative topic-level
    comparisons without claiming agreement.
    """
    comparisons = []

    for theme in themes:
        if not theme.get("cross_paper"):
            continue

        by_paper = _source_groups(theme)

        if len(by_paper) < 2:
            continue

        comparison = _fallback(theme, by_paper)

        if use_llm:
            payload = _model_payload(theme, by_paper)

            for attempt in range(2):
                try:
                    analysis = _analyze_with_gemini(payload)

                    validated = _validate_analysis(
                        analysis,
                        payload,
                    )

                    comparison.update(validated)

                    comparison["analysis_method"] = (
                        "Gemini-assisted thematic comparison"
                    )

                    break

                except Exception as error:
                    logger.warning(
                        "Theme comparison failed for %s "
                        "(attempt %s): %s",
                        theme["theme_id"],
                        attempt + 1,
                        error,
                    )

                    if attempt == 0:
                        time.sleep(1)

        comparisons.append(comparison)

    return comparisons