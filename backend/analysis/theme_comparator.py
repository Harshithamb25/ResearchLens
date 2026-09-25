
"""
ResearchLens: evidence-grounded thematic comparisons.

Compares findings from different papers discussing a shared
research theme. The comparison is distinct from claim-level
support, qualification and contradiction analysis.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any

from google.genai import types

from backend.generation.llm_service import (
    client,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

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
    "PROBLEM_AND_PROPOSED_SOLUTION":
        "Problem and proposed solution",
    "RELATED_FINDINGS":
        "Related findings; agreement not established",
    "INSUFFICIENT_EVIDENCE":
        "Insufficient evidence for comparison",
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

    for item in theme.get("evidence", []):
        if item.paper and item.claim:
            by_paper[item.paper].append(
                _finding(item)
            )

    return dict(sorted(by_paper.items()))


def _fallback(
    theme: dict[str, Any],
    by_paper: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    papers = sorted(by_paper)

    return {
        "theme_id": theme["theme_id"],
        "theme": theme["theme"],
        "relationship": "RELATED_FINDINGS",
        "relationship_label": (
            "Related findings; agreement not established"
        ),
        "explanation": (
            f"The papers {', '.join(papers)} address "
            f"the shared topic '{theme['theme']}'. "
            "Their findings may involve different "
            "methods, conditions or research objectives. "
            "The available evidence does not establish "
            "method-specific agreement or contradiction."
        ),
        "shared_concern": None,
        "important_differences": [],
        "limitations": [
            "Thematic overlap alone does not establish "
            "agreement or independent corroboration."
        ],
        "source_count": len(by_paper),
        "papers": papers,
        "source_summaries": [
            {
                "paper": paper,
                "findings": findings,
            }
            for paper, findings in by_paper.items()
        ],
        "findings": [
            finding
            for findings in by_paper.values()
            for finding in findings
        ],
        "agreement_established": False,
        "conflict_established": False,
        "analysis_method": (
            "Conservative thematic comparison"
        ),
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
    prompt = """
You are ResearchLens's evidence-grounded thematic
comparison analyst.

Compare the supplied findings ONLY. They were extracted
from research-paper passages.

Your task is to explain how the findings relate within
the given theme, not to invent agreement or contradiction.

Allowed relationship categories:

COMMON_CONCERN:
Different papers explicitly identify the same broad
research concern, potentially for different methods.
This does NOT establish agreement on a method-specific claim.

COMPLEMENTARY_FINDINGS:
The papers provide distinct but compatible findings
that illuminate different aspects of the same topic.

PROBLEM_AND_PROPOSED_SOLUTION:
One paper identifies a problem while another proposes
an approach intended to address that type of problem.
A proposed solution is NOT proof of successful resolution.

RELATED_FINDINGS:
The findings share a theme, but their precise relationship
is not established by the supplied evidence.

INSUFFICIENT_EVIDENCE:
The extracted findings do not permit a meaningful
comparison beyond superficial thematic overlap.

Rules:
1. Use only the supplied findings.
2. Do not invent experiments, performance values,
   datasets, methods, results or citations.
3. Distinguish observed limitations, literature-survey
   statements and proposed solutions.
4. Do not imply that a proposed framework has been
   experimentally validated unless the supplied finding
   explicitly reports validation.
5. Different methods discussing the same problem are
   not necessarily independent corroboration.
6. Do not declare direct agreement or contradiction.
7. Reference only source IDs present in the input.
8. Keep the explanation concise and specific.
9. If evidence is ambiguous, choose RELATED_FINDINGS
   or INSUFFICIENT_EVIDENCE.

Return one JSON object with EXACTLY these fields:
{
  "relationship": "one allowed category",
  "explanation": "two or three grounded sentences",
  "shared_concern": "brief description or null",
  "important_differences": [
    "one grounded difference"
  ],
  "limitations": [
    "one important qualification"
  ],
  "source_ids": ["S1", "S2"]
}

The source_ids must identify the findings used to
support the explanation and must include findings
from at least two different papers.

INPUT:
""" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
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
            "Theme comparison must cite at least "
            "two different papers."
        )

    differences = analysis.get(
        "important_differences"
    )

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

    shared_concern = analysis.get(
        "shared_concern"
    )

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
        "shared_concern": shared_concern,
        "important_differences": differences,
        "limitations": limitations,
        "source_ids": list(
            dict.fromkeys(source_ids)
        ),
    }


def compare_cross_paper_themes(
    themes: list[dict[str, Any]],
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """
    Compare each theme represented by at least two papers.

    Invalid model responses or API failures produce a
    conservative fallback. They do not fail the
    existing claim-level audit.
    """
    comparisons = []

    for theme in themes:
        if not theme.get("cross_paper"):
            continue

        by_paper = _source_groups(theme)

        if len(by_paper) < 2:
            continue

        comparison = _fallback(
            theme,
            by_paper,
        )

        if use_llm:
            payload = _model_payload(
                theme,
                by_paper,
            )

            for attempt in range(2):
                try:
                    analysis = _analyze_with_gemini(
                        payload
                    )

                    validated = _validate_analysis(
                        analysis,
                        payload,
                    )

                    comparison.update(validated)

                    comparison["analysis_method"] = (
                        "Gemini-assisted thematic "
                        "comparison"
                    )

                    break

                except Exception as error:
                    logger.warning(
                        "Theme comparison failed for "
                        "%s (attempt %s): %s",
                        theme["theme_id"],
                        attempt + 1,
                        error,
                    )

                    if attempt == 0:
                        time.sleep(1)

        comparisons.append(comparison)

    return comparisons