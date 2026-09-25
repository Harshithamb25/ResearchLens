
"""
Source-grounded relationship analysis for ResearchLens.

Classifies evidence against a claim and validates that
every classification maps to its original evidence item.
"""

import json
import time

from google.genai import types

from backend.generation.llm_service import client, MODEL_NAME
from backend.analysis.cross_paper_analysis import EvidenceRelationship


ALLOWED_RELATIONSHIPS = {
    "SUPPORT",
    "QUALIFY",
    "POTENTIAL_CONFLICT",
    "INSUFFICIENT_EVIDENCE",
}


def _parse_response(response):
    """Parse JSON without silently accepting empty output."""

    raw_text = response.text

    if not raw_text or not raw_text.strip():
        raise ValueError(
            "Gemini returned an empty relationship response."
        )

    raw_text = raw_text.strip()

    # Defensive handling if a model returns Markdown fences.
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_text = "\n".join(lines).strip()

    return json.loads(raw_text)


def _validate_relationships(result, evidence_count):
    """Reject missing, invented or duplicate evidence IDs."""

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    relationships = result.get("relationships")

    if not isinstance(relationships, list):
        raise ValueError("Missing relationships list.")

    if len(relationships) != evidence_count:
        raise ValueError(
            "Relationship count does not match evidence count."
        )

    seen_ids = set()

    for item in relationships:
        if not isinstance(item, dict):
            raise ValueError("Invalid relationship item.")

        evidence_id = item.get("evidence_id")

        if (
            type(evidence_id) is not int
            or evidence_id < 1
            or evidence_id > evidence_count
            or evidence_id in seen_ids
        ):
            raise ValueError(
                f"Invalid or duplicate evidence ID: {evidence_id}"
            )

        seen_ids.add(evidence_id)

        if item.get("relationship") not in ALLOWED_RELATIONSHIPS:
            raise ValueError("Invalid relationship category.")

        explanation = item.get("explanation")

        if (
            not isinstance(explanation, str)
            or not explanation.strip()
        ):
            raise ValueError(
                "Missing relationship explanation."
            )

        item["explanation"] = explanation.strip()

    return sorted(
        relationships,
        key=lambda item: item["evidence_id"],
    )


def analyze_evidence_relationships(claim_group):
    """
    Classify each evidence item against the group's claim.
    """

    claim = claim_group["claim"]
    evidence_items = claim_group["evidence"]

    if not evidence_items:
        return []

    sections = []

    for index, evidence in enumerate(
        evidence_items,
        start=1,
    ):
        sections.append(
            f"""
Evidence ID: {index}
Paper: {evidence.paper}
Page: {evidence.page}
Chunk ID: {evidence.chunk_id}

Extracted claim: {evidence.claim}
Dataset: {evidence.dataset}
Method: {evidence.method}
Metric: {evidence.metric}
Conditions: {evidence.conditions}

Original source passage:
{evidence.evidence_text}
"""
        )

    combined_evidence = "\n----------------\n".join(
        sections
    )

    prompt = f"""
You are the evidence relationship analyzer of ResearchLens.

TARGET CLAIM:
{claim}

SOURCE EVIDENCE:
{combined_evidence}

Classify EACH evidence item against the target claim.

Allowed relationships:

SUPPORT:
The source passage directly supports the target claim.

QUALIFY:
The passage supports or addresses the claim but adds
an important limitation, condition or exception.

POTENTIAL_CONFLICT:
The passage presents a meaningful disagreement with
the target claim that is not explained by differences
in the stated conditions.

INSUFFICIENT_EVIDENCE:
The passage does not establish a reliable relationship
to the target claim.

Rules:
- Use only the supplied source passages.
- Do not invent research findings or missing context.
- A shared topic alone does not establish agreement.
- Different numerical results are not automatically
  contradictions.
- A passage containing the same extracted claim can
  support that claim if the original text substantiates it.
- Preserve every Evidence ID exactly.
- Return one classification per Evidence ID.
- Give a concise, source-grounded explanation.
- Return only a JSON object.

Required JSON structure:

{{
    "relationships": [
        {{
            "evidence_id": 1,
            "relationship": "SUPPORT",
            "explanation": "Reason grounded in the passage"
        }}
    ]
}}

Expected evidence count: {len(evidence_items)}
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )

            result = _parse_response(response)

            return _validate_relationships(
                result,
                len(evidence_items),
            )

        except Exception as error:
            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt

            print(
                "Relationship analysis failed "
                f"({type(error).__name__}). "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


def build_evidence_relationships(claim_group):
    """
    Link validated classifications to their exact
    original Evidence objects.
    """

    analyzed = analyze_evidence_relationships(
        claim_group
    )

    evidence_items = claim_group["evidence"]
    claim = claim_group["claim"]

    relationships = []

    for item in analyzed:
        evidence = evidence_items[
            item["evidence_id"] - 1
        ]

        relationships.append(
            EvidenceRelationship(
                claim=claim,
                paper=evidence.paper,
                page=evidence.page,
                relationship=item["relationship"],
                explanation=item["explanation"],
                evidence=evidence,
            )
        )

    return relationships