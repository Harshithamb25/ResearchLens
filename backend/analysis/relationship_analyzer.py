import json
import time

from backend.generation.llm_service import client, MODEL_NAME
from backend.analysis.cross_paper_analysis import EvidenceRelationship


ALLOWED_RELATIONSHIPS = {
    "SUPPORT",
    "QUALIFY",
    "POTENTIAL_CONFLICT",
    "INSUFFICIENT_EVIDENCE",
}


def analyze_evidence_relationships(claim_group):
    """
    Analyze the relationship between evidence items
    belonging to the same semantic claim group.

    The analysis considers research context such as:
    dataset, method, metric, and conditions.

    Returns:
        List of dictionaries containing the relationship
        assigned to each evidence item.
    """

    claim = claim_group["claim"]
    evidence_items = claim_group["evidence"]

    evidence_text = []

    for index, evidence in enumerate(evidence_items, start=1):

        evidence_text.append(
            f"""
Evidence {index}

Paper: {evidence.paper}
Page: {evidence.page}

Claim:
{evidence.claim}

Dataset:
{evidence.dataset}

Method:
{evidence.method}

Metric:
{evidence.metric}

Conditions:
{evidence.conditions}

Evidence text:
{evidence.evidence_text}
"""
        )

    combined_evidence = (
        "\n-----------------------------\n"
        .join(evidence_text)
    )

    prompt = f"""
You are the cross-paper evidence analysis component
of ResearchLens.

Analyze the research evidence for the following semantic claim:

CLAIM:
{claim}

EVIDENCE FROM MULTIPLE PAPERS:
{combined_evidence}

For EACH evidence item, determine its relationship to the
overall claim.

Allowed relationship values:

SUPPORT
QUALIFY
POTENTIAL_CONFLICT
INSUFFICIENT_EVIDENCE

Important rules:

1. Use ONLY the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent missing information.
4. Consider dataset, method, metric, and conditions.
5. Different numerical results are NOT automatically a conflict.
6. If studies use substantially different datasets, methods,
   metrics, or conditions, do not automatically classify them
   as conflicting.
7. Use POTENTIAL_CONFLICT only when the supplied evidence
   indicates a meaningful disagreement that cannot be explained
   by the stated context.
8. Use QUALIFY when the evidence supports the general claim
   but limits, conditions, or exceptions are reported.
9. Use INSUFFICIENT_EVIDENCE when the supplied passage does
   not contain enough information to determine the relationship.
10. Give a concise evidence-based explanation.
11. Return ONLY valid JSON.
12. Do not include markdown.

Required JSON format:

{{
    "relationships": [
        {{
            "paper": "...",
            "page": 1,
            "relationship": "SUPPORT",
            "explanation": "..."
        }}
    ]
}}
"""

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            result = json.loads(response.text)

            if "relationships" not in result:
                raise ValueError(
                    "Missing relationships field."
                )

            relationships = []

            for item in result["relationships"]:

                relationship = item.get("relationship")

                if relationship not in ALLOWED_RELATIONSHIPS:
                    raise ValueError(
                        f"Invalid relationship: {relationship}"
                    )

                relationships.append(item)

            return relationships

        except Exception as error:

            if attempt == max_attempts - 1:
                raise error

            wait_time = 2 ** attempt

            print(
                f"Relationship analysis failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


def build_evidence_relationships(claim_group):
    """
    Convert analyzed relationship results into
    EvidenceRelationship objects linked to the
    original Evidence objects.

    This preserves the chain:

        Claim
          ↓
        Relationship
          ↓
        Evidence
          ↓
        Paper + Page
    """

    analyzed = analyze_evidence_relationships(
        claim_group
    )

    evidence_items = claim_group["evidence"]
    claim = claim_group["claim"]

    relationships = []

    for item in analyzed:

        matching_evidence = None

        for evidence in evidence_items:

            if (
                evidence.paper == item["paper"]
                and evidence.page == item["page"]
            ):
                matching_evidence = evidence
                break

        if matching_evidence is None:
            continue

        relationships.append(
            EvidenceRelationship(
                claim=claim,
                paper=item["paper"],
                page=item["page"],
                relationship=item["relationship"],
                explanation=item["explanation"],
                evidence=matching_evidence
            )
        )

    return relationships

