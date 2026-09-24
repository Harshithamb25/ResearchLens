
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
    Classify each evidence item using its unique position
    within the current claim group.
    """
    claim = claim_group["claim"]
    evidence_items = claim_group["evidence"]

    if not evidence_items:
        return []

    sections = []

    for index, evidence in enumerate(evidence_items, start=1):
        sections.append(
            f"""
Evidence ID: {index}
Paper: {evidence.paper}
Page: {evidence.page}
Chunk ID: {evidence.chunk_id}

Claim: {evidence.claim}
Dataset: {evidence.dataset}
Method: {evidence.method}
Metric: {evidence.metric}
Conditions: {evidence.conditions}

Evidence text:
{evidence.evidence_text}
"""
        )

    combined_evidence = "\n----------------\n".join(sections)

    prompt = f"""
You are the cross-paper evidence analysis component
of ResearchLens.

CLAIM:
{claim}

EVIDENCE:
{combined_evidence}

For EACH evidence item, classify its relationship
to the claim.

Allowed relationships:
SUPPORT
QUALIFY
POTENTIAL_CONFLICT
INSUFFICIENT_EVIDENCE

Rules:
1. Use only the supplied evidence.
2. Consider datasets, methods, metrics and conditions.
3. Different numerical results are not automatically
   contradictions.
4. Use POTENTIAL_CONFLICT only for meaningful
   disagreements not explained by the stated context.
5. Use QUALIFY for relevant limitations or exceptions.
6. Use INSUFFICIENT_EVIDENCE when the passage cannot
   establish a relationship.
7. Preserve each Evidence ID exactly.
8. Return one result per Evidence ID.
9. Do not invent IDs or omit evidence.
10. Return valid JSON only.

Required format:
{{
    "relationships": [
        {{
            "evidence_id": 1,
            "relationship": "SUPPORT",
            "explanation": "Concise evidence-based reason"
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
                contents=prompt
            )

            result = json.loads(response.text)

            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object.")

            relationships = result.get("relationships")

            if not isinstance(relationships, list):
                raise ValueError("Missing relationships list.")

            if len(relationships) != len(evidence_items):
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
                    or evidence_id > len(evidence_items)
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
                    raise ValueError("Missing relationship explanation.")

            return sorted(
                relationships,
                key=lambda item: item["evidence_id"]
            )

        except Exception:
            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt
            print(
                f"Relationship analysis retrying "
                f"in {wait_time} seconds..."
            )
            time.sleep(wait_time)


def build_evidence_relationships(claim_group):
    """
    Link each classification to its exact original
    Evidence object using the validated Evidence ID.
    """
    analyzed = analyze_evidence_relationships(claim_group)

    evidence_items = claim_group["evidence"]
    claim = claim_group["claim"]

    relationships = []

    for item in analyzed:
        evidence = evidence_items[item["evidence_id"] - 1]

        relationships.append(
            EvidenceRelationship(
                claim=claim,
                paper=evidence.paper,
                page=evidence.page,
                relationship=item["relationship"],
                explanation=item["explanation"],
                evidence=evidence
            )
        )

    return relationships