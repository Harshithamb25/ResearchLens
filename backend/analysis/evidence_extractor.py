
"""
Evidence context extraction for ResearchLens.

Extracts source-grounded research claims and supporting
context from individual or batched research passages.

A claim may be qualitative or quantitative. Missing
information must never be invented.
"""

import json
import time

from backend.generation.llm_service import (
    client,
    MODEL_NAME,
)


REQUIRED_FIELDS = [
    "claim",
    "dataset",
    "method",
    "metric",
    "conditions",
]


EXTRACTION_RULES = """
Rules:

- Use ONLY information explicitly stated in the passage.
- Do NOT use outside knowledge or invent findings.
- A claim is an explicit research assertion, observation,
  conclusion, limitation, challenge, or reported result.
- Claims may be qualitative. They do not require a
  numerical metric, named dataset, or experimental result.
- Extract the most relevant substantive claim from
  the passage.
- Preserve the original meaning of the source.
- Do not turn a proposed method into a proven result.
- If the passage contains only references, author details,
  copyright information, or unrelated text, return null
  for the claim.
- If no substantive claim is explicitly stated, return null.
- Extract dataset, method, metric, and conditions only
  when explicitly stated. Otherwise return null.
- Never invent missing information.
- Keep each extracted field concise.
- Return ONLY valid JSON.
- Do not include markdown or explanations.
"""


def _normalize_result(result):
    """
    Ensure every extraction result contains the required
    fields. Missing or empty fields are represented by None.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Evidence extraction result must be a JSON object."
        )

    normalized = {}

    for field in REQUIRED_FIELDS:
        value = result.get(field)

        if isinstance(value, str):
            value = value.strip()

            if not value or value.lower() in {
                "null",
                "none",
                "n/a",
                "not specified",
            }:
                value = None

        normalized[field] = value

    return normalized


def _parse_json_response(response):
    """
    Parse the model's JSON response.

    Handles an occasional Markdown code fence without
    changing or inventing any extracted information.
    """

    response_text = response.text

    if not response_text:
        raise ValueError(
            "Evidence extraction returned an empty response."
        )

    response_text = response_text.strip()

    if response_text.startswith("```"):
        lines = response_text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

    return json.loads(response_text)


def extract_evidence_context(evidence_text):
    """
    Extract structured research information from one passage.

    Returns:
        {
            "claim": str | None,
            "dataset": str | None,
            "method": str | None,
            "metric": str | None,
            "conditions": str | None
        }
    """

    prompt = f"""
You are the evidence extraction component of ResearchLens.

Analyze ONLY the research passage provided below.

Extract these five fields:

1. claim
2. dataset
3. method
4. metric
5. conditions

{EXTRACTION_RULES}

Required JSON format:

{{
    "claim": "An explicit research claim or null",
    "dataset": null,
    "method": null,
    "metric": null,
    "conditions": null
}}

RESEARCH PASSAGE:

{evidence_text}
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            result = _parse_json_response(response)

            return _normalize_result(result)

        except Exception:
            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt

            print(
                "Evidence extraction failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


def extract_evidence_context_batch(evidence_texts):
    """
    Extract structured information from multiple passages
    in one Gemini request.

    Each passage is analyzed independently, and the
    returned results must match the original input order.
    """

    if not evidence_texts:
        return []

    evidence_sections = []

    for index, evidence_text in enumerate(
        evidence_texts,
        start=1,
    ):
        evidence_sections.append(
            f"""
EVIDENCE {index}

RESEARCH PASSAGE:

{evidence_text}
"""
        )

    combined_evidence = (
        "\n-----------------------------\n".join(
            evidence_sections
        )
    )

    prompt = f"""
You are the evidence extraction component of ResearchLens.

You will receive multiple research passages.

Analyze EACH passage independently.

For EVERY passage, extract:

1. claim
2. dataset
3. method
4. metric
5. conditions

{EXTRACTION_RULES}

Additional batch rules:

- Do not mix information between passages.
- Preserve the original passage order.
- Return exactly one result for each input passage.
- Use evidence_index values starting at 1.
- Do not omit a passage even if its claim is null.

Required JSON format:

{{
    "results": [
        {{
            "evidence_index": 1,
            "claim": "An explicit research claim or null",
            "dataset": null,
            "method": null,
            "metric": null,
            "conditions": null
        }}
    ]
}}

There are {len(evidence_texts)} evidence passages.

{combined_evidence}
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            result = _parse_json_response(response)

            if not isinstance(result, dict):
                raise ValueError(
                    "Batch extraction result must be "
                    "a JSON object."
                )

            results = result.get("results")

            if not isinstance(results, list):
                raise ValueError(
                    "Batch extraction response is missing "
                    "the 'results' list."
                )

            if len(results) != len(evidence_texts):
                raise ValueError(
                    "Batch extraction returned "
                    f"{len(results)} results for "
                    f"{len(evidence_texts)} passages."
                )

            normalized_results = []

            for index, item in enumerate(results):
                if not isinstance(item, dict):
                    raise ValueError(
                        "Invalid extraction result "
                        f"at index {index}."
                    )

                expected_index = index + 1

                actual_index = item.get(
                    "evidence_index",
                    expected_index,
                )

                if actual_index != expected_index:
                    raise ValueError(
                        "Evidence ordering mismatch "
                        "in batch extraction response."
                    )

                normalized_results.append(
                    _normalize_result(item)
                )

            return normalized_results

        except Exception:
            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt

            print(
                "Batch evidence extraction failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)