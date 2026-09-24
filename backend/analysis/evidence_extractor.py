"""
Evidence context extraction for ResearchLens.

Uses the configured LLM to extract structured research
information from retrieved evidence passages.

The module supports both:
- single-passage extraction
- batch extraction for multiple passages

Batch extraction is preferred by the main pipeline because
it reduces the number of LLM requests.
"""

import json
import time

from backend.generation.llm_service import (
    client,
    MODEL_NAME
)


REQUIRED_FIELDS = [
    "claim",
    "dataset",
    "method",
    "metric",
    "conditions"
]


def _normalize_result(result):
    """
    Ensure that an extracted evidence result contains
    all required fields.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Evidence extraction result must be a JSON object."
        )

    normalized = {}

    for field in REQUIRED_FIELDS:
        normalized[field] = result.get(field)

    return normalized


def extract_evidence_context(evidence_text):
    """
    Extract structured research information from a single
    evidence passage.

    This function is kept for compatibility with existing
    tests and any future single-passage use cases.

    Args:
        evidence_text: Retrieved research passage.

    Returns:
        Dictionary containing:
        - claim
        - dataset
        - method
        - metric
        - conditions
    """

    prompt = f"""
You are an evidence extraction component of ResearchLens.

Analyze ONLY the research passage provided below.

Extract:

1. claim
2. dataset
3. method
4. metric
5. conditions

Rules:

- Use ONLY information explicitly stated in the passage.
- Do NOT use outside knowledge.
- Do NOT infer missing information.
- If a field is not explicitly stated, return null.
- Keep the extracted information concise.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not add explanations.

Required JSON format:

{{
    "claim": "...",
    "dataset": "...",
    "method": "...",
    "metric": "...",
    "conditions": "..."
}}

RESEARCH PASSAGE:
{evidence_text}
"""

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            result = json.loads(response.text)

            return _normalize_result(result)

        except Exception as error:

            if attempt == max_attempts - 1:
                raise error

            wait_time = 2 ** attempt

            print(
                f"Evidence extraction failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


def extract_evidence_context_batch(evidence_texts):
    """
    Extract structured research information from multiple
    evidence passages using a single LLM request.

    Args:
        evidence_texts:
            List of retrieved research passages.

    Returns:
        List of dictionaries containing:
        - claim
        - dataset
        - method
        - metric
        - conditions

    The returned list preserves the same order as the
    input evidence_texts.
    """

    if not evidence_texts:
        return []

    evidence_sections = []

    for index, evidence_text in enumerate(
        evidence_texts,
        start=1
    ):
        evidence_sections.append(
            f"""
EVIDENCE {index}

RESEARCH PASSAGE:
{evidence_text}
"""
        )

    combined_evidence = "\n-----------------------------\n".join(
        evidence_sections
    )

    prompt = f"""
You are an evidence extraction component of ResearchLens.

You will receive multiple research evidence passages.

Analyze EACH passage independently.

For every passage, extract:

1. claim
2. dataset
3. method
4. metric
5. conditions

Rules:

- Use ONLY information explicitly stated in each passage.
- Do NOT use outside knowledge.
- Do NOT infer missing information.
- If a field is not explicitly stated, return null.
- Keep extracted information concise.
- Do not mix information between different evidence passages.
- Preserve the original evidence order.
- Return exactly one result for each evidence passage.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not add explanations.

Required JSON format:

{{
    "results": [
        {{
            "evidence_index": 1,
            "claim": "...",
            "dataset": "...",
            "method": "...",
            "metric": "...",
            "conditions": "..."
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
                contents=prompt
            )

            result = json.loads(response.text)

            if not isinstance(result, dict):
                raise ValueError(
                    "Batch extraction result must be a JSON object."
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
                    f"{len(evidence_texts)} evidence passages."
                )

            normalized_results = []

            for index, item in enumerate(results):

                if not isinstance(item, dict):
                    raise ValueError(
                        f"Invalid extraction result at index {index}."
                    )

                expected_index = index + 1
                actual_index = item.get(
                    "evidence_index",
                    expected_index
                )

                if actual_index != expected_index:
                    raise ValueError(
                        "Evidence ordering mismatch in "
                        "batch extraction response."
                    )

                normalized_results.append(
                    _normalize_result(item)
                )

            return normalized_results

        except Exception as error:

            if attempt == max_attempts - 1:
                raise error

            wait_time = 2 ** attempt

            print(
                f"Batch evidence extraction failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)