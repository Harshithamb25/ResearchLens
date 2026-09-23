import json
import time

from generation.llm_service import client, MODEL_NAME


def extract_evidence_context(evidence_text):
    """
    Extract structured research information from a single
    evidence passage.

    The model must only extract information explicitly stated
    in the supplied evidence.
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

            required_fields = [
                "claim",
                "dataset",
                "method",
                "metric",
                "conditions"
            ]

            for field in required_fields:
                if field not in result:
                    result[field] = None

            return result

        except Exception as error:

            if attempt == max_attempts - 1:
                raise error

            wait_time = 2 ** attempt

            print(
                f"Evidence extraction failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)