
import json
from types import SimpleNamespace

import backend.analysis.evidence_extractor as evidence_extractor


MOCK_RESPONSE = {
    "results": [
        {
            "evidence_index": 1,
            "claim": (
                "The proposed deep neural network achieved "
                "an accuracy of 99.2% on the NSL-KDD dataset."
            ),
            "dataset": "NSL-KDD",
            "method": "Deep neural network",
            "metric": "Accuracy",
            "conditions": (
                "The model was evaluated using accuracy "
                "as the primary performance metric."
            ),
            "evidence_scope": "OWN_WORK",
            "contribution_type": "OWN_RESULT",
            "attribution_reason": (
                "The passage describes the proposed model's "
                "own evaluation result."
            ),
            "related_work_claim": None,
        }
    ]
}


def test_evidence_extraction(monkeypatch):
    """Verify extraction without making a Gemini API call."""

    def mock_generate_content(*args, **kwargs):
        return SimpleNamespace(
            text=json.dumps(MOCK_RESPONSE)
        )

    monkeypatch.setattr(
        evidence_extractor,
        "_get_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                generate_content=mock_generate_content
            )
        ),
    )

    passage = """
    The proposed deep neural network achieved an accuracy
    of 99.2% on the NSL-KDD dataset. The model was evaluated
    using accuracy as the primary performance metric.
    """

    result = evidence_extractor.extract_evidence_context(
        passage
    )

    assert result["claim"] == MOCK_RESPONSE["results"][0]["claim"]
    assert result["dataset"] == "NSL-KDD"
    assert result["method"] == "Deep neural network"
    assert result["metric"] == "Accuracy"
    assert result["conditions"]
    assert result["evidence_scope"] == "OWN_WORK"
    assert result["contribution_type"] == "OWN_RESULT"