import json

import backend.analysis.evidence_extractor as evidence_extractor


MOCK_RESPONSE = {
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
    )
}


class MockResponse:
    """
    Simulates a Gemini API response.
    """

    text = json.dumps(MOCK_RESPONSE)


def mock_generate_content(*args, **kwargs):
    """
    Simulates Gemini's generate_content method.
    """

    return MockResponse()


def test_evidence_extraction(monkeypatch):

    monkeypatch.setattr(
        evidence_extractor.client.models,
        "generate_content",
        mock_generate_content
    )

    text = """
    The proposed deep neural network achieved an accuracy
    of 99.2% on the NSL-KDD dataset. The model was evaluated
    using accuracy as the primary performance metric.
    """

    result = evidence_extractor.extract_evidence_context(
        text
    )

    assert result

    assert result["claim"]

    assert result["dataset"] == "NSL-KDD"

    assert result["method"] == "Deep neural network"

    assert result["metric"] == "Accuracy"

    assert result["conditions"]


