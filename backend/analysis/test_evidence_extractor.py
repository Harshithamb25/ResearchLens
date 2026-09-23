from analysis.evidence_extractor import extract_evidence_context


def test_evidence_extraction():

    text = """
    The proposed deep neural network achieved an accuracy
    of 99.2% on the NSL-KDD dataset. The model was evaluated
    using accuracy as the primary performance metric.
    """

    result = extract_evidence_context(text)

    assert isinstance(result, dict)

    assert "claim" in result
    assert "dataset" in result
    assert "method" in result
    assert "metric" in result
    assert "conditions" in result

    assert result["dataset"] is not None
    assert result["metric"] is not None