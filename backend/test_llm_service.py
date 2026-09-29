
from types import SimpleNamespace

import backend.generation.llm_service as llm_service


def test_generate_answer_uses_grounded_evidence(monkeypatch):
    """Verify that the answer uses the supplied source passage."""

    question = "What datasets are discussed in the research paper?"

    evidence = [
        {
            "document": "sample.pdf",
            "page": 9,
            "chunk_id": 8,
            "text": (
                "The datasets used and generated for research "
                "in malware classification and network intrusion "
                "detection are detailed in this section."
            ),
        }
    ]

    captured = {}

    def mock_generate_content(*, model, contents, **kwargs):
        captured["model"] = model
        captured["prompt"] = contents

        return SimpleNamespace(
            text=(
                "The paper discusses malware classification "
                "and network intrusion detection datasets "
                "[sample.pdf, Page 9]."
            )
        )

    # Mock the lazy Gemini client without making a real API call.
    monkeypatch.setattr(
        llm_service,
        "_get_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                generate_content=mock_generate_content
            )
        ),
    )

    answer = llm_service.generate_answer(
        question,
        evidence,
    )

    expected_answer = (
        "The paper discusses malware classification "
        "and network intrusion detection datasets "
        "[sample.pdf, Page 9]."
    )

    assert answer == expected_answer

    # Verify that the question and original passage reach the model.
    assert captured["model"]
    assert question in captured["prompt"]
    assert "sample.pdf" in captured["prompt"]
    assert '"page": 9' in captured["prompt"]
    assert "malware classification" in captured["prompt"]

    # Verify that the answer retains its source citation.
    assert "[sample.pdf, Page 9]" in answer