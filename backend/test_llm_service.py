from backend.generation.llm_service import generate_answer


def test_generate_answer_uses_grounded_evidence(monkeypatch):
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

    captured_prompt = {}

    class MockResponse:
        text = "The paper discusses malware and network intrusion detection datasets."

    def mock_generate_content(*, model, contents):
        captured_prompt["model"] = model
        captured_prompt["contents"] = contents
        return MockResponse()

    monkeypatch.setattr(
        "backend.generation.llm_service.client.models.generate_content",
        mock_generate_content
    )

    answer = generate_answer(
        question,
        evidence
    )

    assert answer == (
        "The paper discusses malware and network intrusion detection datasets."
    )

    assert captured_prompt["model"]

    assert question in captured_prompt["contents"]

    assert "sample.pdf" in captured_prompt["contents"]

    assert "Page: 9" in captured_prompt["contents"]

    assert "malware classification" in captured_prompt["contents"]