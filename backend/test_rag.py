import backend.retrieval.retriever as retriever
import backend.generation.llm_service as llm_service


def test_rag_retrieval_and_grounded_generation(monkeypatch):
    query = "What datasets are discussed in the research paper?"

    # Mock retrieval so the test does not depend on ChromaDB.
    def mock_search(query, top_k=2):
        return {
            "documents": [[
                "The paper discusses malware datasets and "
                "network intrusion detection datasets."
            ]],
            "metadatas": [[
                {
                    "document": "sample.pdf",
                    "page": 9,
                    "chunk_id": 8
                }
            ]],
            "distances": [[0.2]]
        }

    monkeypatch.setattr(
        retriever,
        "search",
        mock_search
    )

    evidence = retriever.search(
        query,
        top_k=2
    )

    documents = evidence["documents"][0]
    metadatas = evidence["metadatas"][0]

    assert documents
    assert metadatas

    # Convert retrieved results into the format expected
    # by the grounded generation service.
    answer_evidence = []

    for i in range(len(documents)):
        answer_evidence.append({
            "document": metadatas[i]["document"],
            "page": metadatas[i]["page"],
            "chunk_id": metadatas[i]["chunk_id"],
            "text": documents[i]
        })

    captured = {}

    def mock_generate_answer(question, evidence):
        captured["question"] = question
        captured["evidence"] = evidence

        return (
            "The paper discusses malware datasets and "
            "network intrusion detection datasets. "
            "[sample.pdf, Page 9]"
        )

    monkeypatch.setattr(
        llm_service,
        "generate_answer",
        mock_generate_answer
    )

    answer = llm_service.generate_answer(
        query,
        answer_evidence
    )

    assert answer

    assert captured["question"] == query

    assert len(captured["evidence"]) == 1

    assert captured["evidence"][0]["document"] == "sample.pdf"
    assert captured["evidence"][0]["page"] == 9
    assert captured["evidence"][0]["chunk_id"] == 8

    assert "datasets" in captured["evidence"][0]["text"]