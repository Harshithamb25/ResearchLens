from retrieval.result_formatter import format_retrieval_results


def test_format_retrieval_results():

    mock_results = {
        "documents": [
            [
                "Evidence from paper A",
                "Evidence from paper B"
            ]
        ],
        "metadatas": [
            [
                {
                    "document": "paperA.pdf",
                    "page": 5,
                    "chunk_id": 12
                },
                {
                    "document": "paperB.pdf",
                    "page": 8,
                    "chunk_id": 21
                }
            ]
        ],
        "distances": [
            [0.25, 0.41]
        ]
    }

    formatted = format_retrieval_results(mock_results)

    assert len(formatted) == 2

    assert formatted[0]["document"] == "paperA.pdf"
    assert formatted[0]["page"] == 5
    assert formatted[0]["chunk_id"] == 12
    assert formatted[0]["text"] == "Evidence from paper A"

    assert formatted[1]["document"] == "paperB.pdf"