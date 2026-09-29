
"""Normalize ChromaDB results and preserve retrieval provenance."""


def format_retrieval_results(results):
    """
    Convert retrieval results into normalized evidence candidates.

    Supports both legacy Chroma results and the extended
    paper-wise retrieval output containing intent labels.
    """
    documents = (
        results.get("documents") or [[]]
    )[0]

    metadatas = (
        results.get("metadatas") or [[]]
    )[0]

    distances = (
        results.get("distances") or [[]]
    )[0]

    intents = (
        results.get("intents") or [[]]
    )[0]

    formatted = []

    for index, text in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        ) or {}

        if not text or not metadata.get("document"):
            continue

        raw_intents = (
            intents[index]
            if index < len(intents)
            else []
        )

        if not isinstance(raw_intents, list):
            raw_intents = []

        formatted.append({
            "text": text,
            "document": metadata.get("document"),
            "document_id": metadata.get("document_id"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "retrieval_distance": (
                distances[index]
                if index < len(distances)
                else None
            ),
            "retrieval_intents": list(
                dict.fromkeys(raw_intents)
            ),
        })

    return formatted