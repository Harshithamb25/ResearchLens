def format_retrieval_results(results):
    """
    Convert ChromaDB query results into a list of
    normalized evidence records.
    """

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted = []

    for i, text in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}

        formatted.append({
            "text": text,
            "document": metadata.get("document"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "retrieval_distance": (
                distances[i] if i < len(distances) else None
            )
        })

    return formatted