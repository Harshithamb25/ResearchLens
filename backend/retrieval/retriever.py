"""
Semantic retrieval for ResearchLens.

Converts a research question into an embedding and
retrieves the most semantically relevant paper chunks
from ChromaDB.
"""

from backend.retrieval.embeddings import generate_embedding
from backend.retrieval.vector_store import collection


def search(query, top_k=5):
    """
    Search the ResearchLens vector store for
    semantically relevant research-paper chunks.

    Args:
        query: User's research question.
        top_k: Number of candidates to retrieve.

    Returns:
        ChromaDB query results.
    """

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results