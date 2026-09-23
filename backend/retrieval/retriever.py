# This converts the user's question into the same type of vector as our paper chunks and asks ChromaDB to find the closest semantic matches.

from retrieval.embeddings import generate_embedding
from retrieval.vector_store import collection


def search(query, top_k=5):
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results