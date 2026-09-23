from retrieval.embeddings import generate_embedding
import numpy as np


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    denominator = (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


def calculate_claim_similarity(claim_a, claim_b):
    """
    Calculate semantic similarity between two claims
    using the project's existing embedding model.
    """

    embedding_a = generate_embedding(claim_a)
    embedding_b = generate_embedding(claim_b)

    return cosine_similarity(
        embedding_a,
        embedding_b
    )