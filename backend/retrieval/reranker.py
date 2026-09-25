
"""
Cross-encoder reranking for ResearchLens.

The model is initialized lazily so importing this module
does not download or load model weights during test collection.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load the cross-encoder once, on its first use."""
    return CrossEncoder(MODEL_NAME)


def rerank(query, evidence, top_k=5):
    """
    Score candidate passages against the research question.

    Returns the highest-scoring passages with an additional
    reranker_score field. Does not modify input dictionaries.
    """

    if not evidence or top_k <= 0:
        return []

    pairs = [
        [query, item["text"]]
        for item in evidence
    ]

    model = get_reranker_model()
    scores = model.predict(pairs)

    ranked = []

    for item, score in zip(evidence, scores):
        ranked.append({
            **item,
            "reranker_score": float(score),
        })

    ranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return ranked[:top_k]