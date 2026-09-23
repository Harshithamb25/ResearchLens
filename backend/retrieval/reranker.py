from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = CrossEncoder(MODEL_NAME)


def rerank(query, evidence, top_k=5):
    pairs = [
        [query, item["text"]]
        for item in evidence
    ]

    scores = model.predict(pairs)

    ranked = []

    for item, score in zip(evidence, scores):
        ranked.append({
            **item,
            "reranker_score": float(score)
        })

    ranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return ranked[:top_k]