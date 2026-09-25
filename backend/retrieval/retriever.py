
"""
Semantic and source-aware retrieval for ResearchLens.

Supports global search, per-paper search, deterministic query
expansion, and deduplication.

Retrieval produces candidates. It does not establish that
retrieved passages support a research claim.
"""

from backend.retrieval.embeddings import generate_embedding
from backend.retrieval.vector_store import collection


def _embedding_list(text):
    embedding = generate_embedding(text)
    return (
        embedding.tolist()
        if hasattr(embedding, "tolist")
        else list(embedding)
    )


def search(query, top_k=5):
    """Global semantic search across all indexed papers."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not query or not query.strip():
        raise ValueError("query must not be empty")

    count = collection.count()

    if count == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    return collection.query(
        query_embeddings=[_embedding_list(query)],
        n_results=min(top_k, count),
    )


def _indexed_paper_counts():
    """Return the number of indexed chunks for each paper."""

    records = collection.get(include=["metadatas"])
    counts = {}

    for metadata in records["metadatas"]:
        if not metadata:
            continue

        paper = metadata.get("document")

        if paper:
            counts[paper] = counts.get(paper, 0) + 1

    return counts


def _indexed_papers():
    return sorted(_indexed_paper_counts())


def _expand_query(question):
    """
    Generate a small, deterministic set of related queries.

    These improve candidate coverage; they do not prove
    relevance or establish a cross-paper relationship.
    """

    question = question.strip()
    queries = [question]
    lower_question = question.lower()

    intrusion_related = any(
        term in lower_question
        for term in (
            "intrusion",
            "cybersecurity",
            "cyber security",
            "network security",
        )
    ) or "ids" in lower_question.split()

    if intrusion_related:
        queries.extend([
            "AI intrusion detection challenges and limitations",
            "Intrusion detection dataset limitations and benchmarking",
            "Machine learning intrusion detection generalization "
            "and dataset quality",
            "Deep learning intrusion detection scalability "
            "and performance",
        ])

    return list(dict.fromkeys(queries))


def search_across_papers(
    query,
    per_paper_k=5,
    expand_queries=True,
):
    """
    Retrieve and deduplicate candidates independently per paper.

    Compatible with the existing result_formatter.py.
    """

    if per_paper_k < 1:
        raise ValueError("per_paper_k must be at least 1")

    if not query or not query.strip():
        raise ValueError("query must not be empty")

    queries = (
        _expand_query(query)
        if expand_queries
        else [query.strip()]
    )

    embeddings = {
        text: _embedding_list(text)
        for text in queries
    }

    combined = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    paper_counts = _indexed_paper_counts()

    for paper in sorted(paper_counts):
        candidates = {}
        search_limit = min(per_paper_k, paper_counts[paper])

        for search_query in queries:
            results = collection.query(
                query_embeddings=[embeddings[search_query]],
                n_results=search_limit,
                where={"document": paper},
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for passage, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            ):
                key = (
                    metadata.get("document"),
                    metadata.get("chunk_id"),
                )

                if (
                    key not in candidates
                    or distance < candidates[key]["distance"]
                ):
                    candidates[key] = {
                        "text": passage,
                        "metadata": metadata,
                        "distance": distance,
                    }

        # Deterministic order makes debugging and tests easier.
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                item["distance"],
                item["metadata"].get("chunk_id", 0),
            ),
        )

        for candidate in ordered:
            combined["documents"][0].append(
                candidate["text"]
            )
            combined["metadatas"][0].append(
                candidate["metadata"]
            )
            combined["distances"][0].append(
                candidate["distance"]
            )

    return combined