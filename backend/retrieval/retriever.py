
"""Project-scoped, paper-wise retrieval for ResearchLens.

Combines semantic retrieval with section-aware candidate discovery.
Retrieval intents are search hints, not verified source attribution.
"""

import re
from collections import defaultdict

from backend.retrieval.embeddings import generate_embedding
from backend.retrieval.vector_store import collection


RESEARCH_ASPECTS = {
    "methodology": (
        "In this paper we propose and implement our system. "
        "Proposed architecture, hardware components, methodology, "
        "system design and implementation."
    ),
    "evaluation": (
        "Our experimental setup, prototype testing, field trial, "
        "evaluation procedure, test cases and measurements."
    ),
    "results": (
        "Our experimental results, results and discussion, "
        "observed measurements, performance, cost and findings."
    ),
    "limitations": (
        "Our system limitations, practical constraints, "
        "deployment challenges, conclusion and future work."
    ),
}

ASPECT_ORDER = (
    "question",
    "methodology",
    "evaluation",
    "results",
    "limitations",
)

# Only use these patterns to find candidates. The evidence
# extractor remains responsible for authorship verification.
SECTION_PATTERNS = {
    "methodology": (
        r"\bproposed (?:system|model|method|architecture)\b",
        r"\bsystem (?:design|architecture|implementation)\b",
        r"\b(?:we|our)\s+(?:propose|develop|implement|design)\b",
        r"\b(?:hardware|circuit|components|sim808)\b",
    ),
    "evaluation": (
        r"\b(?:experimental setup|field trial|test cases?)\b",
        r"\b(?:we|our)\s+(?:tested|evaluated|conducted)\b",
        r"\b(?:testing|experiments?|evaluation)\b",
    ),
    "results": (
        r"\bresults?\s+(?:and|of|are|show|indicate)\b",
        r"\b(?:experimental|simulation|field)\s+results?\b",
        r"\b(?:measured|observed|total cost|accuracy)\b",
    ),
    "limitations": (
        r"\b(?:limitations?|future (?:work|scope))\b",
        r"\b(?:conclusion|challenges?|constraints?)\b",
        r"\b(?:however|restricted|limited to)\b",
    ),
}

EXCLUDE_PATTERNS = (
    r"\breferences\b",
    r"\bbibliography\b",
    r"\backnowledg(?:e)?ments?\b",
)

MAX_SECTION_SCAN_CHUNKS = 300
SECTION_CANDIDATES_PER_ASPECT = 12


def _embedding_list(text):
    embedding = generate_embedding(text)
    return (
        embedding.tolist()
        if hasattr(embedding, "tolist")
        else list(embedding)
    )


def _empty_results():
    return {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "intents": [[]],
    }


def _where(project_id=None, document_id=None):
    conditions = []

    if project_id:
        conditions.append({"project_id": project_id})

    if document_id:
        conditions.append({"document_id": document_id})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def search(
    query,
    top_k=5,
    project_id=None,
    document_id=None,
):
    """Preserve the existing semantic-search interface."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if document_id and not project_id:
        raise ValueError(
            "document_id requires project_id"
        )

    where = _where(project_id, document_id)
    kwargs = {"where": where} if where else {}

    count = len(
        collection.get(
            include=[],
            **kwargs,
        )["ids"]
    )

    if not count:
        return _empty_results()

    return collection.query(
        query_embeddings=[_embedding_list(query)],
        n_results=min(top_k, count),
        **kwargs,
    )


def _indexed_documents(project_id=None):
    kwargs = (
        {"where": {"project_id": project_id}}
        if project_id
        else {}
    )

    records = collection.get(
        include=["metadatas"],
        **kwargs,
    )

    documents = defaultdict(
        lambda: {
            "document": None,
            "document_id": None,
            "chunk_count": 0,
        }
    )

    for metadata in records.get("metadatas") or []:
        if not metadata:
            continue

        document = metadata.get("document")

        if not document:
            continue

        document_id = metadata.get("document_id")
        key = document_id or document

        entry = documents[key]
        entry["document"] = document
        entry["document_id"] = document_id
        entry["chunk_count"] += 1

    return sorted(
        documents.values(),
        key=lambda item: (
            item["document"].casefold(),
            str(item["document_id"] or ""),
        ),
    )


def _indexed_paper_counts(project_id=None):
    counts = {}

    for item in _indexed_documents(project_id):
        name = item["document"]
        counts[name] = (
            counts.get(name, 0)
            + item["chunk_count"]
        )

    return counts


def _indexed_papers(project_id=None):
    return sorted(
        _indexed_paper_counts(project_id)
    )


def _expand_query(question):
    question = question.strip()
    queries = [question]
    lower = question.casefold()

    intrusion_related = any(
        term in lower
        for term in (
            "intrusion",
            "cybersecurity",
            "cyber security",
            "network security",
        )
    ) or "ids" in lower.split()

    if intrusion_related:
        queries.extend([
            "Intrusion detection challenges and limitations",
            "Intrusion detection dataset quality and benchmarking",
            "Machine learning intrusion detection generalization",
            "Intrusion detection scalability and performance",
        ])

    return list(dict.fromkeys(queries))


def _aspect_queries(question, expand_queries):
    queries = [
        ("question", item)
        for item in (
            _expand_query(question)
            if expand_queries
            else [question.strip()]
        )
    ]

    queries.extend(
        RESEARCH_ASPECTS.items()
    )

    return queries


def _passage_key(metadata, text):
    return (
        metadata.get("document_id")
        or metadata.get("document"),
        str(metadata.get("page")),
        str(metadata.get("chunk_id")),
        (
            text[:100]
            if metadata.get("chunk_id") is None
            else ""
        ),
    )


def _section_score(text, aspect):
    """Find likely sections without asserting authorship."""
    text = text or ""
    score = sum(
        bool(re.search(pattern, text, re.I))
        for pattern in SECTION_PATTERNS[aspect]
    )

    # Explicit section headings are stronger signals.
    heading = re.search(
        r"(?:^|\n)\s*(?:[IVX\d.]+\s*[.:-]?\s*)?"
        r"(?:methodology|proposed system|system design|"
        r"implementation|experimental results|"
        r"results and discussion|evaluation|"
        r"conclusion and future scope|limitations)"
        r"\b",
        text,
        re.I,
    )

    if heading:
        score += 3

    # Objectives, literature surveys and reference lists
    # should not displace original experimental evidence.
    if re.search(
        r"\b(?:literature survey|related work|"
        r"review of literature|objective)\b",
        text,
        re.I,
    ):
        score -= 3

    if any(
        re.search(pattern, text, re.I)
        for pattern in EXCLUDE_PATTERNS
    ):
        score -= 5

    return score


def _merge_candidate(
    candidates,
    text,
    metadata,
    distance,
    intent,
):
    if not text or not metadata:
        return

    key = _passage_key(metadata, text)

    if key not in candidates:
        candidates[key] = {
            "text": text,
            "metadata": metadata,
            "distance": distance,
            "intents": set(),
            "intent_distances": {},
        }

    candidate = candidates[key]
    candidate["intents"].add(intent)

    previous = candidate[
        "intent_distances"
    ].get(intent)

    if previous is None or distance < previous:
        candidate[
            "intent_distances"
        ][intent] = distance

    if distance < candidate["distance"]:
        candidate["distance"] = distance


def _section_candidates(where, candidates):
    """Recover section-bearing passages missed by embeddings.

    The scan is bounded per paper and does not alter stored data.
    """
    records = collection.get(
        where=where,
        include=["documents", "metadatas"],
        limit=MAX_SECTION_SCAN_CHUNKS,
    )

    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    for aspect in RESEARCH_ASPECTS:
        ranked = []

        for text, metadata in zip(
            documents,
            metadatas,
        ):
            if not text or not metadata:
                continue

            score = _section_score(
                text,
                aspect,
            )

            if score > 0:
                ranked.append((
                    score,
                    text,
                    metadata,
                ))

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(item[2].get("page")),
                str(item[2].get("chunk_id")),
            )
        )

        for score, text, metadata in ranked[
            :SECTION_CANDIDATES_PER_ASPECT
        ]:
            # This is a selection placeholder, not a
            # fabricated semantic similarity measurement.
            _merge_candidate(
                candidates,
                text,
                metadata,
                distance=1.0,
                intent=aspect,
            )


def search_across_papers(
    query,
    per_paper_k=5,
    expand_queries=True,
    project_id=None,
):
    """Retrieve semantic and section-aware candidates per paper."""
    if per_paper_k < 1:
        raise ValueError(
            "per_paper_k must be at least 1"
        )

    if not query or not query.strip():
        raise ValueError(
            "query must not be empty"
        )

    indexed = _indexed_documents(
        project_id
    )

    if not indexed:
        return _empty_results()

    search_queries = _aspect_queries(
        query,
        expand_queries,
    )

    embeddings = {
        text: _embedding_list(text)
        for _, text in search_queries
    }

    combined = _empty_results()

    for paper in indexed:
        candidates = {}
        document_id = paper["document_id"]

        if document_id and project_id:
            where = _where(
                project_id,
                document_id,
            )
        else:
            conditions = [{
                "document": paper["document"]
            }]

            if project_id:
                conditions.insert(
                    0,
                    {"project_id": project_id},
                )

            where = (
                conditions[0]
                if len(conditions) == 1
                else {"$and": conditions}
            )

        search_limit = min(
            per_paper_k,
            paper["chunk_count"],
        )

        for intent, search_text in search_queries:
            results = collection.query(
                query_embeddings=[
                    embeddings[search_text]
                ],
                n_results=search_limit,
                where=where,
            )

            documents = (
                results.get("documents")
                or [[]]
            )
            metadatas = (
                results.get("metadatas")
                or [[]]
            )
            distances = (
                results.get("distances")
                or [[]]
            )

            for text, metadata, distance in zip(
                documents[0],
                metadatas[0],
                distances[0],
            ):
                _merge_candidate(
                    candidates,
                    text,
                    metadata or {},
                    distance,
                    intent,
                )

        _section_candidates(
            where,
            candidates,
        )

        selected_keys = set()

        for intent in ASPECT_ORDER:
            ranked = sorted(
                (
                    (key, candidate)
                    for key, candidate
                    in candidates.items()
                    if intent in candidate["intents"]
                ),
                key=lambda pair: (
                    pair[1]["intent_distances"][
                        intent
                    ],
                    str(pair[0]),
                ),
            )

            selected_keys.update(
                key
                for key, _ in ranked[
                    :per_paper_k
                ]
            )

        # Also retain section-discovered candidates.
        # They will be scored by the evidence selector.
        for key, candidate in candidates.items():
            if any(
                _section_score(
                    candidate["text"],
                    aspect,
                ) >= 3
                for aspect in RESEARCH_ASPECTS
            ):
                selected_keys.add(key)

        ordered = sorted(
            (
                candidates[key]
                for key in selected_keys
            ),
            key=lambda item: (
                item["distance"],
                str(
                    item["metadata"].get(
                        "chunk_id"
                    )
                ),
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
            combined["intents"][0].append([
                intent
                for intent in ASPECT_ORDER
                if intent in candidate["intents"]
            ])

    return combined