
"""
ResearchLens evidence analysis pipeline.

Source-aware retrieval -> balanced reranking ->
evidence extraction -> claim grouping ->
thematic discovery -> relationship analysis ->
evidence audit -> cross-paper comparison.

Thematic discovery may examine additional retrieved
passages without changing the main answer evidence.
"""

import logging

from backend.retrieval.retriever import search_across_papers
from backend.retrieval.result_formatter import (
    format_retrieval_results,
)
from backend.retrieval.reranker import rerank

from backend.analysis.evidence_builder import build_evidence
from backend.analysis.claim_grouper import group_claims
from backend.analysis.theme_grouper import (
    group_evidence_by_theme,
)
from backend.analysis.theme_comparator import (
    compare_cross_paper_themes,
)
from backend.analysis.relationship_analyzer import (
    build_evidence_relationships,
)
from backend.analysis.evidence_audit import (
    audit_evidence_relationships,
)


logger = logging.getLogger(__name__)


def _evidence_key(item):
    """Identify a passage without relying on object identity."""

    if isinstance(item, dict):
        return (
            item.get("document"),
            item.get("chunk_id"),
        )

    return (
        item.paper,
        item.chunk_id,
    )


def _select_evidence(question, candidates, limit):
    """Rerank independently per paper and balance selection."""

    if not candidates or limit < 1:
        return []

    papers = sorted({
        item["document"]
        for item in candidates
    })

    ranked_by_paper = {}

    for paper in papers:
        paper_candidates = [
            item
            for item in candidates
            if item["document"] == paper
        ]

        ranked_by_paper[paper] = rerank(
            question,
            paper_candidates,
            top_k=len(paper_candidates),
        )

    if limit < len(papers):
        leaders = [
            ranked_by_paper[paper][0]
            for paper in papers
            if ranked_by_paper[paper]
        ]

        return sorted(
            leaders,
            key=lambda item: item["reranker_score"],
            reverse=True,
        )[:limit]

    per_paper_limit = limit // len(papers)

    selected = []
    remaining = []

    for paper in papers:
        ranked = ranked_by_paper[paper]
        selected.extend(ranked[:per_paper_limit])
        remaining.extend(ranked[per_paper_limit:])

    remaining.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    selected.extend(
        remaining[:limit - len(selected)]
    )

    selected.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return selected


def _empty_result(question):
    return {
        "question": question,
        "status": "no_evidence",
        "evidence": [],
        "claim_groups": [],
        "themes": [],
        "theme_comparisons": [],
        "relationships": [],
        "audits": [],
        "analysis_failures": [],
        "source_count": 0,
        "cross_paper_evidence": False,
        "cross_paper_theme_count": 0,
    }


def _discover_additional_themes(
    question,
    candidates,
    selected_results,
    evidence_items,
    extra_per_paper=3,
):
    """
    Expand thematic discovery only when initial evidence
    has no cross-paper themes.

    Reuses retrieved candidates rather than running
    another vector search.
    """

    initial_themes = group_evidence_by_theme(evidence_items)

    if any(
        theme["cross_paper"]
        for theme in initial_themes
    ):
        return initial_themes

    selected_keys = {
        _evidence_key(item)
        for item in selected_results
    }

    remaining = [
        item
        for item in candidates
        if _evidence_key(item) not in selected_keys
    ]

    if not remaining:
        return initial_themes

    papers = {
        item["document"]
        for item in candidates
    }

    if len(papers) < 2:
        return initial_themes

    # Limit extra LLM extraction calls. These passages
    # support theme discovery, not the main claim audit.
    discovery_limit = extra_per_paper * len(papers)

    extra_results = _select_evidence(
        question,
        remaining,
        limit=discovery_limit,
    )

    if not extra_results:
        return initial_themes

    try:
        extra_evidence = build_evidence(extra_results)
    except Exception:
        logger.exception(
            "Additional thematic evidence extraction failed"
        )
        return initial_themes

    # Deduplicate by source passage and extracted claim.
    combined = []
    seen = set()

    for item in [*evidence_items, *extra_evidence]:
        key = (
            item.paper,
            item.chunk_id,
            item.claim,
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(item)

    return group_evidence_by_theme(combined)


def run_evidence_pipeline(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75,
):
    """Run evidence analysis and conservative comparison."""

    # 1. Retrieve candidate passages from each paper.
    retrieval_results = search_across_papers(
        question,
        per_paper_k=max(3, retrieval_k),
        expand_queries=True,
    )

    formatted_results = format_retrieval_results(
        retrieval_results
    )

    if not formatted_results:
        return _empty_result(question)

    # 2. Select the passages for answering and auditing.
    selected_results = _select_evidence(
        question,
        formatted_results,
        limit=rerank_k,
    )

    if not selected_results:
        return _empty_result(question)

    # 3. Extract the main evidence.
    evidence_items = build_evidence(selected_results)

    # 4. Group claims from the main evidence only.
    claim_groups = group_claims(
        evidence_items,
        threshold=claim_threshold,
    )

    # 5. Discover themes. Expand the evidence pool
    # only if the main evidence has no cross-paper theme.
    themes = _discover_additional_themes(
        question=question,
        candidates=formatted_results,
        selected_results=selected_results,
        evidence_items=evidence_items,
    )

    # 6. Compare only themes supported by multiple papers.
    theme_comparisons = compare_cross_paper_themes(
        themes
    )

    source_names = {
        item.paper
        for item in evidence_items
        if item.claim and item.paper
    }

    cross_paper_theme_count = sum(
        bool(theme["cross_paper"])
        for theme in themes
    )

    # 7. Preserve existing claim-level relationship analysis.
    all_relationships = []
    audits = []
    analysis_failures = []

    for group_index, claim_group in enumerate(
        claim_groups,
        start=1,
    ):
        claim = claim_group["claim"]
        candidate_count = len(
            claim_group["evidence"]
        )

        try:
            relationships = build_evidence_relationships(
                claim_group
            )

            if len(relationships) != candidate_count:
                raise ValueError(
                    "Relationship count does not match "
                    "candidate evidence count."
                )

            audit = audit_evidence_relationships(
                claim=claim,
                relationships=relationships,
                total_evidence=candidate_count,
            )

        except Exception:
            logger.exception(
                "Relationship analysis failed "
                "for claim group %s",
                group_index,
            )

            analysis_failures.append({
                "group_index": group_index,
                "claim": claim,
                "candidate_evidence": candidate_count,
                "reason": "relationship_analysis_failed",
            })

            relationships = []

            audit = audit_evidence_relationships(
                claim=claim,
                relationships=[],
                total_evidence=candidate_count,
            )

        all_relationships.extend(relationships)
        audits.append(audit)

    # 8. Report claim-audit status independently
    # of whether shared themes were discovered.
    if not claim_groups:
        status = "no_evidence"
    elif len(analysis_failures) == len(claim_groups):
        status = "failed"
    elif analysis_failures:
        status = "partial"
    else:
        status = "completed"

    return {
        "question": question,
        "status": status,
        "evidence": evidence_items,
        "claim_groups": claim_groups,
        "themes": themes,
        "theme_comparisons": theme_comparisons,
        "relationships": all_relationships,
        "audits": audits,
        "analysis_failures": analysis_failures,
        "source_count": len(source_names),
        "cross_paper_evidence": len(source_names) >= 2,
        "cross_paper_theme_count": cross_paper_theme_count,
    }