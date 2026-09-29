
"""Project-scoped, source-aware cross-paper evidence analysis.

Selects complementary original research evidence across papers,
then performs claim grouping, thematic comparison and audits.
"""

import logging
import re
from collections import defaultdict

from backend.retrieval.retriever import search_across_papers
from backend.retrieval.result_formatter import format_retrieval_results
from backend.retrieval.reranker import rerank

from backend.analysis.evidence_builder import build_evidence
from backend.analysis.claim_grouper import group_claims
from backend.analysis.theme_grouper import group_evidence_by_theme
from backend.analysis.theme_comparator import compare_cross_paper_themes
from backend.analysis.relationship_analyzer import (
    build_evidence_relationships,
)
from backend.analysis.evidence_audit import (
    audit_evidence_relationships,
)


logger = logging.getLogger(__name__)

PASSAGES_PER_PAPER = 5

SELECTION_ROLES = (
    "implementation",
    "measured_results",
    "costs",
    "limitations",
    "complementary",
)

ASPECT_TERMS = {
    "methodology": (
        "proposed system",
        "proposed approach",
        "system architecture",
        "system design",
        "implementation",
        "implemented",
        "developed",
        "methodology",
        "proposed framework",
        "sim808",
        "raspberry pi",
        "rfid",
        "beacon",
        "tracking system",
    ),
    "evaluation": (
        "experimental setup",
        "field trial",
        "field trials",
        "evaluation",
        "experiment",
        "testing",
        "tested",
        "measurements",
        "deployment",
    ),
    "results": (
        "experimental results",
        "results and discussion",
        "our analysis shows",
        "results show",
        "measured",
        "observed",
        "performance",
        "average delay",
        "maximum delay",
        "on-time",
        "journey time",
        "detection range",
    ),
    "costs": (
        "cost benefits analysis",
        "cost benefit",
        "cost comparison",
        "total cost",
        "installation cost",
        "recurring cost",
        "monthly cost",
        "monthly recurring",
        "component cost",
        "cost incurred",
        "cost estimation",
        "cost effective",
        "cost-effective",
        "price",
        "subscription",
        "year 2",
        "parity",
    ),
    "limitations": (
        "limitations",
        "future work",
        "future scope",
        "deployment challenges",
        "challenges",
        "constraints",
        "restricted",
        "limited",
        "maintenance",
        "battery replacement",
        "degradation",
    ),
}

OWN_WORK_SIGNALS = (
    r"\b(?:we|our)\s+"
    r"(?:propose|proposed|implemented|developed|"
    r"designed|tested|evaluated|conducted|"
    r"observed|deployed|present|analysis)\b",
    r"\bin this (?:paper|study|work),?\s+"
    r"(?:we|the authors)\b",
    r"\b(?:our|the proposed)\s+"
    r"(?:system|method|prototype|approach|"
    r"deployment|framework|concept)\b",
    r"\b(?:our|the)\s+(?:field trial|experiment|"
    r"experimental results|implementation)\b",
)

RELATED_WORK_SIGNALS = (
    r"\brelated work\b",
    r"\bliterature (?:survey|review)\b",
    r"\breview of literature\b",
    r"\bet al\.\b",
    r"\b(?:previous|earlier|prior)\s+"
    r"(?:study|work|research|approach)\b",
    r"\b(?:a study by|in a study by|"
    r"the study by|the work of)\b",
    r"\b(?:a more recent study|"
    r"in a different study)\b",
)

WEAK_SECTION_SIGNALS = (
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*acknowledg(?:e)?ments?\s*$",
)

IMPLEMENTATION_PATTERNS = (
    r"\b(?:installed|attached|placed|equipped|"
    r"connected|mounted|deployed)\b",
    r"\b(?:beacons?|sensors?|microcontrollers?|"
    r"modules?|devices?|readers?)\b",
    r"\b(?:raspberry pi|rpi|arduino|esp8266|"
    r"sim808|estimote|rfid|gps|gsm)\b",
    r"\b(?:transmits?|broadcasts?|scans?|"
    r"detects?|sends?|receives?|uploads?)\b",
    r"\b(?:cloud|server|dashboard|bus stops?|"
    r"smartphones?|applications?)\b",
)

IMPLEMENTATION_PHRASES = (
    r"\b(?:placed|attached|installed|mounted)\s+"
    r"(?:to|on|at|inside)\b",
    r"\b(?:will|then)\s+"
    r"(?:send|transmit|detect|upload)\b",
    r"\b(?:data|location|information)\s+"
    r"(?:is|are|was|were)\s+"
    r"(?:sent|transmitted|uploaded|stored)\b",
    r"\b(?:using|utilizing|integrating)\s+"
    r"(?:an?|the)\b",
)

FIELD_RESULT_PATTERNS = (
    r"\b(?:experiment|experimental)\s+"
    r"results?\s+show\b",
    r"\bfield trials?\b",
    r"\bfield tests?\b",
    r"\b(?:detected up to|detection range)\b",
    r"\b(?:our analysis shows|we observed|"
    r"we measured)\b",
    r"\b(?:average|maximum|minimum)\s+"
    r"(?:delay|accuracy|error|journey time)\b",
)

LIMITATION_PATTERNS = (
    r"\b(?:requires?|required)\s+"
    r"(?:timely\s+)?(?:maintenance|replacement|"
    r"additional)\b",
    r"\b(?:hardware|battery)\s+"
    r"(?:degradation|replacement|failure)\b",
    r"\b(?:deployment challenges?|"
    r"practical constraints?|limitations?)\b",
    r"\b(?:additional|more)\s+"
    r"(?:devices?|sites?|infrastructure)\b",
    r"\b(?:limited|restricted|constrained)\s+to\b",
)

COST_AMOUNT = re.compile(
    r"(?:US\s*\$|\$|BDT\s*|Tk\.?\s*)"
    r"[\d,]+(?:\.\d+)?",
    re.IGNORECASE,
)

COST_EVIDENCE = re.compile(
    r"\b(?:total|component|installation|"
    r"recurring|monthly|estimated|operational)"
    r"\s+costs?\b"
    r"|\bcosts?\s+(?:estimation|comparison|"
    r"analysis|parity)\b"
    r"|\bcost\s+benefits?\s+analysis\b"
    r"|\b(?:year\s*2|cost parity)\b",
    re.IGNORECASE,
)

# Require results-oriented context rather than treating
# any number in a hardware description as an experiment.
MEASURED_CONTEXT = re.compile(
    r"\b(?:experimental results?|field trials?|"
    r"field tests?|our analysis|we observed|"
    r"we measured|results show|results showed|"
    r"detected up to|detection range|"
    r"average delay|maximum delay|"
    r"journey times?|on-time|"
    r"accuracy|precision|recall|"
    r"performance evaluation)\b",
    re.IGNORECASE,
)

NUMERIC_RESULT = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:%|percent|minutes?|mins?|"
    r"meters?|metres?|seconds?|hours?|"
    r"km|m\b)",
    re.IGNORECASE,
)

HARDWARE_SPECIFICATIONS = re.compile(
    r"\b(?:antenna|voltage|frequency|"
    r"current|gain|power consumption|"
    r"operating frequency|milliamps|"
    r"mah|mhz|ghz|db)\b",
    re.IGNORECASE,
)


def _paper_key(item):
    if isinstance(item, dict):
        return (
            item.get("document_id")
            or item.get("document")
        )

    return item.paper


def _evidence_key(item):
    if isinstance(item, dict):
        return (
            _paper_key(item),
            str(item.get("page")),
            str(item.get("chunk_id")),
            (
                str(item.get("text", ""))[:100]
                if item.get("chunk_id") is None
                else ""
            ),
        )

    return (
        item.paper,
        str(item.page),
        str(item.chunk_id),
        "",
    )


def _pattern_count(text, patterns):
    return sum(
        bool(re.search(pattern, text, re.I))
        for pattern in patterns
    )


def _is_related_work(text):
    """Conservative heuristic; the extractor checks attribution."""
    own = _pattern_count(
        text,
        OWN_WORK_SIGNALS,
    )

    related = _pattern_count(
        text,
        RELATED_WORK_SIGNALS,
    )

    return related > 0 and own == 0


def _is_weak_section(text):
    """Exclude obvious reference and acknowledgement material."""
    lines = text.splitlines()

    return any(
        re.search(pattern, line, re.I)
        for line in lines
        for pattern in WEAK_SECTION_SIGNALS
    )


def _has_measured_result(text):
    """Require experimental context, not just numerical specs."""
    if not MEASURED_CONTEXT.search(text):
        return False

    return bool(
        NUMERIC_RESULT.search(text)
        or _pattern_count(
            text,
            FIELD_RESULT_PATTERNS,
        )
    )


def _has_cost_evidence(text):
    return bool(
        COST_AMOUNT.search(text)
        or COST_EVIDENCE.search(text)
    )


def _has_limitation_evidence(text):
    return bool(
        _pattern_count(
            text,
            LIMITATION_PATTERNS,
        )
    )


def _implementation_strength(text):
    concrete = _pattern_count(
        text,
        IMPLEMENTATION_PATTERNS,
    )

    workflow = _pattern_count(
        text,
        IMPLEMENTATION_PHRASES,
    )

    return concrete, workflow


def _has_implementation_evidence(text):
    concrete, workflow = _implementation_strength(
        text
    )

    return (
        concrete >= 3
        or (
            concrete >= 2
            and workflow >= 1
        )
    )


def _has_original_research_signal(text):
    """Avoid filling complementary slots with pure background."""
    if _pattern_count(
        text,
        OWN_WORK_SIGNALS,
    ):
        return True

    if _has_measured_result(text):
        return True

    if _has_cost_evidence(text):
        return True

    if _has_limitation_evidence(text):
        return True

    # Concrete architecture passages can be useful even
    # when their author attribution appears in an
    # adjacent paragraph. The extractor verifies them.
    concrete, workflow = _implementation_strength(
        text
    )

    return concrete >= 3 and workflow >= 1


def _passage_score(item, aspect, rank=0):
    """Score a research aspect without equating relevance with proof."""
    text = item.get("text") or ""
    lower = text.casefold()

    intents = item.get(
        "retrieval_intents"
    ) or []

    score = 0.0

    if aspect in intents:
        score += 2.0

    matches = sum(
        term in lower
        for term in ASPECT_TERMS[aspect]
    )

    score += min(matches, 5) * 1.7

    own = _pattern_count(
        text,
        OWN_WORK_SIGNALS,
    )

    score += min(own, 2) * 3.0

    if aspect == "methodology":
        concrete, workflow = (
            _implementation_strength(text)
        )

        score += concrete * 3.0
        score += workflow * 3.0

        if re.search(
            r"\b(?:we implemented|we developed|"
            r"we designed|our proposed|"
            r"in this paper,? we propose)\b",
            text,
            re.I,
        ):
            score += 5.0

        # Prefer architecture and data flow over
        # isolated component specifications.
        if (
            HARDWARE_SPECIFICATIONS.search(text)
            and workflow == 0
        ):
            score -= 9.0

    elif aspect == "evaluation":
        if _has_measured_result(text):
            score += 12.0

        if NUMERIC_RESULT.search(text):
            score += 4.0

    elif aspect == "results":
        if _has_measured_result(text):
            score += 14.0

        if NUMERIC_RESULT.search(text):
            score += 4.0

    elif aspect == "costs":
        if COST_AMOUNT.search(text):
            score += 9.0

        if COST_EVIDENCE.search(text):
            score += 8.0

    elif aspect == "limitations":
        score += (
            _pattern_count(
                text,
                LIMITATION_PATTERNS,
            ) * 6.0
        )

    if _is_related_work(text):
        score -= 25.0

    if _is_weak_section(text):
        score -= 25.0

    if len(text.strip()) < 120:
        score -= 3.0

    # Small relevance tie-breaker.
    score += max(
        0.0,
        2.0 - rank * 0.08,
    )

    return score


def _role_has_evidence(item, role):
    """Never force a paper to supply an unsupported aspect."""
    text = item.get("text") or ""

    if not text.strip():
        return False

    if (
        _is_related_work(text)
        or _is_weak_section(text)
    ):
        return False

    if role == "implementation":
        return _has_implementation_evidence(
            text
        )

    if role == "measured_results":
        return _has_measured_result(
            text
        )

    if role == "costs":
        return _has_cost_evidence(
            text
        )

    if role == "limitations":
        return _has_limitation_evidence(
            text
        )

    if role == "complementary":
        return _has_original_research_signal(
            text
        )

    return False


def _role_score(item, role, rank=0):
    """Score a passage for a specific selection role."""
    text = item.get("text") or ""

    if role == "implementation":
        score = _passage_score(
            item,
            "methodology",
            rank,
        )

        concrete, workflow = (
            _implementation_strength(text)
        )

        score += concrete * 4.0
        score += workflow * 6.0

        if workflow == 0:
            score -= 6.0

    elif role == "measured_results":
        score = max(
            _passage_score(
                item,
                "results",
                rank,
            ),
            _passage_score(
                item,
                "evaluation",
                rank,
            ),
        )

        score += (
            _pattern_count(
                text,
                FIELD_RESULT_PATTERNS,
            ) * 5.0
        )

        if NUMERIC_RESULT.search(text):
            score += 5.0

    elif role == "costs":
        score = _passage_score(
            item,
            "costs",
            rank,
        )

        if COST_AMOUNT.search(text):
            score += 5.0

        if COST_EVIDENCE.search(text):
            score += 5.0

    elif role == "limitations":
        score = _passage_score(
            item,
            "limitations",
            rank,
        )

        score += (
            _pattern_count(
                text,
                LIMITATION_PATTERNS,
            ) * 5.0
        )

    else:
        score = max(
            _passage_score(
                item,
                aspect,
                rank,
            )
            for aspect in ASPECT_TERMS
        )

        # Reward genuinely complementary evidence,
        # rather than selecting general context
        # merely to fill the quota.
        if _has_measured_result(text):
            score += 4.0

        if _has_cost_evidence(text):
            score += 4.0

        if _has_limitation_evidence(text):
            score += 4.0

    if _is_related_work(text):
        score -= 25.0

    return score


def _rank_for_role(
    candidates,
    role,
    rank_map,
):
    return sorted(
        candidates,
        key=lambda item: (
            -_role_score(
                item,
                role,
                rank_map.get(
                    _evidence_key(item),
                    len(rank_map),
                ),
            ),
            str(item.get("page")),
            str(item.get("chunk_id")),
        ),
    )


def _select_evidence(
    question,
    candidates,
    limit,
):
    """
    Select balanced original research evidence.

    First reserve distinct research aspects per paper.
    Then fill remaining slots with strong complementary
    passages. Do not force unsupported aspects or
    select background merely to reach the limit.
    """
    if not candidates or limit < 1:
        return []

    by_paper = defaultdict(list)
    seen_candidates = set()

    for item in candidates:
        if not isinstance(item, dict):
            continue

        if not item.get("document"):
            continue

        if not (
            item.get("text") or ""
        ).strip():
            continue

        key = _evidence_key(item)

        if key in seen_candidates:
            continue

        seen_candidates.add(key)

        by_paper[
            _paper_key(item)
        ].append(item)

    if not by_paper:
        return []

    papers = sorted(
        by_paper,
        key=str,
    )

    rank_maps = {}

    for paper in papers:
        ranked = rerank(
            question,
            by_paper[paper],
            top_k=len(
                by_paper[paper]
            ),
        )

        rank_maps[paper] = {
            _evidence_key(item): index
            for index, item in enumerate(
                ranked
            )
        }

    selected = []
    selected_keys = set()

    selected_pages = defaultdict(
        set
    )

    per_paper_count = defaultdict(
        int
    )

    base_quota, remainder = divmod(
        limit,
        len(papers),
    )

    quotas = {
        paper: (
            base_quota
            + (
                1
                if index < remainder
                else 0
            )
        )
        for index, paper in enumerate(
            papers
        )
    }

    def add(paper, item):
        key = _evidence_key(
            item
        )

        if key in selected_keys:
            return False

        selected_keys.add(key)
        selected.append(item)

        per_paper_count[
            paper
        ] += 1

        selected_pages[
            paper
        ].add(
            str(item.get("page"))
        )

        return True

    def best_available(
        paper,
        role,
    ):
        available = [
            item
            for item in by_paper[
                paper
            ]
            if (
                _evidence_key(item)
                not in selected_keys
                and _role_has_evidence(
                    item,
                    role,
                )
            )
        ]

        if not available:
            return None

        ranked = _rank_for_role(
            available,
            role,
            rank_maps[paper],
        )

        best = ranked[0]

        best_score = _role_score(
            best,
            role,
            rank_maps[paper].get(
                _evidence_key(best),
                0,
            ),
        )

        # Prefer page diversity only when the
        # alternative is nearly as relevant.
        for item in ranked:
            page = str(
                item.get("page")
            )

            if (
                page
                in selected_pages[
                    paper
                ]
            ):
                continue

            score = _role_score(
                item,
                role,
                rank_maps[
                    paper
                ].get(
                    _evidence_key(
                        item
                    ),
                    0,
                ),
            )

            if (
                score
                >= best_score - 3.0
            ):
                return item

        return best

    # Reserve core aspects. If a paper does
    # not report an aspect, skip that slot.
    for role in (
        "implementation",
        "measured_results",
        "costs",
        "limitations",
    ):
        for paper in papers:
            if (
                len(selected)
                >= limit
            ):
                return selected

            if (
                per_paper_count[
                    paper
                ]
                >= quotas[paper]
            ):
                continue

            item = best_available(
                paper,
                role,
            )

            if item is not None:
                add(
                    paper,
                    item,
                )

    # Fill each paper's remaining quota with
    # passages containing meaningful evidence.
    for paper in papers:
        while (
            len(selected) < limit
            and per_paper_count[
                paper
            ] < quotas[
                paper
            ]
        ):
            item = best_available(
                paper,
                "complementary",
            )

            if item is None:
                break

            add(
                paper,
                item,
            )

    # Redistribute unused capacity to papers
    # that have additional suitable evidence.
    while (
        len(selected) < limit
    ):
        progress = False

        for paper in papers:
            item = best_available(
                paper,
                "complementary",
            )

            if item is None:
                continue

            if add(
                paper,
                item,
            ):
                progress = True

            if (
                len(selected)
                >= limit
            ):
                break

        if not progress:
            break

    logger.info(
        "Selected %s of at most %s "
        "passages across %s papers",
        len(selected),
        limit,
        len(papers),
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
    """Search for additional themes only when necessary."""
    initial_themes = (
        group_evidence_by_theme(
            evidence_items
        )
    )

    if any(
        theme.get(
            "cross_paper",
            False,
        )
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
        if (
            _evidence_key(item)
            not in selected_keys
        )
    ]

    papers = {
        _paper_key(item)
        for item in candidates
    }

    if (
        not remaining
        or len(papers) < 2
    ):
        return initial_themes

    extra_results = (
        _select_evidence(
            question,
            remaining,
            limit=(
                extra_per_paper
                * len(papers)
            ),
        )
    )

    if not extra_results:
        return initial_themes

    try:
        extra_evidence = (
            build_evidence(
                extra_results
            )
        )

    except Exception:
        logger.exception(
            "Additional thematic "
            "evidence extraction failed"
        )

        return initial_themes

    combined = []
    seen = set()

    for item in (
        list(evidence_items)
        + list(extra_evidence)
    ):
        if not item.claim:
            continue

        key = (
            item.paper,
            item.page,
            item.chunk_id,
            item.claim,
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(
            item
        )

    return (
        group_evidence_by_theme(
            combined
        )
    )


def run_evidence_pipeline(
    question,
    retrieval_k=10,
    rerank_k=5,
    claim_threshold=0.75,
    project_id=None,
):
    """Run the full project-scoped research evidence audit."""
    retrieval_results = (
        search_across_papers(
            question,
            per_paper_k=max(
                3,
                retrieval_k,
            ),
            expand_queries=True,
            project_id=project_id,
        )
    )

    formatted_results = (
        format_retrieval_results(
            retrieval_results
        )
    )

    if not formatted_results:
        return _empty_result(
            question
        )

    paper_count = len({
        _paper_key(item)
        for item in formatted_results
    })

    effective_limit = max(
        rerank_k,
        (
            paper_count
            * PASSAGES_PER_PAPER
        ),
    )

    selected_results = (
        _select_evidence(
            question,
            formatted_results,
            limit=effective_limit,
        )
    )

    if not selected_results:
        return _empty_result(
            question
        )

    logger.info(
        "Selected %s passages "
        "across %s papers",
        len(selected_results),
        paper_count,
    )

    for item in selected_results:
        logger.info(
            "Evidence candidate: "
            "paper=%s page=%s "
            "chunk=%s intents=%s",
            item.get("document"),
            item.get("page"),
            item.get("chunk_id"),
            item.get(
                "retrieval_intents"
            ),
        )

    evidence_items = [
        item
        for item in build_evidence(
            selected_results
        )
        if item.claim
    ]

    if not evidence_items:
        return _empty_result(
            question
        )

    claim_groups = group_claims(
        evidence_items,
        threshold=claim_threshold,
    )

    themes = (
        _discover_additional_themes(
            question=question,
            candidates=formatted_results,
            selected_results=selected_results,
            evidence_items=evidence_items,
        )
    )

    theme_comparisons = (
        compare_cross_paper_themes(
            themes
        )
    )

    source_names = {
        item.paper
        for item in evidence_items
        if item.paper
    }

    cross_paper_theme_count = sum(
        bool(
            theme.get(
                "cross_paper",
                False,
            )
        )
        for theme in themes
    )

    all_relationships = []
    audits = []
    analysis_failures = []

    for group_index, claim_group in enumerate(
        claim_groups,
        start=1,
    ):
        claim = claim_group[
            "claim"
        ]

        candidate_count = len(
            claim_group[
                "evidence"
            ]
        )

        try:
            relationships = (
                build_evidence_relationships(
                    claim_group
                )
            )

            if (
                len(relationships)
                != candidate_count
            ):
                raise ValueError(
                    "Relationship count "
                    "does not match "
                    "candidate evidence count."
                )

            audit = (
                audit_evidence_relationships(
                    claim=claim,
                    relationships=relationships,
                    total_evidence=(
                        candidate_count
                    ),
                )
            )

        except Exception:
            logger.exception(
                "Relationship analysis "
                "failed for claim group %s",
                group_index,
            )

            analysis_failures.append({
                "group_index": (
                    group_index
                ),
                "claim": claim,
                "candidate_evidence": (
                    candidate_count
                ),
                "reason": (
                    "relationship_analysis_failed"
                ),
            })

            relationships = []

            audit = (
                audit_evidence_relationships(
                    claim=claim,
                    relationships=[],
                    total_evidence=(
                        candidate_count
                    ),
                )
            )

        all_relationships.extend(
            relationships
        )

        audits.append(
            audit
        )

    if not claim_groups:
        status = "no_evidence"

    elif (
        len(analysis_failures)
        == len(claim_groups)
    ):
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
        "theme_comparisons": (
            theme_comparisons
        ),
        "relationships": (
            all_relationships
        ),
        "audits": audits,
        "analysis_failures": (
            analysis_failures
        ),
        "source_count": len(
            source_names
        ),
        "cross_paper_evidence": (
            len(source_names) >= 2
        ),
        "cross_paper_theme_count": (
            cross_paper_theme_count
        ),
    }