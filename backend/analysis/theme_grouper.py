
"""
Evidence-grounded thematic grouping.

A shared theme identifies a topic worth comparing.
It does not establish agreement, contradiction,
independent corroboration, or empirical validation.
"""

import re


THEMES = {
    "DATASET_LIMITATIONS": {
        "label": "Dataset quality and generalizability",
        "indicators": (
            "class imbalance",
            "imbalanced data",
            "imbalanced dataset",
            "label noise",
            "noisy label",
            "dataset representation",
            "dataset construction",
            "dataset creation",
            "dataset quality",
            "data quality",
            "data scarcity",
            "insufficient training data",
            "limited training data",
            "single dataset",
            "one dataset",
            "limited dataset",
            "outdated dataset",
            "dataset shortage",
            "generalizab",
            "generaliz",
            "benchmarking gap",
            "lack of benchmark",
            "benchmark dataset",
            "multiple datasets",
            "various datasets",
            "cross-dataset",
        ),
    },
    "DETECTION_ERRORS": {
        "label": "Intrusion detection errors",
        "indicators": (
            "false positive",
            "false negative",
            "detection error",
            "misclassification",
            "incorrect classification",
            "missed attack",
            "missed detection",
        ),
    },
    "SCALABILITY": {
        "label": "Scalability and deployment",
        "indicators": (
            "high traffic",
            "high-speed network",
            "large traffic volume",
            "large volumes",
            "network traffic",
            "scalab",
            "latency",
            "computational cost",
            "deployment complexity",
            "deployment constraint",
            "real-world deployment",
            "resource constraint",
        ),
    },
}


def _normalize(text):
    """Normalize whitespace and hyphenation for matching."""

    text = (text or "").lower()
    text = re.sub(r"[-‐‑–—]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _matches(text, indicator):
    indicator = _normalize(indicator)

    # Prefix indicators intentionally match morphological variants.
    if indicator in ("generalizab", "generaliz", "scalab"):
        return bool(
            re.search(
                rf"\b{re.escape(indicator)}\w*",
                text,
            )
        )

    # Match whole phrases, not fragments inside unrelated words.
    return bool(
        re.search(
            rf"(?<!\w){re.escape(indicator)}(?!\w)",
            text,
        )
    )


def group_evidence_by_theme(evidence_items):
    """
    Group extracted claims by research topic.

    Only the extracted claim is classified. Original passage
    text remains available for source inspection.
    """

    groups = {}
    seen = {}

    for evidence in evidence_items:
        if not evidence.claim:
            continue

        claim_text = _normalize(evidence.claim)

        evidence_key = (
            evidence.paper,
            evidence.chunk_id,
            evidence.claim,
        )

        for theme_id, definition in THEMES.items():
            if not any(
                _matches(claim_text, indicator)
                for indicator in definition["indicators"]
            ):
                continue

            if theme_id not in groups:
                groups[theme_id] = {
                    "theme_id": theme_id,
                    "theme": definition["label"],
                    "evidence": [],
                }
                seen[theme_id] = set()

            if evidence_key in seen[theme_id]:
                continue

            seen[theme_id].add(evidence_key)
            groups[theme_id]["evidence"].append(evidence)

    for group in groups.values():
        sources = {
            item.paper
            for item in group["evidence"]
            if item.paper
        }

        group["source_count"] = len(sources)
        group["cross_paper"] = len(sources) >= 2

    return list(groups.values())