
"""
Evidence-grounded thematic grouping for ResearchLens.

Supports multiple research domains through explicit topic
indicators and optional sentence-embedding similarity.

A shared theme indicates a topic suitable for comparison.
It does not establish agreement, contradiction or
independent validation.
"""

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

THEMES = {
    "TRACKING_TECHNOLOGY": {
        "label": "Vehicle tracking and positioning methods",
        "description": (
            "Technologies used to identify, locate and monitor "
            "vehicles, including RFID, GPS, GSM and BLE."
        ),
        "indicators": (
            "vehicle tracking",
            "bus tracking",
            "fleet tracking",
            "vehicle monitoring",
            "bus monitoring",
            "location tracking",
            "position tracking",
            "positioning",
            "geolocation",
            "gps",
            "gsm",
            "rfid",
            "bluetooth low energy",
            "ble beacon",
            "beacon detection",
            "proximity sensing",
            "latitude",
            "longitude",
        ),
    },
    "TRACKING_ACCURACY": {
        "label": "Tracking accuracy and reliability",
        "description": (
            "Accuracy, signal availability, detection range "
            "and reliability of tracking technologies."
        ),
        "indicators": (
            "tracking accuracy",
            "location accuracy",
            "positioning accuracy",
            "gps accuracy",
            "detection accuracy",
            "detection range",
            "signal availability",
            "signal loss",
            "satellite signal",
            "gps signal",
            "signal reception",
            "tracking reliability",
            "location error",
            "positioning error",
            "missed detection",
            "false detection",
            "detection failure",
        ),
    },
    "ARRIVAL_ESTIMATION": {
        "label": "Arrival-time estimation and journey analysis",
        "description": (
            "Travel-time observations, route monitoring "
            "and predicted arrival times."
        ),
        "indicators": (
            "estimated time of arrival",
            "arrival time",
            "arrival estimation",
            "eta prediction",
            "journey duration",
            "journey time",
            "travel time",
            "bus schedule",
            "route monitoring",
            "traffic condition",
            "journey prediction",
        ),
    },
    "COMMUNICATION": {
        "label": "Communication and data transmission",
        "description": (
            "Communication channels and data exchange "
            "between tracking devices and users."
        ),
        "indicators": (
            "data transmission",
            "communication network",
            "wireless communication",
            "mobile communication",
            "sms",
            "text message",
            "cloud server",
            "cloud platform",
            "internet connection",
            "gsm network",
            "bluetooth communication",
            "real time updates",
            "real time information",
        ),
    },
    "COST_AND_RESOURCES": {
        "label": "Cost and resource requirements",
        "description": (
            "Hardware cost, operating cost, power "
            "consumption and resource constraints."
        ),
        "indicators": (
            "cost effective",
            "low cost",
            "hardware cost",
            "installation cost",
            "operating cost",
            "maintenance cost",
            "affordable",
            "budget",
            "energy efficient",
            "power consumption",
            "battery life",
            "computational cost",
            "resource constraint",
        ),
    },
    "DEPLOYMENT": {
        "label": "Infrastructure and real-world deployment",
        "description": (
            "Practical implementation, infrastructure, "
            "scalability and deployment limitations."
        ),
        "indicators": (
            "infrastructure",
            "deployment",
            "installation",
            "bus stop",
            "field trial",
            "real world testing",
            "real world deployment",
            "prototype",
            "scalability",
            "scalable",
            "maintenance",
            "device failure",
            "coverage",
            "network coverage",
            "system reliability",
        ),
    },
    "DATASET_LIMITATIONS": {
        "label": "Dataset quality and generalizability",
        "description": (
            "Training-data quality, representativeness, "
            "benchmark coverage and generalization."
        ),
        "indicators": (
            "class imbalance",
            "imbalanced data",
            "label noise",
            "dataset quality",
            "data quality",
            "data scarcity",
            "insufficient training data",
            "limited training data",
            "single dataset",
            "limited dataset",
            "outdated dataset",
            "dataset shortage",
            "generalization",
            "generalisation",
            "generalizability",
            "generalizability",
            "benchmark dataset",
            "cross dataset",
        ),
    },
    "DETECTION_ERRORS": {
        "label": "Detection and classification errors",
        "description": (
            "False alarms, missed detections and "
            "classification performance limitations."
        ),
        "indicators": (
            "false positive",
            "false negative",
            "detection error",
            "misclassification",
            "incorrect classification",
            "missed attack",
            "missed detection",
            "classification accuracy",
            "precision",
            "recall",
            "f1 score",
        ),
    },
    "SCALABILITY": {
        "label": "Scalability and computational deployment",
        "description": (
            "Processing latency, computational requirements "
            "and large-scale system deployment."
        ),
        "indicators": (
            "high traffic",
            "high speed network",
            "large traffic volume",
            "large volumes",
            "network traffic",
            "scalability",
            "scalable",
            "latency",
            "computational cost",
            "deployment complexity",
            "deployment constraint",
            "real world deployment",
            "resource constraint",
        ),
    },
}

_EMBEDDING_MODEL = None
_EMBEDDING_UNAVAILABLE = False

# A relatively high threshold limits broad, misleading matches.
SEMANTIC_THRESHOLD = 0.56


def _normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[-‐‑–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matches(text, indicator):
    indicator = _normalize(indicator)

    if not indicator:
        return False

    return bool(
        re.search(
            rf"(?<!\w){re.escape(indicator)}(?!\w)",
            text,
        )
    )


def _get_embedding_model():
    global _EMBEDDING_MODEL
    global _EMBEDDING_UNAVAILABLE

    if _EMBEDDING_UNAVAILABLE:
        return None

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        return _EMBEDDING_MODEL
    except Exception:
        logger.exception(
            "Semantic theme model unavailable; "
            "using explicit topic indicators."
        )
        _EMBEDDING_UNAVAILABLE = True
        return None


def _semantic_matches(claims):
    """
    Find semantically relevant themes for claims.

    Returns one set of matching theme IDs per claim.
    Semantic matching supplements, rather than overrides,
    explicit topic indicators.
    """
    if not claims:
        return []

    model = _get_embedding_model()

    if model is None:
        return [set() for _ in claims]

    theme_ids = list(THEMES)

    descriptions = [
        (
            THEMES[theme_id]["label"]
            + ". "
            + THEMES[theme_id]["description"]
        )
        for theme_id in theme_ids
    ]

    try:
        from sentence_transformers import util

        claim_vectors = model.encode(
            claims,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        theme_vectors = model.encode(
            descriptions,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = util.cos_sim(
            claim_vectors,
            theme_vectors,
        )

        matches = []

        for row in similarities:
            scores = row.tolist()
            matched = {
                theme_ids[index]
                for index, score in enumerate(scores)
                if score >= SEMANTIC_THRESHOLD
            }
            matches.append(matched)

        return matches

    except Exception:
        logger.exception(
            "Semantic theme matching failed; "
            "continuing with explicit indicators."
        )
        return [set() for _ in claims]


def group_evidence_by_theme(evidence_items):
    """
    Group evidence into meaningful research themes.

    Only extracted claims are classified. Original
    passages remain available for source inspection.
    """
    usable = [
        item
        for item in evidence_items
        if item.paper
        and isinstance(item.claim, str)
        and item.claim.strip()
    ]

    if not usable:
        return []

    claim_texts = [_normalize(item.claim) for item in usable]

    # First identify explicit matches. Semantic matching
    # is only needed for claims with no explicit matches.
    explicit_matches = []

    for claim_text in claim_texts:
        matches = {
            theme_id
            for theme_id, definition in THEMES.items()
            if any(
                _matches(claim_text, indicator)
                for indicator in definition["indicators"]
            )
        }
        explicit_matches.append(matches)

    unmatched_indices = [
        index
        for index, matches in enumerate(explicit_matches)
        if not matches
    ]

    semantic_by_index = defaultdict(set)

    if unmatched_indices:
        semantic_results = _semantic_matches(
            [claim_texts[index] for index in unmatched_indices]
        )

        for index, matched in zip(
            unmatched_indices,
            semantic_results,
        ):
            semantic_by_index[index].update(matched)

    groups = {}
    seen = defaultdict(set)

    for index, evidence in enumerate(usable):
        matched_themes = (
            explicit_matches[index]
            | semantic_by_index[index]
        )

        evidence_key = (
            evidence.paper,
            evidence.page,
            evidence.chunk_id,
            evidence.claim,
        )

        for theme_id in THEMES:
            if theme_id not in matched_themes:
                continue

            if evidence_key in seen[theme_id]:
                continue

            seen[theme_id].add(evidence_key)

            if theme_id not in groups:
                groups[theme_id] = {
                    "theme_id": theme_id,
                    "theme": THEMES[theme_id]["label"],
                    "evidence": [],
                }

            groups[theme_id]["evidence"].append(evidence)

    for group in groups.values():
        sources = {
            item.paper
            for item in group["evidence"]
            if item.paper
        }

        group["source_count"] = len(sources)
        group["cross_paper"] = len(sources) >= 2

    return sorted(
        groups.values(),
        key=lambda group: (
            not group["cross_paper"],
            -group["source_count"],
            -len(group["evidence"]),
            group["theme"],
        ),
    )