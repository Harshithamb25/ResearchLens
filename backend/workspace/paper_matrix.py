"""Evidence-grounded paper-wise matrix extraction.

The matrix is built from the original project PDFs. Candidate findings are
cleaned into complete, readable evidence sentences while preserving the
original source page and passage for verification.
"""

import csv
import io
import re
from pathlib import Path
from typing import Any, Callable

from backend.ingestion.pdf_loader import extract_text_from_pdf


NOT_IDENTIFIED = "Not identified in extracted evidence"

MATRIX_COLUMNS = [
    ("serial_number", "S. No."),
    ("paper_title", "Paper Title"),
    ("methodology", "Methodology / Algorithm used"),
    ("datasets", "Dataset(s)"),
    ("evaluation_metrics", "Eval Metrics"),
    ("accuracy_results", "Accuracy / Results"),
    ("advantages", "Advantages"),
    ("limitations", "Disadvantages / Limitations"),
    ("applications", "Applications"),
]

FIELDS = [
    key for key, _ in MATRIX_COLUMNS
    if key not in {"serial_number", "paper_title"}
]

MAX_ENTRIES = 6
MAX_PAGE_CHARS = 14000
MAX_FINDING_CHARS = 420


def _normalize(text: str) -> str:
    replacements = {
        "â¢": "•",
        "â€“": "–",
        "â€”": "—",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "ﬁ": "fi",
        "ﬂ": "fl",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_pdf_noise(text: str) -> bool:
    value = _normalize(text)
    lowered = value.casefold()

    noise_signals = [
        "doi:",
        "corresponding author",
        "all rights reserved",
        "open access",
        "creativecommons",
        "frontiersin.org",
        "received:",
        "accepted:",
        "published:",
        "issn",
        "e-issn",
        "pp1826-1833",
        "vol.",
        "journal of",
        "international conference on",
    ]

    if any(signal in lowered for signal in noise_signals):
        return True

    # Page numbers and academic header fragments are not useful findings.
    if re.search(r"\b(?:pp?\.?\s*)?\d{3,4}\s*[–-]\s*\d{3,4}\b", value):
        return True

    if re.fullmatch(r"\d+(?:\s+\d+)*", value):
        return True

    return False


def _split_sentences(text: str) -> list[str]:
    text = _normalize(text)
    if not text:
        return []

    # Preserve normal decimal numbers while splitting on sentence-ending
    # punctuation followed by a likely sentence start.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [_normalize(part) for part in parts if _normalize(part)]


def _complete_sentence(text: str) -> str:
    """Return one readable complete sentence, never a character-truncated one."""
    text = _normalize(text).strip(" -•·\t")
    if not text:
        return ""

    sentences = _split_sentences(text)
    if sentences:
        candidate = sentences[0]

        # A sentence ending in an obvious abbreviation should not be treated
        # as complete merely because the extractor encountered a period.
        if len(candidate) >= 35:
            if candidate[-1] not in ".!?":
                return ""
            return candidate

    return ""


def _sentence_context(text: str, start: int, end: int) -> str:
    """Find a complete sentence around a matched signal."""
    left_boundary = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
    )
    left = left_boundary + 1

    right_candidates = [
        position
        for position in (
            text.find(".", end),
            text.find("!", end),
            text.find("?", end),
        )
        if position != -1
    ]

    if not right_candidates:
        return ""

    right = min(right_candidates) + 1
    candidate = _complete_sentence(text[left:right])

    if candidate:
        return candidate

    return ""


def _clean_candidate(value: str) -> str:
    value = _normalize(value).strip(" -•·\t")
    if not value or _looks_like_pdf_noise(value):
        return ""

    # Remove common citation markers that can leak from extracted PDF text.
    value = re.sub(r"\s*\[[0-9,\-– ]+\]\s*", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    sentence = _complete_sentence(value)
    if not sentence:
        return ""

    if len(sentence) > MAX_FINDING_CHARS:
        # Do not create a broken sentence by slicing. Prefer the first
        # complete sentence if the input contains multiple sentences.
        sentences = _split_sentences(sentence)
        if sentences:
            sentence = sentences[0]

    if len(sentence) < 35:
        return ""

    return sentence


def _add(
    fields: dict[str, list[dict[str, Any]]],
    seen: dict[str, set[str]],
    field: str,
    value: str,
    evidence_text: str,
    filename: str,
    document_id: str,
    page: int,
) -> None:
    clean_value = _clean_candidate(value)
    clean_evidence = _normalize(evidence_text)

    if not clean_value:
        return

    # The finding must also be traceable to a real sentence in the source.
    if not clean_evidence:
        clean_evidence = clean_value

    key = clean_value.casefold()
    if key in seen[field] or len(fields[field]) >= MAX_ENTRIES:
        return

    seen[field].add(key)
    fields[field].append(
        {
            "value": clean_value,
            "evidence_text": clean_evidence,
            "source": {
                "paper": filename,
                "document_id": document_id,
                "page": page,
            },
        }
    )


def _match_patterns(
    text: str,
    patterns: list[tuple[str, str]],
) -> list[tuple[str, re.Match[str]]]:
    matches: list[tuple[str, re.Match[str]]] = []

    for label, pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((label, match))

    matches.sort(key=lambda item: (item[1].start(), item[0]))
    return matches


METHOD_PATTERNS = [
    (
        "Methodology",
        r"\b(?:we|our|this paper|the authors?)\s+"
        r"(?:propose|proposed|develop|developed|design|designed|"
        r"implement|implemented|use|used|adopt|adopted)\b"
        r"[^.!?\n]{0,260}[.!?]",
    ),
    (
        "System / architecture",
        r"\b(?:system architecture|system design|"
        r"proposed architecture|implementation design|"
        r"experimental setup)\b[^.!?\n]{0,260}[.!?]",
    ),
]

DATASET_PATTERNS = [
    (
        "Dataset",
        r"\b(?:dataset|data set)\b[^.!?\n]{0,260}[.!?]",
    ),
    (
        "Collected data",
        r"\b(?:data|measurements?)\s+(?:were|was)\s+"
        r"(?:collected|recorded|gathered)\b[^.!?\n]{0,260}[.!?]",
    ),
    (
        "Field-trial data",
        r"\b(?:field trial|field trials|pilot study|"
        r"real-world deployment|real-world use case)\b[^.!?\n]{0,260}[.!?]",
    ),
    (
        "Study population / sample",
        r"\b(?:sample|participants?|subjects?|observations?|cases?)\b"
        r"[^.!?\n]{0,220}[.!?]",
    ),
]

METRIC_PATTERNS = [
    (
        "Accuracy",
        r"\b(?:accuracy|accurate|precision|recall|F1(?:[- ]score)?|"
        r"sensitivity|specificity|AUC|RMSE|MAE|mAP)\b[^.!?\n]{0,180}[.!?]",
    ),
    (
        "Detection / range",
        r"\b(?:detection|detect(?:ed|ion)?|range)\b"
        r"[^.!?\n]{0,180}\b\d+(?:\.\d+)?\s*(?:m|km|cm|ms|s|%)\b[^.!?\n]*[.!?]",
    ),
    (
        "Delay / timing",
        r"\b(?:delay|latency|journey time|travel time|"
        r"on-time|early departures?|delayed departures?)\b"
        r"[^.!?\n]{0,180}[.!?]",
    ),
    (
        "Cost",
        r"\b(?:cost|price|budget|expenditure)\b[^.!?\n]{0,180}[.!?]",
    ),
]

RESULT_PATTERNS = [
    (
        "Quantitative result",
        r"\b(?:accuracy|precision|recall|F1(?:[- ]score)?|"
        r"on-time|early|delayed|detection|latency|delay|"
        r"journey time|travel time|cost|total cost)\b"
        r"[^.!?\n]{0,220}"
        r"\b\d+(?:\.\d+)?\s*(?:%|m|km|cm|ms|s|BDT|USD|RM|"
        r"minutes?|hours?)\b[^.!?\n]*[.!?]",
    ),
    (
        "Observed result",
        r"\b(?:results?|performance|experiment(?:al)?|"
        r"observed|measured|show(?:s|ed)?)\b[^.!?\n]{0,260}[.!?]",
    ),
    (
        "Tabulated result",
        r"\b(?:table|figure)\s+\d+\b[^.!?\n]{0,220}[.!?]",
    ),
]

ADVANTAGE_PATTERNS = [
    (
        "Explicit advantage",
        r"\b(?:advantage|benefit|benefits|strength|"
        r"cost-effective|cost effective|low-cost|low cost|"
        r"user-friendly|easy(?: to)? implement|reliable|"
        r"scalable|efficient|redundancy|robust)\b[^.!?\n]{0,220}[.!?]",
    ),
]

LIMITATION_PATTERNS = [
    (
        "Explicit limitation",
        r"\b(?:limitation|limitations|drawback|drawbacks|"
        r"disadvantage|disadvantages|challenge|challenges|"
        r"constraint|constraints|restricted|limited|"
        r"dependency|depends on|requires|maintenance|"
        r"fault|failure|unavailable|future work|future scope)\b"
        r"[^.!?\n]{0,260}[.!?]",
    ),
]

APPLICATION_PATTERNS = [
    (
        "Application",
        r"\b(?:application|applications|use case|use cases|"
        r"used for|useful for|designed for|deployed for|"
        r"fleet management|vehicle tracking|bus tracking|"
        r"ETA|estimated time of arrival|monitoring|"
        r"smart transportation|public transportation)\b"
        r"[^.!?\n]{0,220}[.!?]",
    ),
]


def _scan_with_patterns(
    *,
    text: str,
    patterns: list[tuple[str, str]],
    field: str,
    add_match: Callable[[str, str, int, int], None],
) -> None:
    for _label, match in _match_patterns(text, patterns):
        context = _sentence_context(text, match.start(), match.end())
        if not context:
            continue

        add_match(
            field,
            context,
            match.start(),
            match.end(),
        )


def _scan_paper(
    filename: str,
    document_id: str,
    storage_path: str,
) -> dict[str, list[dict[str, Any]]]:
    fields = {field: [] for field in FIELDS}
    seen = {field: set() for field in FIELDS}

    pages = extract_text_from_pdf(Path(storage_path))

    for page_data in pages:
        page = int(page_data["page"])
        text = _normalize(page_data.get("text", ""))[:MAX_PAGE_CHARS]

        if not text:
            continue

        def add_match(
            field: str,
            value: str,
            start: int,
            end: int,
        ) -> None:
            _add(
                fields=fields,
                seen=seen,
                field=field,
                value=value,
                evidence_text=_sentence_context(text, start, end),
                filename=filename,
                document_id=document_id,
                page=page,
            )

        _scan_with_patterns(
            text=text,
            patterns=METHOD_PATTERNS,
            field="methodology",
            add_match=add_match,
        )
        _scan_with_patterns(
            text=text,
            patterns=DATASET_PATTERNS,
            field="datasets",
            add_match=add_match,
        )
        _scan_with_patterns(
            text=text,
            patterns=METRIC_PATTERNS,
            field="evaluation_metrics",
            add_match=add_match,
        )
        _scan_with_patterns(
            text=text,
            patterns=RESULT_PATTERNS,
            field="accuracy_results",
            add_match=add_match,
        )
        _scan_with_patterns(
            text=text,
            patterns=ADVANTAGE_PATTERNS,
            field="advantages",
            add_match=add_match,
        )
        _scan_with_patterns(
            text=text,
            patterns=LIMITATION_PATTERNS,
            field="limitations",
            add_match=add_match,
        )
        _scan_with_patterns(
            text=text,
            patterns=APPLICATION_PATTERNS,
            field="applications",
            add_match=add_match,
        )

    return fields


def build_paper_matrix(
    project_id: str,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []

    for index, paper in enumerate(papers, start=1):
        extracted = _scan_paper(
            filename=paper["filename"],
            document_id=paper["id"],
            storage_path=paper["storage_path"],
        )

        row: dict[str, Any] = {
            "serial_number": index,
            "paper_title": paper["filename"],
            "document_id": paper["id"],
            "fields": {},
        }

        for field in FIELDS:
            entries = extracted[field]
            row[field] = (
                "; ".join(item["value"] for item in entries)
                if entries
                else NOT_IDENTIFIED
            )
            row["fields"][field] = {
                "reported": bool(entries),
                "entries": entries,
            }

        rows.append(row)

    return {
        "success": True,
        "project_id": project_id,
        "extraction_mode": "page_aware_targeted_scan",
        "columns": [
            {"key": key, "label": label}
            for key, label in MATRIX_COLUMNS
        ],
        "rows": rows,
        "paper_count": len(rows),
        "note": (
            "Values are candidate findings extracted directly from the "
            "original project PDFs. Findings are presented as complete "
            "source sentences; the original passage and page remain "
            "available for verification. A populated cell does not mean "
            "the field was scientifically validated. Empty fields are "
            "shown as 'Not identified in extracted evidence'."
        ),
    }


def paper_matrix_to_csv(matrix: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(label for _, label in MATRIX_COLUMNS)

    for row in matrix["rows"]:
        writer.writerow(row.get(key, "") for key, _ in MATRIX_COLUMNS)

    return output.getvalue()
