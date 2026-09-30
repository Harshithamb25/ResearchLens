"""Evidence-grounded paper-wise matrix extraction.

The matrix is built directly from the original project PDFs. It uses
domain-agnostic field signals and preserves the exact source page for
every extracted candidate.
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
MAX_EXCERPT = 360
MAX_PAGE_CHARS = 14000


def _normalize(text: str) -> str:
    text = text.replace("â¢", "•")
    text = text.replace("â€“", "–")
    text = text.replace("â€”", "—")
    return re.sub(r"\s+", " ", text).strip()


def _sentence_context(text: str, start: int, end: int) -> str:
    left = max(0, text.rfind(".", 0, start) + 1)
    right_dot = text.find(".", end)

    if right_dot == -1:
        right = min(len(text), end + MAX_EXCERPT)
    else:
        right = min(len(text), right_dot + 1)

    excerpt = _normalize(text[left:right])

    if len(excerpt) < 80:
        left = max(0, start - 120)
        right = min(len(text), end + 220)
        excerpt = _normalize(text[left:right])

    return excerpt[:MAX_EXCERPT]


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
    value = _normalize(value)

    if not value:
        return

    key = value.casefold()

    if key in seen[field] or len(fields[field]) >= MAX_ENTRIES:
        return

    seen[field].add(key)

    fields[field].append({
        "value": value,
        "evidence_text": _normalize(evidence_text),
        "source": {
            "paper": filename,
            "document_id": document_id,
            "page": page,
        },
    })


def _match_patterns(
    text: str,
    patterns: list[tuple[str, str]],
) -> list[tuple[str, re.Match[str]]]:
    matches = []

    for label, pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            matches.append((label, match))

    matches.sort(
        key=lambda item: (
            item[1].start(),
            item[0],
        )
    )

    return matches


# Generic signals intentionally cover research/document domains
# without assuming a particular topic.
METHOD_PATTERNS = [
    (
        "Methodology",
        r"\b(?:we|our|this paper|the authors?)\s+"
        r"(?:propose|proposed|develop|developed|design|designed|"
        r"implement|implemented|use|used|adopt|adopted)\b"
        r"[^.!?\n]{0,180}",
    ),
    (
        "System / architecture",
        r"\b(?:system architecture|system design|"
        r"proposed architecture|implementation design|"
        r"experimental setup)\b[^.!?\n]{0,180}",
    ),
    (
        "Named technology",
        r"\b(?:RFID|BLE|Bluetooth Low Energy|GPS|GSM|GPRS|"
        r"Arduino(?: UNO)?|Raspberry Pi(?: Zero)?|"
        r"SIM808|SIM900[A-Z]?|ESP8266|ThingSpeak|"
        r"Random Forest|XGBoost|SVM|CNN|LSTM|BERT|"
        r"YOLO|Transformer|neural network|support vector machine)\b",
    ),
]


DATASET_PATTERNS = [
    (
        "Dataset",
        r"\b(?:dataset|data set)\b[^.!?\n]{0,220}",
    ),
    (
        "Collected data",
        r"\b(?:data|measurements?)\s+(?:were|was)\s+"
        r"(?:collected|recorded|gathered)\b[^.!?\n]{0,220}",
    ),
    (
        "Field-trial data",
        r"\b(?:field trial|field trials|pilot study|"
        r"real-world deployment|real-world use case)\b"
        r"[^.!?\n]{0,220}",
    ),
    (
        "Study population / sample",
        r"\b(?:sample|participants?|subjects?|observations?|cases?)\b"
        r"[^.!?\n]{0,180}",
    ),
]


METRIC_PATTERNS = [
    (
        "Accuracy",
        r"\b(?:accuracy|accurate|precision|recall|F1(?:[- ]score)?|"
        r"sensitivity|specificity|AUC|RMSE|MAE|mAP)\b"
        r"[^.!?\n]{0,120}",
    ),
    (
        "Detection / range",
        r"\b(?:detection|detect(?:ed|ion)?|range)\b"
        r"[^.!?\n]{0,120}"
        r"\b\d+(?:\.\d+)?\s*(?:m|km|cm|ms|s|%)\b",
    ),
    (
        "Delay / timing",
        r"\b(?:delay|latency|journey time|travel time|"
        r"on-time|early departures?|delayed departures?)\b"
        r"[^.!?\n]{0,140}",
    ),
    (
        "Cost",
        r"\b(?:cost|price|budget|expenditure)\b"
        r"[^.!?\n]{0,160}",
    ),
]


RESULT_PATTERNS = [
    (
        "Quantitative result",
        r"\b(?:accuracy|precision|recall|F1(?:[- ]score)?|"
        r"on-time|early|delayed|detection|latency|delay|"
        r"journey time|travel time|cost|total cost)\b"
        r"[^.!?\n]{0,180}"
        r"\b\d+(?:\.\d+)?\s*(?:%|m|km|cm|ms|s|BDT|USD|RM|"
        r"minutes?|hours?)\b",
    ),
    (
        "Observed result",
        r"\b(?:results?|performance|experiment(?:al)?|"
        r"observed|measured|show(?:s|ed)?)\b"
        r"[^.!?\n]{0,220}",
    ),
    (
        "Tabulated result",
        r"\b(?:table|figure)\s+\d+\b[^.!?\n]{0,180}",
    ),
]


ADVANTAGE_PATTERNS = [
    (
        "Explicit advantage",
        r"\b(?:advantage|benefit|benefits|strength|"
        r"cost-effective|cost effective|low-cost|low cost|"
        r"user-friendly|easy(?: to)? implement|reliable|"
        r"scalable|efficient|redundancy|robust)\b"
        r"[^.!?\n]{0,180}",
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
        r"[^.!?\n]{0,220}",
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
        r"[^.!?\n]{0,180}",
    ),
]


def _scan_with_patterns(
    *,
    text: str,
    patterns: list[tuple[str, str]],
    field: str,
    add_match: Callable[[str, str, int, int], None],
) -> None:
    for label, match in _match_patterns(text, patterns):
        excerpt = _sentence_context(
            text,
            match.start(),
            match.end(),
        )

        if label in {
            "Methodology",
            "System / architecture",
            "Dataset",
            "Collected data",
            "Field-trial data",
            "Study population / sample",
            "Application",
        }:
            value = _normalize(match.group(0))
        else:
            value = excerpt

        add_match(
            field,
            value,
            match.start(),
            match.end(),
        )


def _scan_paper(
    filename: str,
    document_id: str,
    storage_path: str,
) -> dict[str, list[dict[str, Any]]]:
    fields = {
        field: []
        for field in FIELDS
    }

    seen = {
        field: set()
        for field in FIELDS
    }

    pages = extract_text_from_pdf(
        Path(storage_path)
    )

    for page_data in pages:
        page = int(
            page_data["page"]
        )

        text = _normalize(
            page_data.get("text", "")
        )[:MAX_PAGE_CHARS]

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
                evidence_text=_sentence_context(
                    text,
                    start,
                    end,
                ),
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

    for index, paper in enumerate(
        papers,
        start=1,
    ):
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
                "; ".join(
                    item["value"]
                    for item in entries
                )
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
            {
                "key": key,
                "label": label,
            }
            for key, label in MATRIX_COLUMNS
        ],
        "rows": rows,
        "paper_count": len(rows),
        "note": (
            "Values are candidate findings extracted directly "
            "from the original project PDFs. A populated cell "
            "means source text matched a field-specific signal; "
            "it is not a claim that the field was scientifically "
            "validated. Empty fields are shown as 'Not identified "
            "in extracted evidence'. Inspect the linked passage "
            "before citing a finding."
        ),
    }


def paper_matrix_to_csv(
    matrix: dict[str, Any],
) -> str:
    output = io.StringIO(
        newline=""
    )

    writer = csv.writer(
        output
    )

    writer.writerow(
        label
        for _, label in MATRIX_COLUMNS
    )

    for row in matrix["rows"]:
        writer.writerow(
            row.get(
                key,
                ""
            )
            for key, _ in MATRIX_COLUMNS
        )

    return output.getvalue()