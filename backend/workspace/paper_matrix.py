
"""Concise, source-linked paper-wise evidence matrix extraction."""

import csv
import io
import re
from pathlib import Path
from typing import Any

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

ALGORITHMS = {
    "MobileNetV2": r"\bmobile\s*net\s*v\s*2\b",
    "MobileNet": r"\bmobile\s*net\b",
    "ResNet": r"\bres\s*net\b",
    "EfficientNet": r"\befficient\s*net\b",
    "Random Forest": r"\brandom\s+forest\b",
    "XGBoost": r"\bxgboost\b",
    "SVM": r"\bSVM\b|\bsupport vector machine\b",
    "CNN": r"\bCNN\b|\bconvolutional neural network\b",
    "LSTM": r"\bLSTM\b",
    "BERT": r"\bBERT\b",
    "YOLO": r"\bYOLO\b",
    "Transfer learning": r"\btransfer learning\b",
}

METRICS = {
    "Validation accuracy": r"\bvalidation accuracy\b",
    "Test accuracy": r"\btest accuracy\b",
    "Training accuracy": r"\btraining accuracy\b",
    "F1-score": r"\bF1[- ]?score\b",
    "Precision": r"\bprecision\b",
    "Recall": r"\brecall\b",
    "Accuracy": r"\baccuracy\b",
    "AUC": r"\bAUC\b",
    "Sensitivity": r"\bsensitivity\b",
    "Specificity": r"\bspecificity\b",
    "RMSE": r"\bRMSE\b",
    "MAE": r"\bMAE\b",
    "mAP": r"\bmAP\b",
}

DATASET_PATTERNS = [
    (
        r"\b(?:selected |existing )?three[- ]class dataset\b",
        "Three-class dataset",
    ),
    (
        r"\b(?:plant image|plant species|botanical image)"
        r"\s+dataset\b",
        "Plant-image dataset",
    ),
]

RESULT_PATTERNS = [
    # Presentation-style wording: "100% REPORTED VALIDATION ACCURACY".
    r"\b\d+(?:\.\d+)?\s*%\s+"
    r"(?:reported |achieved |obtained )?"
    r"(?:validation |test |training )?"
    r"(?:accuracy|precision|recall|F1[- ]?score)\b",

    # Conventional wording: "validation accuracy of 100%".
    r"\b(?:validation |test |training )?"
    r"(?:accuracy|precision|recall|F1[- ]?score)"
    r"\s*(?:of|was|is|:|=|at|reached)?\s*"
    r"\d+(?:\.\d+)?\s*%",

    # "achieved 100% validation accuracy".
    r"\b(?:achieved|obtained|attained|reported)\s+"
    r"\d+(?:\.\d+)?\s*%\s+"
    r"(?:validation |test |training )?"
    r"(?:accuracy|precision|recall|F1[- ]?score)\b",
]

LIMITATION_PATTERNS = [
    r"\bdoes not establish performance\b",
    r"\bnot (?:yet )?deployed\b",
    r"\bnot a deployed\b",
    r"\blimited to\b",
    r"\blimited by\b",
    r"\bsmall sample size\b",
    r"\bclass imbalance\b",
    r"\boverfitting\b",
    r"\bdata scarcity\b",
    r"\brequires? further validation\b",
    r"\bfuture work\b",
]

ADVANTAGE_PATTERNS = [
    r"\breduced computational cost\b",
    r"\blower computational cost\b",
    r"\bimproved accuracy\b",
    r"\bfaster inference\b",
    r"\bcomputationally efficient\b",
    r"\breproducible model workflow\b",
]

APPLICATION_PATTERNS = [
    (
        r"\bplant species identification\b",
        "Plant species identification",
    ),
    (
        r"\bbotanical identification\b",
        "Botanical identification",
    ),
    (
        r"\bweed screening\b",
        "Weed screening",
    ),
    (
        r"\bfield monitoring\b",
        "Field monitoring",
    ),
    (
        r"\bcrop disease detection\b",
        "Crop disease detection",
    ),
]

MAX_EXCERPT = 230
MAX_ENTRIES = 5


def _normalize(text: str) -> str:
    """Clean extracted text without altering reported numbers."""
    text = text.replace("â¢", "•")
    text = text.replace("â€“", "–")
    text = text.replace("â€”", "—")
    return re.sub(r"\s+", " ", text).strip()


def _context(
    text: str,
    start: int,
    end: int,
) -> str:
    """Retain a short excerpt around an exact source match."""
    left = max(0, start - 65)
    right = min(len(text), end + 100)

    excerpt = text[left:right]

    if left:
        excerpt = "…" + excerpt

    if right < len(text):
        excerpt += "…"

    return excerpt[:MAX_EXCERPT + 1]


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

    if key in seen[field]:
        return

    if len(fields[field]) >= MAX_ENTRIES:
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


def _scan_paper(
    filename: str,
    document_id: str,
    storage_path: str,
) -> dict[str, list[dict[str, Any]]]:
    fields = {
        field: [] for field in FIELDS
    }

    seen = {
        field: set() for field in FIELDS
    }

    pages = extract_text_from_pdf(
        Path(storage_path)
    )

    for page_data in pages:
        page = int(page_data["page"])
        text = _normalize(
            page_data.get("text", "")
        )

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
                evidence_text=_context(
                    text, start, end
                ),
                filename=filename,
                document_id=document_id,
                page=page,
            )

        # Methodology: extract named algorithms, not
        # paragraphs that merely contain "model".
        for label, pattern in ALGORITHMS.items():
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                # MobileNetV2 is more informative than
                # the general MobileNet family name.
                if (
                    label == "MobileNet"
                    and re.search(
                        r"\bmobile\s*net\s*v\s*2\b",
                        text,
                        re.IGNORECASE,
                    )
                ):
                    continue

                add_match(
                    "methodology",
                    label,
                    match.start(),
                    match.end(),
                )

        # Dataset: extract a specific description
        # only when the source explicitly states it.
        for pattern, label in DATASET_PATTERNS:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    "datasets",
                    label,
                    match.start(),
                    match.end(),
                )

        # Metrics: extract metric names without
        # confusing them with their measured values.
        specific_accuracy_found = bool(
            re.search(
                r"\b(?:validation|test|training)"
                r"\s+accuracy\b",
                text,
                re.IGNORECASE,
            )
        )

        for label, pattern in METRICS.items():
            if (
                label == "Accuracy"
                and specific_accuracy_found
            ):
                continue

            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    "evaluation_metrics",
                    label,
                    match.start(),
                    match.end(),
                )

        # Results: recognize both number-first and
        # metric-first expressions.
        for pattern in RESULT_PATTERNS:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    "accuracy_results",
                    match.group(0),
                    match.start(),
                    match.end(),
                )

        # Explicit limitations. Keep the original
        # excerpt instead of inventing a summary.
        for pattern in LIMITATION_PATTERNS:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                excerpt = _context(
                    text,
                    match.start(),
                    match.end(),
                )

                add_match(
                    "limitations",
                    excerpt,
                    match.start(),
                    match.end(),
                )

        # Advantages require specific evidence,
        # not generic words such as "efficient".
        for pattern in ADVANTAGE_PATTERNS:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    "advantages",
                    match.group(0),
                    match.start(),
                    match.end(),
                )

        # Applications are labelled as candidate
        # applications. A mention is not proof that
        # the application has been deployed.
        for pattern, label in APPLICATION_PATTERNS:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    "applications",
                    label,
                    match.start(),
                    match.end(),
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
        "extraction_mode": "targeted_pdf_scan",
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
            "Values are automatically identified "
            "candidate findings, not verified "
            "scientific conclusions. Applications "
            "may be proposed rather than deployed. "
            "Check source passages and pages before "
            "using the matrix in a literature review."
        ),
    }


def paper_matrix_to_csv(
    matrix: dict[str, Any],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)

    writer.writerow(
        label
        for _, label in MATRIX_COLUMNS
    )

    for row in matrix["rows"]:
        writer.writerow(
            row.get(key, "")
            for key, _ in MATRIX_COLUMNS
        )

    return output.getvalue()