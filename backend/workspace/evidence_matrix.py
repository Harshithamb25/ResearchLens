
"""Build an evidence-grounded, paper-wise research matrix.

The matrix uses structured fields already returned by the
ResearchLens evidence pipeline. It does not invent values for
fields that the retrieved evidence has not explicitly identified.
"""

import csv
import io
import re
from pathlib import Path
from typing import Any


NOT_IDENTIFIED = "Not identified in retrieved evidence"

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

FIELD_ALIASES = {
    "methodology": [
        "method",
        "methodology",
        "algorithm",
        "algorithms",
        "model",
        "models",
    ],
    "datasets": [
        "dataset",
        "datasets",
        "dataset_name",
        "dataset_names",
    ],
    "evaluation_metrics": [
        "metric",
        "metrics",
        "evaluation_metric",
        "evaluation_metrics",
    ],
    "accuracy_results": [
        "accuracy",
        "accuracy_results",
        "result",
        "results",
        "performance",
        "performance_results",
        "reported_results",
    ],
    "advantages": [
        "advantage",
        "advantages",
        "strength",
        "strengths",
        "benefit",
        "benefits",
    ],
    "limitations": [
        "limitation",
        "limitations",
        "disadvantage",
        "disadvantages",
        "challenge",
        "challenges",
    ],
    "applications": [
        "application",
        "applications",
        "use_case",
        "use_cases",
    ],
}


def _filename(value: Any) -> str:
    """Normalize a source filename for project-safe matching."""
    if not isinstance(value, str):
        return ""

    return Path(
        value.replace("\\", "/")
    ).name.strip().casefold()


def _display_values(value: Any) -> list[str]:
    """Accept strings, numeric values and simple lists."""
    if value is None or isinstance(value, bool):
        return []

    if isinstance(value, (str, int, float)):
        text = re.sub(r"\s+", " ", str(value)).strip()

        if not text or text.casefold() in {
            "none",
            "null",
            "n/a",
            "not available",
            "not specified",
            "unknown",
        }:
            return []

        return [text]

    if isinstance(value, (list, tuple)):
        values: list[str] = []

        for item in value:
            values.extend(_display_values(item))

        return values

    return []


def _page(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if number >= 1 else None


def _source_name(item: dict[str, Any]) -> str:
    for key in ("paper", "paper_title", "document"):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def _collect_evidence(
    research_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect structured passages without relying on one result view."""
    analysis = research_result.get("result") or {}

    if not isinstance(analysis, dict):
        return []

    collected: list[dict[str, Any]] = []

    primary = analysis.get("evidence") or []

    if isinstance(primary, list):
        collected.extend(
            item for item in primary
            if isinstance(item, dict)
        )

    themes = analysis.get("themes") or []

    if isinstance(themes, list):
        for theme in themes:
            if not isinstance(theme, dict):
                continue

            passages = theme.get("evidence") or []

            if isinstance(passages, list):
                collected.extend(
                    item for item in passages
                    if isinstance(item, dict)
                )

    comparisons = analysis.get("theme_comparisons") or []

    if isinstance(comparisons, list):
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                continue

            summaries = (
                comparison.get("source_summaries") or []
            )

            if not isinstance(summaries, list):
                continue

            for summary in summaries:
                if not isinstance(summary, dict):
                    continue

                paper = summary.get("paper")
                findings = summary.get("findings") or []

                if not isinstance(findings, list):
                    continue

                for finding in findings:
                    if not isinstance(finding, dict):
                        continue

                    collected.append({
                        **finding,
                        "paper": paper,
                    })

    return collected


def _field_entries(
    items: list[dict[str, Any]],
    field: str,
    filename: str,
) -> list[dict[str, Any]]:
    """Return unique explicit values with their source references."""
    aliases = FIELD_ALIASES[field]
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        if _filename(_source_name(item)) != _filename(filename):
            continue

        page = _page(item.get("page"))

        for alias in aliases:
            for value in _display_values(item.get(alias)):
                key = value.casefold()

                if key in seen:
                    continue

                seen.add(key)

                entries.append({
                    "value": value,
                    "source": {
                        "paper": filename,
                        "page": page,
                        "chunk_id": item.get("chunk_id"),
                    },
                })

    return entries


def build_evidence_matrix(
    project_id: str,
    session_id: str,
    question: str,
    research_result: dict[str, Any],
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one row per indexed project paper."""
    evidence = _collect_evidence(research_result)

    rows: list[dict[str, Any]] = []

    for index, paper in enumerate(papers, start=1):
        filename = str(paper["filename"])

        row: dict[str, Any] = {
            "serial_number": index,
            "paper_title": filename,
            "document_id": paper["id"],
            "fields": {},
        }

        for field in FIELD_ALIASES:
            entries = _field_entries(
                evidence,
                field,
                filename,
            )

            row[field] = (
                "; ".join(
                    entry["value"] for entry in entries
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
        "session_id": session_id,
        "question": question,
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
            "Cells contain only explicitly structured values "
            "identified in the saved research evidence. "
            "An unidentified field does not mean that the "
            "original paper contains no information about it. "
            "Inspect the original PDF before treating the "
            "matrix as an exhaustive literature review."
        ),
    }


def matrix_to_csv(
    matrix: dict[str, Any],
) -> str:
    """Export the matrix using the exact requested headings."""
    output = io.StringIO(newline="")

    writer = csv.writer(output)

    writer.writerow(
        label for _, label in MATRIX_COLUMNS
    )

    for row in matrix["rows"]:
        writer.writerow(
            row.get(key, "")
            for key, _ in MATRIX_COLUMNS
        )

    return output.getvalue()