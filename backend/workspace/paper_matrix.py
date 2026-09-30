"""AI-assisted, evidence-grounded paper comparison matrix.

The table shows concise analytical paraphrases. The original PDF passage is
retained separately for verification. Accuracy is percentage-only.
"""

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from backend.generation.llm_service import MODEL_NAME, _get_client
from backend.ingestion.pdf_loader import extract_text_from_pdf
from google.genai import types


NOT_IDENTIFIED = "Not identified in extracted evidence"
NOT_REPORTED = "Not reported"

MATRIX_COLUMNS = [
    ("serial_number", "S. No."),
    ("paper_title", "Paper Title"),
    ("methodology", "Methodology / Algorithm"),
    ("datasets", "Dataset(s)"),
    ("evaluation_metrics", "Evaluation Metrics"),
    ("accuracy_percent", "Accuracy (%)"),
    ("accuracy_results", "Results / Findings"),
    ("advantages", "Advantages"),
    ("limitations", "Limitations"),
    ("applications", "Applications"),
]

FIELDS = [key for key, _ in MATRIX_COLUMNS
          if key not in {"serial_number", "paper_title"}]

MAX_FINDINGS = 3
MAX_SOURCE_CHARS = 70000
MAX_PAGE_CHARS = 10000
MAX_FINDING_WORDS = 32

FIELD_RULES = {
    "methodology": "Summarize the paper's actual proposed or implemented approach.",
    "datasets": (
        "Identify the actual data, field trial, sample, route, population, or "
        "observations used. Do not treat one example coordinate as a dataset."
    ),
    "evaluation_metrics": (
        "List evaluation measures actually used, excluding percentage accuracy. "
        "Examples: detection range, delay, journey time, cost, precision, recall, RMSE."
    ),
    "accuracy_percent": (
        "Return only an explicitly reported accuracy percentage. "
        "Return null for 'accurate', 'high accuracy', reliability, range, delay, cost, "
        "precision, recall, or any non-percentage metric."
    ),
    "accuracy_results": (
        "Summarize important reported results or observations, separate from accuracy."
    ),
    "advantages": "Summarize supported benefits of the paper's own approach.",
    "limitations": (
        "Summarize explicit limitations, dependencies, constraints, failures, "
        "maintenance needs, or clearly supported weaknesses of the paper's approach."
    ),
    "applications": "Summarize concrete intended or demonstrated application areas.",
}


def _normalize(value: Any) -> str:
    replacements = {
        "â¢": "•", "â€“": "–", "â€”": "—", "â€™": "'",
        "â€œ": '"', "â€": '"', "ﬁ": "fi", "ﬂ": "fl",
    }
    text = str(value or "")
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def _source_text(pages: list[dict[str, Any]]) -> str:
    chunks = []
    total = 0
    for item in pages:
        page = int(item.get("page", 0))
        text = _normalize(item.get("text", ""))[:MAX_PAGE_CHARS]
        if not text:
            continue
        chunk = f"\n--- PAGE {page} ---\n{text}"
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_SOURCE_CHARS:
            break
    return "".join(chunks)[:MAX_SOURCE_CHARS]


def _schema() -> dict[str, Any]:
    finding = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "Concise analytical paraphrase, normally 5-18 words. "
                    "Do not quote the PDF."
                ),
            },
            "page": {"type": "integer", "minimum": 1},
        },
        "required": ["summary", "page"],
        "additionalProperties": False,
    }
    findings = {
        "type": "array",
        "items": finding,
        "maxItems": MAX_FINDINGS,
    }
    return {
        "type": "object",
        "properties": {
            "paper_title": {
                "type": "string",
                "description": "Exact paper title printed inside the PDF.",
            },
            "title_page": {"type": "integer", "minimum": 1},
            "methodology": findings,
            "datasets": findings,
            "evaluation_metrics": findings,
            "accuracy_percent": {
                "type": ["number", "null"],
                "description": "Explicit percentage accuracy only; otherwise null.",
            },
            "accuracy_page": {
                "type": ["integer", "null"],
                "minimum": 1,
            },
            "accuracy_results": findings,
            "advantages": findings,
            "limitations": findings,
            "applications": findings,
        },
        "required": [
            "paper_title", "title_page", "methodology", "datasets",
            "evaluation_metrics", "accuracy_percent", "accuracy_page",
            "accuracy_results", "advantages", "limitations", "applications",
        ],
        "additionalProperties": False,
    }


def _prompt(filename: str, source: str) -> str:
    rules = "\n".join(f"- {key}: {value}" for key, value in FIELD_RULES.items())
    return f"""
You are ResearchLens, an academic paper-analysis component.

Analyze the uploaded PDF below and create a concise research comparison record.

The source is evidence, NOT text to reproduce.

STRICT OUTPUT RULES:
1. Paraphrase every matrix finding. Do not copy complete PDF sentences.
2. Do not reproduce long phrases from the PDF.
3. Use short, polished academic wording; each finding must be <= {MAX_FINDING_WORDS} words.
4. Every finding must start with a capital letter and be a complete thought.
5. Never include authors, affiliations, DOI, journal metadata, page headers,
   section headings, citation-only material, or unrelated prior-work claims.
6. Attribute results to this paper only when the source supports that.
7. If a field is not supported, return an empty array rather than guessing.
8. Dataset(s) means the actual data/study material used by this paper.
9. Accuracy (%) is ONLY an explicit numeric percentage labelled as accuracy.
   Never convert precision, recall, range, delay, cost, reliability, or another
   metric into accuracy.
10. Every finding needs the supporting PDF page number.

FIELD DEFINITIONS:
{rules}

SOURCE FILE: {filename}

SOURCE PDF:
{source}
"""


def _generate(filename: str, source: str) -> dict[str, Any]:
    response = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=_prompt(filename, source),
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=_schema(),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty matrix analysis.")
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Matrix analysis did not return an object.")
    return result


def _clean_summary(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = _normalize(value).strip(" -•·")
    if not text or text[0].islower() or len(text.split()) > MAX_FINDING_WORDS:
        return ""
    lowered = text.casefold()
    noise = (
        "doi", "corresponding author", "received", "accepted", "published",
        "issn", "copyright", "frontiersin.org", "journal of", "school of",
        "faculty of", "department of",
    )
    if any(token in lowered for token in noise):
        return ""
    return text


def _ngrams(source: str, size: int = 9) -> set[str]:
    words = re.findall(r"[a-z0-9%]+", source.casefold())
    return {" ".join(words[i:i + size])
            for i in range(max(0, len(words) - size + 1))}


def _too_verbatim(summary: str, source_ngrams: set[str]) -> bool:
    words = re.findall(r"[a-z0-9%]+", summary.casefold())
    return any(
        " ".join(words[i:i + 9]) in source_ngrams
        for i in range(max(0, len(words) - 8))
    )


def _page(value: Any, count: int) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if 1 <= page <= count else None


def _entries(
    analysis: dict[str, Any],
    field: str,
    filename: str,
    document_id: str,
    pages: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    raw = analysis.get(field) or []
    if not isinstance(raw, list):
        return []

    source_ngrams = _ngrams(source)
    result = []
    seen = set()

    for item in raw:
        if not isinstance(item, dict):
            continue

        summary = _clean_summary(item.get("summary"))
        page = _page(item.get("page"), len(pages))

        if not summary or page is None or _too_verbatim(summary, source_ngrams):
            continue

        key = summary.casefold()
        if key in seen:
            continue
        seen.add(key)

        result.append({
            "value": summary,
            "evidence_text": _normalize(pages[page - 1].get("text", "")),
            "source": {
                "paper": filename,
                "document_id": document_id,
                "page": page,
            },
        })

        if len(result) >= MAX_FINDINGS:
            break

    return result


def _accuracy(
    analysis: dict[str, Any],
    filename: str,
    document_id: str,
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = analysis.get("accuracy_percent")
    page = _page(analysis.get("accuracy_page"), len(pages))

    if raw is None or page is None:
        return []

    try:
        number = float(raw)
    except (TypeError, ValueError):
        return []

    if not 0 <= number <= 100:
        return []

    value = f"{int(number)}%" if number.is_integer() else f"{number:.2f}".rstrip("0").rstrip(".") + "%"
    return [{
        "value": value,
        "evidence_text": _normalize(pages[page - 1].get("text", "")),
        "source": {"paper": filename, "document_id": document_id, "page": page},
    }]


def _fallback_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").strip()


def _scan(
    filename: str,
    document_id: str,
    storage_path: str,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    pages = extract_text_from_pdf(Path(storage_path))
    if not pages:
        raise ValueError(f"No extractable text found in {filename}.")

    source = _source_text(pages)
    analysis = _generate(filename, source)

    title = _normalize(analysis.get("paper_title"))
    if not title or len(title.split()) > 30:
        title = _fallback_title(filename)

    fields = {field: [] for field in FIELDS}
    for field in FIELDS:
        if field == "accuracy_percent":
            fields[field] = _accuracy(analysis, filename, document_id, pages)
        else:
            fields[field] = _entries(
                analysis, field, filename, document_id, pages, source
            )

    return title, fields


def build_paper_matrix(
    project_id: str,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    errors = []

    for index, paper in enumerate(papers, start=1):
        filename = str(paper["filename"])
        document_id = str(paper["id"])

        try:
            title, extracted = _scan(
                filename, document_id, paper["storage_path"]
            )
        except Exception as error:
            errors.append(f"{filename}: {error}")
            title = _fallback_title(filename)
            extracted = {field: [] for field in FIELDS}

        row = {
            "serial_number": index,
            "paper_title": title,
            "document_id": document_id,
            "filename": filename,
            "fields": {},
        }

        for field in FIELDS:
            entries = extracted[field]
            if field == "accuracy_percent":
                value = entries[0]["value"] if entries else NOT_REPORTED
            else:
                value = (
                    " · ".join(entry["value"] for entry in entries)
                    if entries else NOT_IDENTIFIED
                )
            row[field] = value
            row["fields"][field] = {
                "reported": bool(entries),
                "entries": entries,
            }

        rows.append(row)

    return {
        "success": True,
        "project_id": project_id,
        "extraction_mode": "structured_ai_paper_analysis",
        "columns": [{"key": key, "label": label} for key, label in MATRIX_COLUMNS],
        "rows": rows,
        "paper_count": len(rows),
        "note": (
            "The matrix displays concise AI-generated paraphrases grounded in "
            "the uploaded PDFs. Original passages and page references remain "
            "available in the evidence inspector. Accuracy (%) contains only "
            "explicitly reported percentage accuracy."
        ),
        "errors": errors,
    }


def matrix_to_csv(matrix: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(label for _, label in MATRIX_COLUMNS)
    for row in matrix.get("rows", []):
        writer.writerow(row.get(key, "") for key, _ in MATRIX_COLUMNS)
    return output.getvalue()
