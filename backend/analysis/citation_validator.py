
"""Structural citation validation for ResearchLens research answers.

Validates exact document/page references supplied to answer generation.
This is a provenance check, not a semantic entailment check.
"""

import re


CITATION_LIKE = re.compile(
    r"\[[^\[\]\n]*\bPage\b[^\[\]\n]*\]",
    re.IGNORECASE,
)

SINGLE_CITATION = re.compile(
    r"^\[(.+?),\s*Page\s+(\d+)\]$",
    re.IGNORECASE,
)

MARKDOWN_HEADING = re.compile(
    r"^\s{0,3}#{1,6}\s+",
)

MARKDOWN_SEPARATOR = re.compile(
    r"^\s*[-*_]{3,}\s*$",
)

MARKDOWN_LIST_MARKER = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+",
)


def _parse_citation(citation):
    """Accept exactly one document and one page per citation."""
    match = SINGLE_CITATION.fullmatch(citation.strip())

    if not match:
        return None

    document = match.group(1).strip()

    try:
        page = int(match.group(2))
    except (TypeError, ValueError):
        return None

    if not document or page < 1:
        return None

    # Reject compound citations even if their beginning
    # resembles a valid single-document reference.
    if re.search(
        r",\s*Page\s+\d+",
        document,
        flags=re.IGNORECASE,
    ):
        return None

    if ";" in document:
        return None

    return document, page


def _allowed_sources(source_passages):
    allowed = set()

    for item in source_passages or []:
        document = str(
            item.get("document") or ""
        ).strip()

        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue

        if document and page >= 1:
            allowed.add(
                (document.casefold(), page)
            )

    return allowed


def _substantive_sections(answer):
    """Find paragraphs and list items that may require citations."""
    sections = []

    for paragraph in re.split(
        r"\n\s*\n",
        answer or "",
    ):
        lines = []

        for line in paragraph.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if MARKDOWN_HEADING.match(stripped):
                continue

            if MARKDOWN_SEPARATOR.match(stripped):
                continue

            if MARKDOWN_LIST_MARKER.match(stripped):
                if lines:
                    sections.append(" ".join(lines))
                    lines = []

                sections.append(
                    MARKDOWN_LIST_MARKER.sub(
                        "",
                        stripped,
                    )
                )
            else:
                lines.append(stripped)

        if lines:
            sections.append(" ".join(lines))

    return sections


def validate_answer_citations(answer, source_passages):
    """Return a JSON-serializable structural citation report."""
    allowed = _allowed_sources(source_passages)

    citations = []
    malformed = []

    for match in CITATION_LIKE.finditer(answer or ""):
        citation = match.group(0)
        parsed = _parse_citation(citation)

        if parsed is None:
            malformed.append(citation)
            continue

        document, page = parsed

        citations.append({
            "citation": citation,
            "document": document,
            "page": page,
            "valid_source": (
                document.casefold(),
                page,
            ) in allowed,
        })

    invalid = [
        item
        for item in citations
        if not item["valid_source"]
    ]

    uncited = []

    for section in _substantive_sections(answer):
        without_citations = CITATION_LIKE.sub(
            "",
            section,
        ).strip()

        word_count = len(
            re.findall(
                r"\b[\w-]+\b",
                without_citations,
            )
        )

        if word_count < 12:
            continue

        section_citations = [
            _parse_citation(match.group(0))
            for match in CITATION_LIKE.finditer(section)
        ]

        valid_section_citation = any(
            parsed is not None
            and (
                parsed[0].casefold(),
                parsed[1],
            ) in allowed
            for parsed in section_citations
        )

        if not valid_section_citation:
            uncited.append(section[:240])

    if invalid or malformed:
        status = "invalid_citations"
    elif not citations or uncited:
        status = "review_required"
    else:
        status = "structurally_valid"

    return {
        "status": status,
        "checked_citation_count": len(citations),
        "valid_citation_count": (
            len(citations) - len(invalid)
        ),
        "invalid_citations": invalid,
        "malformed_citations": malformed,
        "uncited_passage_warnings": uncited,
        "note": (
            "Structural validation checks that each citation "
            "uses a supplied document and page. It does not "
            "prove that the passage supports the associated "
            "claim. Missing-citation warnings are heuristic."
        ),
    }