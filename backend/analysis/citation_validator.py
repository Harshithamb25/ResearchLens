
"""Conservative structural citation checks for generated research answers.

Checks whether cited document/page pairs were supplied to answer generation.
Does not establish whether a passage semantically supports a claim.
"""

import re


# Extract complete bracketed references that contain the word "Page".
CITATION_LIKE = re.compile(
    r"\[[^\[\]\n]*\bPage\b[^\[\]\n]*\]",
    re.IGNORECASE,
)

# Each document segment must begin with a document name and its first page.
DOCUMENT_SEGMENT = re.compile(
    r"^\s*(.+?),\s*Page\s+(\d+)",
    re.IGNORECASE,
)

# Additional pages may follow the first page of the same document.
ADDITIONAL_PAGE = re.compile(
    r"\s*,\s*Page\s+(\d+)",
    re.IGNORECASE,
)


def _parse_citation(citation):
    """Parse all document/page pairs in one bracketed citation.

    Examples:
        [paper.pdf, Page 2]
        [paper.pdf, Page 1, Page 2]
        [paper.pdf, Page 2; other.pdf, Page 4]

    Return None if any part is malformed.
    """
    content = citation[1:-1].strip()
    segments = content.split(";")
    references = []

    for segment in segments:
        match = DOCUMENT_SEGMENT.match(segment)

        if not match:
            return None

        document = match.group(1).strip()
        if not document:
            return None

        references.append((document, int(match.group(2))))
        position = match.end()

        while position < len(segment):
            remaining = segment[position:]

            if not remaining.strip():
                break

            page_match = ADDITIONAL_PAGE.match(remaining)
            if not page_match:
                return None

            references.append((document, int(page_match.group(1))))
            position += page_match.end()

    return references or None


def validate_answer_citations(answer, source_passages):
    """Return a JSON-serializable structural audit of the final answer.

    Each document/page pair is checked individually. Multiple references
    inside one pair of brackets are counted as separate citations.
    """
    allowed = {
        (
            str(item["document"]).strip().casefold(),
            str(int(item["page"])),
        )
        for item in source_passages
    }

    citations = []
    malformed = []

    for match in CITATION_LIKE.finditer(answer or ""):
        citation = match.group(0)
        references = _parse_citation(citation)

        if references is None:
            malformed.append(citation)
            continue

        for document, page in references:
            citations.append(
                {
                    "citation": citation,
                    "document": document,
                    "page": page,
                    "valid_source": (
                        document.casefold(),
                        str(page),
                    ) in allowed,
                }
            )

    invalid = [
        item for item in citations if not item["valid_source"]
    ]

    # Flag substantive paragraphs or bullets without a source reference.
    # This is a heuristic warning, not semantic claim verification.
    uncited = []

    for paragraph in re.split(r"\n\s*\n", answer or ""):
        cleaned = re.sub(
            r"(?m)^\s{0,3}#{1,6}\s+.*$",
            "",
            paragraph,
        ).strip()

        cleaned = re.sub(
            r"(?m)^\s*[-*_]{3,}\s*$",
            "",
            cleaned,
        ).strip()

        if not cleaned:
            continue

        if len(re.findall(r"\b[\w-]+\b", cleaned)) < 12:
            continue

        has_valid_format = any(
            _parse_citation(match.group(0)) is not None
            for match in CITATION_LIKE.finditer(cleaned)
        )

        if not has_valid_format:
            uncited.append(cleaned[:240])

    if invalid or malformed:
        status = "invalid_citations"
    elif not citations or uncited:
        status = "review_required"
    else:
        status = "structurally_valid"

    return {
        "status": status,
        "checked_citation_count": len(citations),
        "valid_citation_count": len(citations) - len(invalid),
        "invalid_citations": invalid,
        "malformed_citations": malformed,
        "uncited_passage_warnings": uncited,
        "note": (
            "Structural validation only. A valid document/page citation "
            "does not prove that its passage supports the associated claim. "
            "Uncited-passage warnings are heuristic."
        ),
    }