
"""Regression tests for compound research citations."""

import pytest

from backend.analysis.citation_validator import validate_answer_citations


SOURCES = [
    {"document": "paper2.pdf", "page": 1},
    {"document": "paper2.pdf", "page": 2},
    {"document": "sample.pdf", "page": 4},
    {"document": "sample.pdf", "page": 10},
]


@pytest.mark.parametrize(
    "citation, expected_count",
    [
        ("[paper2.pdf, Page 2]", 1),
        ("[paper2.pdf, Page 1, Page 2]", 2),
        ("[paper2.pdf, Page 2; sample.pdf, Page 4]", 2),
        (
            "[paper2.pdf, Page 1, Page 2; sample.pdf, Page 4]",
            3,
        ),
    ],
)
def test_valid_compound_citations(citation, expected_count):
    answer = (
        "The retrieved studies identify several research challenges "
        f"and deployment limitations {citation}."
    )

    result = validate_answer_citations(answer, SOURCES)

    assert result["status"] == "structurally_valid"
    assert result["checked_citation_count"] == expected_count
    assert result["valid_citation_count"] == expected_count
    assert result["invalid_citations"] == []
    assert result["malformed_citations"] == []


def test_invalid_page_inside_compound_citation():
    answer = (
        "The studies identify several research challenges and "
        "deployment limitations [paper2.pdf, Page 1, Page 99]."
    )

    result = validate_answer_citations(answer, SOURCES)

    assert result["status"] == "invalid_citations"
    assert result["checked_citation_count"] == 2
    assert result["valid_citation_count"] == 1
    assert result["invalid_citations"][0]["page"] == 99


def test_invalid_document_inside_compound_citation():
    answer = (
        "The studies identify several research challenges and "
        "deployment limitations "
        "[paper2.pdf, Page 2; unknown.pdf, Page 4]."
    )

    result = validate_answer_citations(answer, SOURCES)

    assert result["status"] == "invalid_citations"
    assert result["valid_citation_count"] == 1
    assert result["invalid_citations"][0]["document"] == "unknown.pdf"


@pytest.mark.parametrize(
    "citation",
    [
        "[paper2.pdf, Page X]",
        "[paper2.pdf, Page 2; sample.pdf, Page X]",
        "[paper2.pdf, Page 2, Page X]",
    ],
)
def test_malformed_compound_citations(citation):
    answer = (
        "The studies identify several research challenges and "
        f"deployment limitations {citation}."
    )

    result = validate_answer_citations(answer, SOURCES)

    assert result["status"] == "invalid_citations"
    assert citation in result["malformed_citations"]