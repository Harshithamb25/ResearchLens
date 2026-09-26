"""No external services are needed for these citation-validation tests."""

from backend.analysis.citation_validator import validate_answer_citations

SOURCES = [
    {"document": "sample.pdf", "page": 10, "text": "Dataset construction has challenges."},
    {"document": "paper2.pdf", "page": 2, "text": "Generalizability is limited."},
]


def test_valid_citations_from_both_papers():
    answer = (
        "Dataset construction presents challenges [sample.pdf, Page 10].\n\n"
        "Single-dataset evaluation limits generalizability [paper2.pdf, Page 2]."
    )
    result = validate_answer_citations(answer, SOURCES)
    assert result["status"] == "structurally_valid"
    assert result["valid_citation_count"] == 2


def test_fabricated_page_is_rejected():
    result = validate_answer_citations(
        "The model achieved perfect accuracy [paper2.pdf, Page 999].", SOURCES
    )
    assert result["status"] == "invalid_citations"
    assert result["invalid_citations"][0]["page"] == 999


def test_missing_citations_require_review():
    result = validate_answer_citations(
        "Researchers report several challenges with training datasets "
        "and deploying intrusion detection models at scale.", SOURCES
    )
    assert result["status"] == "review_required"
    assert result["uncited_passage_warnings"]


def test_malformed_citation_is_flagged():
    result = validate_answer_citations(
        "A claim appears here [sample.pdf Page 10].", SOURCES
    )
    assert result["status"] == "invalid_citations"
    assert result["malformed_citations"]


def test_empty_answer_requires_review():
    result = validate_answer_citations("", SOURCES)
    assert result["status"] == "review_required"
