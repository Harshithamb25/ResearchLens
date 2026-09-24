from pathlib import Path

from backend.ingestion.pdf_loader import extract_text_from_pdf


def test_pdf_extraction():
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "papers"
        / "sample.pdf"
    )

    pages = extract_text_from_pdf(str(pdf_path))

    assert pages
    assert len(pages) > 0

    for page in pages:
        assert "page" in page
        assert "text" in page
        assert page["text"]