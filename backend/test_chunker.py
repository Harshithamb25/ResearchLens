from pathlib import Path

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks


def test_chunker_preserves_metadata():
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "papers"
        / "sample.pdf"
    )

    pages = extract_text_from_pdf(str(pdf_path))

    document_name = pdf_path.name

    chunks = create_chunks(
        pages,
        document_name
    )

    assert pages
    assert chunks

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "document" in chunk
        assert "page" in chunk
        assert "text" in chunk

        assert chunk["document"] == "sample.pdf"
        assert chunk["page"] >= 1
        assert chunk["text"]