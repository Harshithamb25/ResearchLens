from pathlib import Path

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks
from backend.retrieval.embeddings import generate_embeddings


def test_chunk_embeddings():
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "papers"
        / "sample.pdf"
    )

    pages = extract_text_from_pdf(
        str(pdf_path)
    )

    document_name = pdf_path.name

    chunks = create_chunks(
        pages,
        document_name
    )

    embeddings = generate_embeddings(
        chunks
    )

    assert pages
    assert chunks
    assert embeddings.size > 0

    assert len(embeddings) == len(chunks)

    embedding_dimension = embeddings.shape[1]

    assert embedding_dimension > 0

    for embedding in embeddings:
        assert len(embedding) == embedding_dimension