from pathlib import Path

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.vector_store import add_chunks, collection


def test_vector_store_adds_chunks():
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

    embeddings = generate_embeddings(chunks)

    add_chunks(
        chunks,
        embeddings
    )

    stored = collection.get(
        where={"document": document_name}
    )

    assert stored["ids"]
    assert len(stored["ids"]) == len(chunks)
    assert len(stored["documents"]) == len(chunks)
    assert len(stored["metadatas"]) == len(chunks)

    for metadata in stored["metadatas"]:
        assert metadata["document"] == document_name
        assert "page" in metadata
        assert "chunk_id" in metadata