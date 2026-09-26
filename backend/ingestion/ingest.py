
"""
ResearchLens PDF ingestion.

Extracts page-numbered text, creates searchable chunks,
generates embeddings, and stores them in ChromaDB.
"""

from pathlib import Path

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.vector_store import add_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


def ingest_paper(pdf_path):
    """
    Extract, chunk, embed, and index one research PDF.

    Returns the document name and processing counts.
    """

    pdf_path = Path(pdf_path)
    document_name = pdf_path.name

    if not pdf_path.is_file():
        raise ValueError(
            f"PDF file not found: {document_name}"
        )

    print(f"\nProcessing: {document_name}")

    # 1. Extract text while preserving page numbers.
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        raise ValueError(
            "The PDF contains no readable pages."
        )

    # 2. Create page-aware searchable chunks.
    chunks = create_chunks(
        pages,
        document_name,
    )

    # Reject image-only or otherwise unreadable PDFs.
    if not chunks:
        raise ValueError(
            "This PDF contains no extractable text. "
            "Scanned PDFs are not yet supported."
        )

    # 3. Generate embeddings for the extracted chunks.
    embeddings = generate_embeddings(chunks)

    # 4. Store the chunks, vectors, and source metadata.
    add_chunks(
        chunks,
        embeddings,
    )

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    return {
        "document": document_name,
        "pages": len(pages),
        "chunks": len(chunks),
    }


def ingest_all_papers():
    """
    Discover and ingest every PDF in data/papers/.
    """

    if not PAPERS_DIR.exists():
        print("No PDF directory found.")
        return []

    pdf_files = sorted(
        path
        for path in PAPERS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print("No PDF files found.")
        return []

    results = []

    for pdf_path in pdf_files:
        result = ingest_paper(pdf_path)
        results.append(result)

    return results