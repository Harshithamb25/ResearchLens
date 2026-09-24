from pathlib import Path

from backend.ingestion.pdf_loader import extract_text_from_pdf
from backend.ingestion.chunker import create_chunks
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.vector_store import add_chunks


PAPERS_DIR = Path("data/papers")

def ingest_paper(pdf_path):
    """
    Extract, chunk, embed, and store one research paper.
    """

    document_name = Path(pdf_path).name

    print(f"\nProcessing: {document_name}")

    # 1. Extract text while preserving page numbers
    pages = extract_text_from_pdf(pdf_path)

    # 2. Create metadata-aware chunks
    chunks = create_chunks(
        pages,
        document_name
    )

    # 3. Generate embeddings
    embeddings = generate_embeddings(chunks)

    # 4. Store chunks, embeddings, and metadata
    add_chunks(
        chunks,
        embeddings
    )

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    return {
        "document": document_name,
        "pages": len(pages),
        "chunks": len(chunks)
    }


def ingest_all_papers():
    """
    Discover and ingest every PDF in data/papers/.
    """

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return []

    results = []

    for pdf_path in pdf_files:
        result = ingest_paper(pdf_path)
        results.append(result)

    return results

