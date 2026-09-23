from pathlib import Path

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks
from retrieval.embeddings import generate_embeddings
from retrieval.vector_store import add_chunks


pdf_path = "../data/papers/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

document_name = Path(pdf_path).name

chunks = create_chunks(
    pages,
    document_name
)

embeddings = generate_embeddings(chunks)

add_chunks(
    chunks,
    embeddings
)

print("Total chunks stored:", len(chunks))
print("Vector database: ChromaDB")
print("Collection: research_papers")