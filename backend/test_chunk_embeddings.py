from pathlib import Path

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks
from retrieval.embeddings import generate_embeddings


pdf_path = "../data/papers/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

document_name = Path(pdf_path).name

chunks = create_chunks(pages, document_name)

embeddings = generate_embeddings(chunks)

print("Total chunks:", len(chunks))
print("Total embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))

print("\nFirst chunk:")
print("Document:", chunks[0]["document"])
print("Page:", chunks[0]["page"])
print("Text:", chunks[0]["text"][:200])

print("\nFirst embedding:")
print(embeddings[0][:10])