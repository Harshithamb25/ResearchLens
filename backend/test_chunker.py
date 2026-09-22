# The test verifies the complete metadata we expect to carry into our vector database.

from pathlib import Path

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks


pdf_path = "../data/papers/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

document_name = Path(pdf_path).name

chunks = create_chunks(pages, document_name)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

for chunk in chunks[:5]:
    print("\n-------------------------")
    print("Chunk ID:", chunk["chunk_id"])
    print("Document:", chunk["document"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:300])