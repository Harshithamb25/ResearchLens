# chunk size = 1000 meaning: approx 1000 characters are placed in each chunk
# overlap = The next chunk repeats approximately 200 characters from the previous chunk.

from pathlib import Path


def create_chunks(pages, document_name, chunk_size=1000, overlap=200):
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
# output format 

            chunks.append({
                "chunk_id": chunk_id,
                "document": document_name,
                "page": page["page"],
                "text": chunk_text
            })

            chunk_id += 1
            start += chunk_size - overlap

    return chunks