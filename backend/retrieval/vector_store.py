
"""Persistent ChromaDB vector store for research paper chunks."""

from pathlib import Path

import chromadb


# Always use the same database location, regardless of
# the terminal's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name="research_papers"
)


def add_chunks(chunks, embeddings):
    """
    Replace the indexed chunks for one paper.

    Each chunk retains its document name, page number
    and chunk ID for evidence traceability.
    """

    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings must match."
        )

    document_names = {chunk["document"] for chunk in chunks}

    if len(document_names) != 1:
        raise ValueError(
            "add_chunks expects chunks from exactly one document."
        )

    document_name = chunks[0]["document"]

    ids = [
        f"{chunk['document']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    # Delete the previous chunks for this document.
    existing = collection.get(
        where={"document": document_name}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Insert the newly extracted and embedded chunks.
    collection.add(
        ids=ids,
        embeddings=(
            embeddings.tolist()
            if hasattr(embeddings, "tolist")
            else embeddings
        ),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "document": chunk["document"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]
    )