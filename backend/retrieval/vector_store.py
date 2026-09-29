
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


def add_project_chunks(chunks, embeddings, project_id, document_id):
    """
    Index one document within a research project.

    Uses unique document IDs to prevent filename collisions
    and stores project metadata for isolated retrieval.
    """
    if not project_id or not document_id:
        raise ValueError(
            "Both project_id and document_id are required."
        )

    if not chunks:
        raise ValueError("Cannot index an empty document.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings must match."
        )

    document_names = {
        chunk["document"] for chunk in chunks
    }

    if len(document_names) != 1:
        raise ValueError(
            "Expected chunks from exactly one document."
        )

    ids = [
        f"project:{project_id}:document:{document_id}:"
        f"chunk:{chunk['chunk_id']}"
        for chunk in chunks
    ]

    existing = collection.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        raise ValueError(
            "This document ID is already indexed."
        )

    collection.add(
        ids=ids,
        embeddings=(
            embeddings.tolist()
            if hasattr(embeddings, "tolist")
            else embeddings
        ),
        documents=[
            chunk["text"] for chunk in chunks
        ],
        metadatas=[
            {
                "project_id": project_id,
                "document_id": document_id,
                "document": chunk["document"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ],
    )


def remove_project_document(document_id):
    """Remove only the indexed chunks of one document."""
    if not document_id:
        raise ValueError("document_id is required.")

    existing = collection.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])