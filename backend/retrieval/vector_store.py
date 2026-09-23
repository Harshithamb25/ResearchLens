# Persistent ChromaDB vector store for research paper chunks.

import chromadb


# Create a persistent ChromaDB client.
client = chromadb.PersistentClient(
    path="../chroma_db"
)


# Create or reuse the research paper collection.
collection = client.get_or_create_collection(
    name="research_papers"
)


def add_chunks(chunks, embeddings):
    """
    Store research paper chunks, embeddings, and metadata.

    If the same document is ingested again, its existing
    chunks are removed before the new version is stored.
    """

    if not chunks:
        return

    # Identify the document being ingested.
    document_name = chunks[0]["document"]

    # Check whether this document already exists.
    existing = collection.get(
        where={
            "document": document_name
        }
    )

    # Remove the previous version of this document.
    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    # Create unique IDs using document name + chunk ID.
    ids = [
        f"{chunk['document']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    # Store the new chunks and their embeddings.
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        metadatas=[
            {
                "document": chunk["document"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]
    )