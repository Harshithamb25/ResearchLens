# This creates a persistent ChromaDB collection and gives us a reusable function for storing our chunks, embeddings, and citation metadata.

import chromadb


client = chromadb.PersistentClient(
    path="../chroma_db"
)

collection = client.get_or_create_collection(
    name="research_papers"
)


def add_chunks(chunks, embeddings):
    collection.add(
        ids=[str(chunk["chunk_id"]) for chunk in chunks],
        embeddings=embeddings.tolist(),
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