from retrieval.retriever import search
from retrieval.reranker import rerank


question = "What datasets are discussed in the research paper?"


# Retrieve a larger candidate pool
results = search(
    question,
    top_k=10
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]


# Build structured evidence
evidence = []

for document, metadata in zip(documents, metadatas):
    evidence.append({
        "document": metadata["document"],
        "page": metadata["page"],
        "chunk_id": metadata["chunk_id"],
        "text": document
    })


# Rerank candidates
ranked_results = rerank(
    question,
    evidence,
    top_k=5
)


print("\n==============================")
print("RESEARCHLENS RERANKING")
print("==============================")

print("\nQuestion:")
print(question)

print("\nReranked Evidence:")

for i, item in enumerate(ranked_results):

    print("\n-----------------------------")
    print("Rank:", i + 1)
    print("Document:", item["document"])
    print("Page:", item["page"])
    print("Chunk:", item["chunk_id"])
    print("Reranker Score:", item["reranker_score"])
    print("Text:", item["text"][:500])