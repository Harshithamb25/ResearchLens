from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank


question = "What datasets are used for intrusion detection?"

print("\n======================================")
print("RESEARCHLENS MULTI-DOCUMENT RETRIEVAL")
print("======================================")

# 1. Retrieve candidates from the vector database
results = search(
    question,
    top_k=10
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

evidence = []

for document, metadata in zip(documents, metadatas):
    evidence.append({
        "document": metadata["document"],
        "page": metadata["page"],
        "chunk_id": metadata["chunk_id"],
        "text": document
    })

# 2. Rerank the retrieved evidence
ranked_evidence = rerank(
    question,
    evidence,
    top_k=5
)

print("\nTop Retrieved Evidence:\n")

for rank, item in enumerate(ranked_evidence, start=1):
    print(f"Rank {rank}")
    print(f"Document: {item['document']}")
    print(f"Page: {item['page']}")
    print(f"Chunk: {item['chunk_id']}")
    print(f"Reranker score: {item['reranker_score']:.4f}")
    print(f"Text: {item['text'][:300]}")
    print("-" * 60)

print("\nDocuments represented in final evidence:")

documents_found = sorted(
    set(item["document"] for item in ranked_evidence)
)

for document in documents_found:
    print("-", document)
