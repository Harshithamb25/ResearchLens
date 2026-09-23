from retrieval.retriever import search
from retrieval.reranker import rerank
from generation.llm_service import generate_answer


question = "What datasets are discussed in the research paper?"


# Stage 1: Semantic retrieval
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


# Stage 2: Cross-encoder reranking
ranked_evidence = rerank(
    question,
    evidence,
    top_k=5
)


# Stage 3: Grounded generation
answer = generate_answer(
    question,
    ranked_evidence
)


print("\n==============================")
print("RESEARCHLENS RAG")
print("==============================")

print("\nQuestion:")
print(question)


print("\nReranked Evidence:")

for i, item in enumerate(ranked_evidence):

    print("\n-----------------------------")
    print("Rank:", i + 1)
    print("Document:", item["document"])
    print("Page:", item["page"])
    print("Chunk:", item["chunk_id"])
    print("Reranker Score:", item["reranker_score"])
    print("Text:", item["text"][:300])


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)