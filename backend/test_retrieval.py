from retrieval.retriever import search


query = "What dataset was used in the research paper?"

results = search(query, top_k=5)

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

print("\nQuery:", query)

for i in range(len(documents)):
    print("\n-----------------------------")
    print("Result:", i + 1)
    print("Document:", metadatas[i]["document"])
    print("Page:", metadatas[i]["page"])
    print("Chunk ID:", metadatas[i]["chunk_id"])
    print("Distance:", distances[i])
    print("Text:", documents[i][:500])