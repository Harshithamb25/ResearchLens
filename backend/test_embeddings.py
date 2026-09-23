from retrieval.embeddings import generate_embedding


text = "Machine learning can be used to detect cybersecurity threats."

embedding = generate_embedding(text)

print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])