# Instead of embedding one sentence at a time, this lets us convert all research-paper chunks into vectors in one operation.

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    return model.encode(text)


def generate_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings