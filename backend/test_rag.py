from retrieval.retriever import search
from generation.llm_service import generate_answer


query = "What datasets are discussed in the research paper?"


# Step 1: Retrieve relevant evidence
results = search(query, top_k=2)
documents = results["documents"][0]
metadatas = results["metadatas"][0]


# Step 2: Build evidence context
context_parts = []

for i in range(len(documents)):
    document = metadatas[i]["document"]
    page = metadatas[i]["page"]
    text = documents[i]

    context_parts.append(
        f"[Source {i + 1} | {document} | Page {page}]\n{text}"
    )


context = "\n\n".join(context_parts)


# Step 3: Give only retrieved evidence to Gemini
prompt = f"""
You are ResearchLens, an evidence-grounded research assistant.

Answer the user's question using ONLY the evidence provided below.

If the evidence does not contain enough information to answer,
say:

"Insufficient evidence in the retrieved documents."

Do not invent facts.

For every important claim, mention the relevant paper and page.

User question:
{query}

Retrieved evidence:
{context}
"""


# Step 4: Generate grounded answer
answer = generate_answer(prompt)


print("\n==============================")
print("RESEARCHLENS ANSWER")
print("==============================\n")

print(answer)