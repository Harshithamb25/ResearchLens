from analysis.query_router import classify_query


test_questions = [
    "What dataset was used in the research paper?",
    "What are the limitations of the proposed approach?",
    "Compare the datasets used in both papers.",
    "What are the research gaps identified in these papers?",
    "How does the proposed intrusion detection system work?"
]


print("\n==============================")
print("RESEARCHLENS QUERY ROUTER")
print("==============================")

for question in test_questions:
    query_type = classify_query(question)

    print(f"\nQuestion: {question}")
    print(f"Route: {query_type}")