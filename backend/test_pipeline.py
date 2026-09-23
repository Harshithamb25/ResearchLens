from pipeline import process_query


questions = [
    "How does the proposed intrusion detection system work?",
    "What datasets are used?",
    "Compare the approaches used in both papers.",
    "What are the limitations?",
    "What research gaps remain?"
]


print("\n================================")
print("RESEARCHLENS QUERY PIPELINE")
print("================================")

for question in questions:

    result = process_query(question)

    print(f"\nQuestion: {result['question']}")
    print(f"Query type: {result['query_type']}")
    print(f"Selected route: {result['route']}")