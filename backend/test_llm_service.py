from generation.llm_service import generate_answer


question = "What datasets are discussed in the research paper?"

context = """
Document: sample.pdf
Page: 9

The datasets used and generated for research in malware classification
and network intrusion detection are detailed in this section. The section
discusses malware datasets, network intrusion detection datasets, and
dataset generation. MISP tools are also discussed as part of the available
resources for malware detection research.
"""


answer = generate_answer(
    question,
    context
)

print("\nQuestion:")
print(question)

print("\nResearchLens Answer:")
print(answer)