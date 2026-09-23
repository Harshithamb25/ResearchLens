"""
Lightweight query router for ResearchLens.

Routes research questions to specialized analysis pipelines.
"""


def classify_query(question):
    """
    Classify a research question into a supported query type.

    Returns:
        str: Query type.
    """

    question_lower = question.lower().strip()

    # Research-gap questions
    gap_keywords = [
        "research gap",
        "research gaps",
        "future work",
        "future research",
        "open problem",
        "open problems",
        "what is missing",
        "what remains"
    ]

    if any(keyword in question_lower for keyword in gap_keywords):
        return "research_gap"

    # Limitation questions
    limitation_keywords = [
        "limitation",
        "limitations",
        "drawback",
        "drawbacks",
        "weakness",
        "weaknesses",
        "disadvantage",
        "disadvantages"
    ]

    if any(keyword in question_lower for keyword in limitation_keywords):
        return "limitation"

    # Comparison questions
    comparison_keywords = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "similarities",
        "similar",
        "versus",
        "vs",
        "better than"
    ]

    if any(keyword in question_lower for keyword in comparison_keywords):
        return "comparison"

    # Dataset-focused questions
    dataset_keywords = [
        "dataset",
        "datasets",
        "data used",
        "training data",
        "test data"
    ]

    if any(keyword in question_lower for keyword in dataset_keywords):
        return "dataset"

    # Default
    return "factual"