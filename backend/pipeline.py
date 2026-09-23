# This connects the query classifier to actual downstream routes, giving us one central orchestration point.

from analysis.query_router import classify_query


def process_query(question):
    """
    Determine the appropriate ResearchLens
    processing route for a user question.
    """

    query_type = classify_query(question)

    if query_type == "factual":
        route = "rag"

    elif query_type == "dataset":
        route = "dataset_analysis"

    elif query_type == "comparison":
        route = "comparison_analysis"

    elif query_type == "limitation":
        route = "limitation_analysis"

    elif query_type == "research_gap":
        route = "research_gap_analysis"

    else:
        route = "rag"

    return {
        "question": question,
        "query_type": query_type,
        "route": route
    }