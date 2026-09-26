"""Central orchestration for ResearchLens research answers and evidence audits."""

from backend.analysis.query_router import classify_query
from backend.analysis.evidence_pipeline import run_evidence_pipeline
from backend.analysis.evidence_builder import evidence_to_answer_context
from backend.generation.llm_service import generate_answer
from backend.analysis.citation_validator import validate_answer_citations


def _analysis_message(status, cross_paper_evidence):
    if status == "no_evidence":
        return "No analyzable evidence was available for claim-level analysis."
    if status == "failed":
        return (
            "Evidence relationship analysis failed. Retrieved passages remain "
            "available, but their relationships are unverified."
        )
    if status == "partial":
        return "Evidence analysis is incomplete. Some claim groups could not be classified."
    if status == "completed":
        if cross_paper_evidence:
            return (
                "Claim-level analysis completed using evidence from multiple "
                "papers. Shared themes do not establish agreement."
            )
        return (
            "Claim-level analysis completed, but the selected evidence comes "
            "from one paper. Cross-paper agreement is not established."
        )
    raise ValueError(f"Unexpected analysis status: {status}")


def _comparison_source_context(evidence_result, main_context):
    """Include original passages used by thematic comparisons, including discovery-only ones."""
    comparisons = evidence_result.get("theme_comparisons") or []
    themes = evidence_result.get("themes") or []
    main_keys = {
        (item["document"], str(item["page"]), str(item.get("chunk_id")))
        for item in main_context
    }
    available = {}
    for theme in themes:
        for item in theme.get("evidence", []):
            key = (item.paper, str(item.page), str(item.chunk_id))
            if item.evidence_text and key not in available:
                available[key] = {
                    "document": item.paper,
                    "page": item.page,
                    "chunk_id": item.chunk_id,
                    "text": item.evidence_text,
                }
    extra = []
    for comparison in comparisons:
        for finding in comparison.get("findings", []):
            key = (
                finding.get("paper"),
                str(finding.get("page")),
                str(finding.get("chunk_id")),
            )
            if key in available and key not in main_keys:
                extra.append(available[key])
                main_keys.add(key)
    return extra


def process_query(question, retrieval_k=10, rerank_k=5, claim_threshold=0.75):
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    query_type = classify_query(question)
    evidence_result = run_evidence_pipeline(
        question=question,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        claim_threshold=claim_threshold,
    )
    analysis_status = evidence_result["status"]
    answer = None
    citation_validation = None

    if evidence_result["evidence"]:
        answer_context = evidence_to_answer_context(evidence_result["evidence"])
        comparisons = evidence_result.get("theme_comparisons") or []
        if comparisons:
            answer_context.extend(
                _comparison_source_context(evidence_result, answer_context)
            )
            answer = generate_answer(
                question,
                answer_context,
                theme_comparisons=comparisons,
                analysis_status=analysis_status,
            )
        else:
            answer = generate_answer(question, answer_context)
        if answer is not None:
            citation_validation = validate_answer_citations(answer, answer_context)

    return {
        "question": question,
        "query_type": query_type,
        "route": "evidence_audit",
        "answer": answer,
        "answer_generated": answer is not None,
        "citation_validation": citation_validation,
        "analysis_status": analysis_status,
        "analysis_message": _analysis_message(
            analysis_status, evidence_result["cross_paper_evidence"]
        ),
        "audit_completed": analysis_status == "completed",
        "failed_group_count": len(evidence_result["analysis_failures"]),
        "result": evidence_result,
    }
