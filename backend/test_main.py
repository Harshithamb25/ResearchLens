
"""
FastAPI endpoint tests for ResearchLens.

Verify that HTTP request success and evidence-audit
completion are reported independently.
"""

import pytest
from fastapi.testclient import TestClient

from backend import main


client = TestClient(main.app)


@pytest.mark.parametrize(
    "status, audit_completed",
    [
        ("completed", True),
        ("partial", False),
        ("failed", False),
        ("no_evidence", False),
    ]
)
def test_query_exposes_analysis_status(
    monkeypatch,
    status,
    audit_completed
):
    """The API must expose the correct audit status."""

    def mock_process_query(
        question,
        retrieval_k,
        rerank_k,
        claim_threshold
    ):
        return {
            "question": question,
            "query_type": "comparison",
            "route": "evidence_audit",
            "answer": (
                None
                if status == "no_evidence"
                else "Example evidence-grounded answer."
            ),
            "answer_generated": status != "no_evidence",
            "analysis_status": status,
            "analysis_message": "Example status message.",
            "audit_completed": audit_completed,
            "failed_group_count": (
                1 if status in ("partial", "failed") else 0
            ),
            "result": {
                "status": status,
                "evidence": [],
                "claim_groups": [],
                "relationships": [],
                "audits": [],
                "analysis_failures": []
            }
        }

    monkeypatch.setattr(
        main,
        "process_query",
        mock_process_query
    )

    response = client.post(
        "/query",
        json={
            "question": "Compare these research papers."
        }
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["analysis_status"] == status
    assert body["audit_completed"] is audit_completed

    assert body["data"]["analysis_status"] == status
    assert body["data"]["audit_completed"] is audit_completed


def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_query_handles_internal_failure(monkeypatch):
    """Internal exception details must not leak to users."""

    def failing_process_query(**kwargs):
        raise RuntimeError("PRIVATE_INTERNAL_ERROR")

    monkeypatch.setattr(
        main,
        "process_query",
        failing_process_query
    )

    response = client.post(
        "/query",
        json={
            "question": "Compare these research papers."
        }
    )

    assert response.status_code == 500
    assert "PRIVATE_INTERNAL_ERROR" not in response.text


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"