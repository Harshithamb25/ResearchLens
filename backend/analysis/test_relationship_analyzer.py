import json

from backend.analysis.evidence import Evidence

import backend.analysis.relationship_analyzer as relationship_analyzer


MOCK_RESPONSE = {
    "relationships": [
        {
            "paper": "paperA.pdf",
            "page": 5,
            "relationship": "SUPPORT",
            "explanation": (
                "The evidence reports improved detection "
                "performance using deep learning."
            )
        },
        {
            "paper": "paperB.pdf",
            "page": 8,
            "relationship": "SUPPORT",
            "explanation": (
                "The evidence reports high detection "
                "performance using deep learning."
            )
        }
    ]
}


class MockResponse:
    """
    Simulates a Gemini API response.
    """

    text = json.dumps(MOCK_RESPONSE)


def mock_generate_content(*args, **kwargs):
    """
    Simulates Gemini's generate_content method.
    """

    return MockResponse()


def test_relationship_analyzer(monkeypatch):

    monkeypatch.setattr(
        relationship_analyzer.client.models,
        "generate_content",
        mock_generate_content
    )

    claim = (
        "Deep learning improves intrusion detection"
    )

    evidence_items = [

        Evidence(
            paper="paperA.pdf",
            page=5,
            evidence_text=(
                "The proposed deep learning model "
                "achieved improved detection performance."
            ),
            claim=claim,
            dataset="NSL-KDD",
            method="Deep neural network",
            metric="Accuracy",
            conditions="Test dataset"
        ),

        Evidence(
            paper="paperB.pdf",
            page=8,
            evidence_text=(
                "The deep learning approach achieved "
                "high detection performance."
            ),
            claim=claim,
            dataset="UNSW-NB15",
            method="Deep neural network",
            metric="Detection accuracy",
            conditions="Test dataset"
        )
    ]

    claim_group = {
        "claim": claim,
        "evidence": evidence_items
    }

    relationships = (
        relationship_analyzer.analyze_evidence_relationships(
            claim_group
        )
    )

    assert relationships

    assert len(relationships) == 2

    for relationship in relationships:

        assert relationship["paper"]

        assert relationship["page"]

        assert relationship["relationship"] in {
            "SUPPORT",
            "QUALIFY",
            "POTENTIAL_CONFLICT",
            "INSUFFICIENT_EVIDENCE"
        }

        assert relationship["explanation"]


def test_relationships_are_linked_to_evidence(
    monkeypatch
):

    monkeypatch.setattr(
        relationship_analyzer.client.models,
        "generate_content",
        mock_generate_content
    )

    claim = (
        "Deep learning improves intrusion detection"
    )

    evidence = Evidence(
        paper="paperA.pdf",
        page=5,
        evidence_text=(
            "Deep learning improved detection."
        ),
        claim=claim,
        dataset="NSL-KDD",
        method="Deep neural network",
        metric="Accuracy",
        conditions="Test dataset"
    )

    claim_group = {
        "claim": claim,
        "evidence": [evidence]
    }

    relationships = (
        relationship_analyzer.build_evidence_relationships(
            claim_group
        )
    )

    assert relationships

    relationship = relationships[0]

    assert relationship.claim == claim

    assert relationship.paper == "paperA.pdf"

    assert relationship.page == 5

    assert relationship.relationship == "SUPPORT"

    assert relationship.evidence is evidence

    assert (
        relationship.evidence.evidence_text
        == "Deep learning improved detection."
    )

    assert (
        relationship.evidence.dataset
        == "NSL-KDD"
    )

    assert (
        relationship.evidence.method
        == "Deep neural network"
    )

    assert (
        relationship.evidence.metric
        == "Accuracy"
    )

    assert (
        relationship.evidence.conditions
        == "Test dataset"
    )


