
import json

import pytest

from backend.analysis.evidence import Evidence
import backend.analysis.relationship_analyzer as relationship_analyzer


CLAIM = "Deep learning improves intrusion detection"


def make_evidence(chunk_id, text):
    return Evidence(
        paper="paperA.pdf",
        page=5,
        chunk_id=chunk_id,
        evidence_text=text,
        claim=CLAIM,
        dataset="NSL-KDD",
        method="Deep neural network",
        metric="Accuracy",
        conditions="Test dataset"
    )


def mock_gemini(monkeypatch, relationships):
    """Replace Gemini with a deterministic response."""

    class MockResponse:
        text = json.dumps({
            "relationships": relationships
        })

    def mock_generate_content(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        relationship_analyzer.client.models,
        "generate_content",
        mock_generate_content
    )


def test_relationship_analyzer(monkeypatch):
    evidence_items = [
        make_evidence(10, "The model improved accuracy."),
        make_evidence(11, "The improvement was dataset-specific.")
    ]

    mock_gemini(
        monkeypatch,
        [
            {
                "evidence_id": 1,
                "relationship": "SUPPORT",
                "explanation": "The passage reports improved accuracy."
            },
            {
                "evidence_id": 2,
                "relationship": "QUALIFY",
                "explanation": "The result is limited to one dataset."
            }
        ]
    )

    result = relationship_analyzer.analyze_evidence_relationships({
        "claim": CLAIM,
        "evidence": evidence_items
    })

    assert len(result) == 2
    assert result[0]["evidence_id"] == 1
    assert result[1]["evidence_id"] == 2
    assert result[0]["relationship"] == "SUPPORT"
    assert result[1]["relationship"] == "QUALIFY"


def test_relationships_link_to_exact_chunks(monkeypatch):
    """
    Two chunks from the same paper and page must
    receive their own correct classifications.
    """
    first = make_evidence(
        42,
        "The model achieved improved detection accuracy."
    )
    second = make_evidence(
        43,
        "Performance decreased on unseen datasets."
    )

    # Deliberately return the results in reverse order.
    mock_gemini(
        monkeypatch,
        [
            {
                "evidence_id": 2,
                "relationship": "QUALIFY",
                "explanation": "Generalization is limited."
            },
            {
                "evidence_id": 1,
                "relationship": "SUPPORT",
                "explanation": "Detection accuracy improved."
            }
        ]
    )

    relationships = (
        relationship_analyzer.build_evidence_relationships({
            "claim": CLAIM,
            "evidence": [first, second]
        })
    )

    assert len(relationships) == 2

    assert relationships[0].evidence is first
    assert relationships[0].evidence.chunk_id == 42
    assert relationships[0].relationship == "SUPPORT"

    assert relationships[1].evidence is second
    assert relationships[1].evidence.chunk_id == 43
    assert relationships[1].relationship == "QUALIFY"


@pytest.mark.parametrize(
    "invalid_relationships",
    [
        # Duplicate evidence ID
        [
            {
                "evidence_id": 1,
                "relationship": "SUPPORT",
                "explanation": "Supported."
            },
            {
                "evidence_id": 1,
                "relationship": "QUALIFY",
                "explanation": "Limited."
            }
        ],
        # Unknown evidence ID
        [
            {
                "evidence_id": 1,
                "relationship": "SUPPORT",
                "explanation": "Supported."
            },
            {
                "evidence_id": 99,
                "relationship": "QUALIFY",
                "explanation": "Limited."
            }
        ],
        # Invalid relationship category
        [
            {
                "evidence_id": 1,
                "relationship": "CONFIRMED",
                "explanation": "Supported."
            },
            {
                "evidence_id": 2,
                "relationship": "QUALIFY",
                "explanation": "Limited."
            }
        ]
    ]
)
def test_invalid_relationship_responses(
    monkeypatch,
    invalid_relationships
):
    mock_gemini(monkeypatch, invalid_relationships)

    # Avoid waiting during retry tests.
    monkeypatch.setattr(
        relationship_analyzer.time,
        "sleep",
        lambda seconds: None
    )

    evidence_items = [
        make_evidence(42, "First passage."),
        make_evidence(43, "Second passage.")
    ]

    with pytest.raises(ValueError):
        relationship_analyzer.analyze_evidence_relationships({
            "claim": CLAIM,
            "evidence": evidence_items
        })