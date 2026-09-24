from backend.analysis.evidence import Evidence
from backend.analysis.evidence_matrix import build_evidence_matrix


def test_build_evidence_matrix():

    evidence_items = [
        Evidence(
            paper="paperA.pdf",
            page=5,
            evidence_text="Evidence A",
            claim="Deep learning improves intrusion detection"
        ),
        Evidence(
            paper="paperB.pdf",
            page=8,
            evidence_text="Evidence B",
            claim=(
                "Deep neural networks improve "
                "intrusion detection performance"
            )
        ),
        Evidence(
            paper="paperC.pdf",
            page=11,
            evidence_text="Evidence C",
            claim="Traditional methods remain useful"
        )
    ]

    matrix = build_evidence_matrix(
        evidence_items,
        threshold=0.75
    )

    assert len(matrix) == 2

    assert len(matrix[0]["evidence"]) == 2
    assert len(matrix[1]["evidence"]) == 1


