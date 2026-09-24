from backend.analysis.evidence import Evidence
from backend.analysis.claim_grouper import group_claims


def test_group_semantically_similar_claims():

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

    groups = group_claims(
        evidence_items,
        threshold=0.75
    )

    assert len(groups) == 2

    assert len(groups[0]["evidence"]) == 2
    assert len(groups[1]["evidence"]) == 1


