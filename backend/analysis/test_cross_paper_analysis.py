from backend.analysis.evidence import Evidence
from backend.analysis.cross_paper_analysis import EvidenceRelationship


def test_evidence_relationship():

    evidence = Evidence(
        paper="paperA.pdf",
        page=5,
        evidence_text="Deep learning achieved high detection performance.",
        claim="Deep learning improves intrusion detection"
    )

    relationship = EvidenceRelationship(
        claim=evidence.claim,
        paper=evidence.paper,
        page=evidence.page,
        relationship="SUPPORT",
        explanation="The evidence reports improved detection performance.",
        evidence=evidence
    )

    assert relationship.claim == "Deep learning improves intrusion detection"
    assert relationship.paper == "paperA.pdf"
    assert relationship.page == 5
    assert relationship.relationship == "SUPPORT"
    assert relationship.evidence == evidence


