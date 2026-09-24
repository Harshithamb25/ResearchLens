from backend.analysis.cross_paper_analysis import EvidenceRelationship
from backend.analysis.evidence import Evidence
from backend.analysis.evidence_audit import audit_evidence_relationships


def create_relationship(
    paper,
    page,
    relationship
):
    evidence = Evidence(
        paper=paper,
        page=page,
        evidence_text="Research evidence.",
        claim="Deep learning improves intrusion detection.",
        dataset="NSL-KDD",
        method="Deep neural network",
        metric="Accuracy",
        conditions="Test dataset"
    )

    return EvidenceRelationship(
        claim="Deep learning improves intrusion detection.",
        paper=paper,
        page=page,
        relationship=relationship,
        explanation="Evidence-based relationship.",
        evidence=evidence
    )


def test_evidence_audit():

    relationships = [
        create_relationship(
            "paperA.pdf",
            5,
            "SUPPORT"
        ),
        create_relationship(
            "paperB.pdf",
            8,
            "SUPPORT"
        ),
        create_relationship(
            "paperC.pdf",
            10,
            "QUALIFY"
        ),
        create_relationship(
            "paperD.pdf",
            12,
            "POTENTIAL_CONFLICT"
        )
    ]

    audit = audit_evidence_relationships(
        claim="Deep learning improves intrusion detection.",
        relationships=relationships,
        total_evidence=5
    )

    assert audit.claim == (
        "Deep learning improves intrusion detection."
    )

    assert audit.total_evidence == 5

    assert audit.analyzed_evidence == 4

    assert audit.supporting_evidence == 2

    assert audit.qualifying_evidence == 1

    assert audit.potential_conflicts == 1

    assert audit.insufficient_evidence == 0

    assert audit.source_count == 4

    assert audit.evidence_coverage == 80.0

    assert audit.unresolved_rate == 25.0


def test_evidence_audit_with_insufficient_evidence():

    relationships = [
        create_relationship(
            "paperA.pdf",
            5,
            "INSUFFICIENT_EVIDENCE"
        )
    ]

    audit = audit_evidence_relationships(
        claim="Example research claim.",
        relationships=relationships,
        total_evidence=2
    )

    assert audit.total_evidence == 2

    assert audit.analyzed_evidence == 1

    assert audit.insufficient_evidence == 1

    assert audit.evidence_coverage == 50.0

    assert audit.unresolved_rate == 100.0


