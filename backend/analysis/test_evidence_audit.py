from backend.analysis.evidence_audit import (
    audit_evidence_relationships
)


def make_relationship(
    paper,
    relationship
):
    return type(
        "MockRelationship",
        (),
        {
            "paper": paper,
            "relationship": relationship
        }
    )()


def test_evidence_audit_metrics():

    relationships = [
        make_relationship(
            "paper1.pdf",
            "SUPPORT"
        ),
        make_relationship(
            "paper1.pdf",
            "QUALIFY"
        ),
        make_relationship(
            "paper2.pdf",
            "SUPPORT"
        ),
        make_relationship(
            "paper3.pdf",
            "POTENTIAL_CONFLICT"
        )
    ]

    audit = audit_evidence_relationships(
        claim="Example research claim",
        relationships=relationships,
        total_evidence=4
    )

    assert audit.total_evidence == 4
    assert audit.analyzed_evidence == 4

    assert audit.supporting_evidence == 2
    assert audit.qualifying_evidence == 1
    assert audit.potential_conflicts == 1
    assert audit.insufficient_evidence == 0

    # paper1, paper2, paper3 = 3 distinct sources
    assert audit.source_count == 3

    # 3 unique sources / 4 analyzed evidence items × 100
    assert audit.source_diversity == 75.0

    # 4 analyzed / 4 retrieved × 100
    assert audit.evidence_coverage == 100.0

    # 1 unresolved item / 4 analyzed × 100
    assert audit.unresolved_rate == 25.0


def test_evidence_audit_with_no_evidence():

    audit = audit_evidence_relationships(
        claim="Unsupported claim",
        relationships=[],
        total_evidence=0
    )

    assert audit.total_evidence == 0
    assert audit.analyzed_evidence == 0

    assert audit.source_count == 0

    assert audit.evidence_coverage == 0.0
    assert audit.source_diversity == 0.0
    assert audit.unresolved_rate == 0.0