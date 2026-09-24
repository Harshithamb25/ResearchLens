
import pytest

from backend.analysis.evidence_audit import (
    audit_evidence_relationships
)


def make_relationship(paper, relationship):
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
        make_relationship("paper1.pdf", "SUPPORT"),
        make_relationship("paper1.pdf", "QUALIFY"),
        make_relationship("paper2.pdf", "SUPPORT"),
        make_relationship("paper3.pdf", "POTENTIAL_CONFLICT")
    ]

    audit = audit_evidence_relationships(
        claim="Example research claim",
        relationships=relationships,
        total_evidence=4
    )

    assert audit.total_evidence == 4
    assert audit.analyzed_evidence == 4
    assert audit.unanalyzed_evidence == 0

    assert audit.supporting_evidence == 2
    assert audit.qualifying_evidence == 1
    assert audit.potential_conflicts == 1
    assert audit.insufficient_evidence == 0

    assert audit.source_count == 3
    assert audit.evidence_coverage == 100.0
    assert audit.evidence_sufficiency == 100.0
    assert audit.source_diversity == 75.0
    assert audit.unresolved_rate == 25.0


def test_partial_evidence_coverage():
    """
    Five candidate passages, but only three
    received a classification.
    """
    relationships = [
        make_relationship("paper1.pdf", "SUPPORT"),
        make_relationship("paper2.pdf", "QUALIFY"),
        make_relationship(
            "paper3.pdf",
            "INSUFFICIENT_EVIDENCE"
        )
    ]

    audit = audit_evidence_relationships(
        claim="Partially analyzed claim",
        relationships=relationships,
        total_evidence=5
    )

    assert audit.total_evidence == 5
    assert audit.analyzed_evidence == 3
    assert audit.unanalyzed_evidence == 2

    assert audit.evidence_coverage == 60.0

    assert audit.evidence_sufficiency == pytest.approx(
        66.67,
        abs=0.01
    )

    assert audit.unresolved_rate == pytest.approx(
        33.33,
        abs=0.01
    )


def test_insufficient_evidence_still_counts_as_analyzed():
    relationships = [
        make_relationship(
            "paper1.pdf",
            "INSUFFICIENT_EVIDENCE"
        )
    ]

    audit = audit_evidence_relationships(
        claim="Unresolved claim",
        relationships=relationships,
        total_evidence=1
    )

    assert audit.analyzed_evidence == 1
    assert audit.unanalyzed_evidence == 0
    assert audit.evidence_coverage == 100.0
    assert audit.evidence_sufficiency == 0.0
    assert audit.unresolved_rate == 100.0


def test_evidence_audit_with_no_evidence():
    audit = audit_evidence_relationships(
        claim="Unsupported claim",
        relationships=[],
        total_evidence=0
    )

    assert audit.total_evidence == 0
    assert audit.analyzed_evidence == 0
    assert audit.unanalyzed_evidence == 0

    assert audit.source_count == 0
    assert audit.evidence_coverage == 0.0
    assert audit.evidence_sufficiency == 0.0
    assert audit.source_diversity == 0.0
    assert audit.unresolved_rate == 0.0


def test_audit_rejects_inconsistent_counts():
    relationships = [
        make_relationship("paper1.pdf", "SUPPORT"),
        make_relationship("paper2.pdf", "QUALIFY")
    ]

    with pytest.raises(ValueError):
        audit_evidence_relationships(
            claim="Invalid audit",
            relationships=relationships,
            total_evidence=1
        )


def test_audit_rejects_invalid_relationship():
    relationships = [
        make_relationship("paper1.pdf", "UNKNOWN")
    ]

    with pytest.raises(ValueError):
        audit_evidence_relationships(
            claim="Invalid relationship",
            relationships=relationships,
            total_evidence=1
        )