from backend.analysis.evidence import Evidence


def test_evidence_creation():
    evidence = Evidence(
        paper="paper2.pdf",
        page=12,
        evidence_text="The model achieved high detection performance.",
        claim="The model performs well for intrusion detection.",
        dataset="NSL-KDD",
        method="Deep neural network",
        metric="Accuracy",
        conditions="Experimental evaluation"
    )

    assert evidence.paper == "paper2.pdf"
    assert evidence.page == 12
    assert evidence.claim is not None
    assert evidence.dataset == "NSL-KDD"
