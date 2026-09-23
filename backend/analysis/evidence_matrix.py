from analysis.claim_grouper import group_claims


def build_evidence_matrix(evidence_items, threshold=0.75):
    """
    Build an evidence matrix by grouping semantically
    similar claims.

    Returns:
        [
            {
                "claim": representative claim,
                "evidence": [Evidence, Evidence, ...]
            },
            ...
        ]
    """

    return group_claims(
        evidence_items,
        threshold=threshold
    )