from analysis.semantic_similarity import calculate_claim_similarity


def group_claims(evidence_items, threshold=0.75):
    """
    Group evidence items whose claims are
    semantically similar.
    """

    groups = []

    for evidence in evidence_items:

        if not evidence.claim:
            continue

        assigned = False

        for group in groups:

            representative_claim = group["claim"]

            similarity = calculate_claim_similarity(
                evidence.claim,
                representative_claim
            )

            if similarity >= threshold:
                group["evidence"].append(evidence)
                assigned = True
                break

        if not assigned:
            groups.append({
                "claim": evidence.claim,
                "evidence": [evidence]
            })

    return groups