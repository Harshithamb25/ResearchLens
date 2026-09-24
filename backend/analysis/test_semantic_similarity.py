from backend.analysis.semantic_similarity import calculate_claim_similarity


def test_similar_claims_have_high_similarity():

    claim_a = "Deep learning improves intrusion detection"

    claim_b = (
        "Deep neural networks improve "
        "intrusion detection performance"
    )

    similarity = calculate_claim_similarity(
        claim_a,
        claim_b
    )

    print(f"\nSimilarity: {similarity}")

    assert similarity > 0.5


