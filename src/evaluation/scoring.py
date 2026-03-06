from typing import Dict, Any

# Combine rubric and agreement scores into final scalar score
def combine_scores(
    rubric: Dict[str, Any],
    agreement: Dict[str, Any],
    weights: Dict[str, float] | None = None
) -> float:

    if weights is None:
        weights = {
            "coverage": 1,
            "faithfulness": 2,
            "organization": 1,
            "clarity": 1,
            "style": 1
        }

    # Denominator for normalizing to [0,1]
    denom = float(sum(weights.values()) * 5)   # each rubric is 1–5

    # Weighted rubric score
    weighted_rubric = (
        weights["coverage"]      * rubric.get("coverage", 0) +
        weights["faithfulness"]  * rubric.get("faithfulness", 0) +
        weights["organization"]  * rubric.get("organization", 0) +
        weights["clarity"]       * rubric.get("clarity", 0) +
        weights["style"]         * rubric.get("style", 0)
    ) / denom

    # Agreement score in [0,1]
    agree = agreement.get("agreement_1to5", 0)
    agree_normalized = max(0, min(5, agree)) / 5.0

    # Final combination = average
    final_score = float(0.5 * weighted_rubric + 0.5 * agree_normalized)

    return final_score


# Enhanced multi-layered evaluation combining domain-aware rubric, NLP metrics, and METEOR
def compute_comprehensive_score(
    rubric_result: Dict[str, Any],
    agreement_result: Dict[str, Any],
    meteor_score: float,
    rubric_weights: Dict[str, float] | None = None,
    layer_weights: Dict[str, float] | None = None
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation score combining:
    - Domain-aware rubric evaluation
    - NLP-based agreement metrics
    - METEOR semantic similarity

    Returns detailed scoring breakdown and final score.
    """

    if rubric_weights is None:
        # Higher weight on faithfulness for academic content
        rubric_weights = {
            "coverage": 1.0,
            "faithfulness": 2.5,  # Most important
            "organization": 1.0,
            "clarity": 1.0,
            "style": 0.5
        }

    if layer_weights is None:
        # Weights for the three evaluation layers
        layer_weights = {
            "domain_rubric": 0.6,    # Domain-aware rubric (most important)
            "nlp_agreement": 0.2,    # NLP agreement metrics
            "semantic_similarity": 0.2  # METEOR semantic similarity
        }

    # Layer 1: Domain-aware rubric score (weighted average of dimensions)
    denom = sum(rubric_weights.values()) * 5  # Max possible score
    rubric_score = sum(
        rubric_weights[dim] * rubric_result.get(dim, 0)
        for dim in rubric_weights.keys()
    ) / denom

    # Layer 2: NLP agreement score (normalized to 0-1)
    agreement_score = max(0, min(5, agreement_result.get("agreement_1to5", 0))) / 5.0

    # Layer 3: METEOR semantic similarity (already 0-1)
    semantic_score = meteor_score

    # Final weighted combination
    final_score = (
        layer_weights["domain_rubric"] * rubric_score +
        layer_weights["nlp_agreement"] * agreement_score +
        layer_weights["semantic_similarity"] * semantic_score
    )

    # Return comprehensive results
    return {
        "final_score": final_score,
        "layer_scores": {
            "domain_rubric": rubric_score,
            "nlp_agreement": agreement_score,
            "semantic_similarity": semantic_score
        },
        "layer_weights": layer_weights,
        "rubric_breakdown": {
            dim: rubric_result.get(dim, 0)
            for dim in ["coverage", "faithfulness", "organization", "clarity", "style"]
        },
        "detected_domain": rubric_result.get("detected_domain", "unknown"),
        "rubric_weights": rubric_weights
    }
