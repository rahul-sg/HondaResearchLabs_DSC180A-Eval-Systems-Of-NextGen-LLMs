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
