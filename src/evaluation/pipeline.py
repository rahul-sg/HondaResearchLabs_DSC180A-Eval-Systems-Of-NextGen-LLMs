import os
from typing import Dict, Any
from pathlib import Path

from src.utils.io import (
    load_slides,
    write_iteration_summary,
    write_final_summary,
    write_json,
)
from src.utils.signals import compute_signals
from src.models.judge import (
    judge_rubric_ensemble,
)
from src.models.refinement import (
    iterative_refinement,
    iterative_refinement_lever_based,
)
from src.evaluation.scoring import combine_scores, compute_comprehensive_score
from src.utils.filter_slides import filter_content_slides


# Evaluation pipeline for one lecture
def evaluate_summary(
    slide_path: str,
    initial_summary: str,
    human_reference: str | None,
    cfg_judge,
    cfg_refine,
    out_dir: str,
    target_words: int = 350,
    use_lever_based: bool = True,
    min_avg_score: float = 4.5,
    min_change_threshold: float = 0.03,
    max_iterations: int = 12,
    min_iterations: int = 4,
    min_agreement: float = 0.7,
    use_pairwise_selection: bool = True,
    scoring_policy: str = "tuned",
    hallucination_damping_alpha: float | None = None,
    hallucination_subtractive_beta: float | None = None,
) -> Dict[str, Any]:
    """
    Evaluate and refine a summary with optional lever-based iterative refinement.
    
    Args:
        slide_path: Path to lecture slides PDF
        initial_summary: Starting summary text
        human_reference: Optional legacy reference summary (unused by default reference-free mode)
        cfg_judge: LLM config for judge
        cfg_refine: LLM config for refiner
        out_dir: Output directory for results
        target_words: Target summary length (default 350)
        use_lever_based: If True, use intelligent lever-based refinement;
                        if False, use fixed 3-iteration refinement (default True)
        min_avg_score: Stopping criterion - avg rubric score (default 4.5/5)
        min_change_threshold: Stopping criterion - convergence threshold (default 0.03)
        max_iterations: Safety limit on iterations (default 10)
        min_iterations: Minimum number of iterations to run (default 2)
        min_agreement: Legacy/compat parameter retained for CLI compatibility
        use_pairwise_selection: Whether to use pairwise judge selection at each
                       refinement step (default True)
        scoring_policy: Scoring profile. "tuned" uses lighter single-penalty,
                "legacy" preserves previous dual-penalty behavior.
        hallucination_damping_alpha: Optional override for multiplicative damping
                coefficient alpha in final score term (1 - alpha * H).
        hallucination_subtractive_beta: Optional override for subtractive penalty
                coefficient beta in manual score term beta * (2**H).
    
    Returns:
        Dict with refined_summary, signals, rubric, agreement, final_score, etc.
    """

    slides_dict = load_slides(slide_path)
    slides_full = slides_dict["slides"]            # raw slides
    slides_content = filter_content_slides(slides_full)  # Goal B applied

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    prev_summary = initial_summary

    def save_callback(iter_idx: int, summary_text: str):
        nonlocal prev_summary

        # Prevent blank summaries from propagating
        if not summary_text.strip():
            summary_text = prev_summary

        write_iteration_summary(out_dir, iter_idx, summary_text)
        prev_summary = summary_text

    # ---- CHOOSE REFINEMENT STRATEGY ----
    if use_lever_based:
        print("\n[PIPELINE] Using LEVER-BASED iterative refinement...")
        refined, refinement_metadata = iterative_refinement_lever_based(
            slides=slides_full,
            initial_summary=initial_summary,
            cfg_judge=cfg_judge,
            cfg_refine=cfg_refine,
            save_callback=save_callback,
            target_words=target_words,
            min_avg_score=min_avg_score,
            min_change_threshold=min_change_threshold,
            max_iterations=max_iterations,
            min_iterations=min_iterations,
            min_agreement=min_agreement,
            use_pairwise=use_pairwise_selection,
            select_best_recent_k=3,
        )
        print(f"[PIPELINE] Lever-based refinement complete. {refinement_metadata['stopping_reason']}")
    else:
        print("\n[PIPELINE] Using FIXED 3-iteration refinement (legacy)...")
        refined = iterative_refinement(
            slides=slides_full,
            initial_summary=initial_summary,
            cfg_judge=cfg_judge,
            cfg_refine=cfg_refine,
            iters=3,
            save_callback=save_callback,
            use_pairwise=use_pairwise_selection,
        )
        refinement_metadata = {
            "iterations_completed": 3,
            "stopping_reason": "Fixed 3 iterations (legacy mode)",
            "pairwise_selection_enabled": use_pairwise_selection,
        }

    if not refined.strip():
        refined = prev_summary

    # Save final result
    write_final_summary(out_dir, refined)

    signals = compute_signals(
        slides_content,        # not full slides
        refined,
        target_words=target_words
    )

    rubric = judge_rubric_ensemble(
        slides_full,           # judges see full lecture
        refined,
        cfg_judge,
        runs=3,
        use_domain_aware=True  # Enable domain-aware evaluation
    )

    # Reference-free default: agreement/METEOR disabled in main scoring path.
    # Preserve schema compatibility with a lightweight placeholder object.
    agree = {
        "used": False,
        "agreement_1to5": 0,
        "missing_key_points": [],
        "added_inaccuracies": [],
    }

    # ---- COMPREHENSIVE MULTI-LAYERED SCORING ----
    # Reference-free comprehensive scoring (domain-aware rubric only)
    comprehensive_result = compute_comprehensive_score(
        rubric_result=rubric,
        agreement_result=None,
        meteor_score=None,
    )

    detected_domain = comprehensive_result.get("detected_domain", "humanities")

    if scoring_policy not in {"legacy", "tuned"}:
        scoring_policy = "tuned"

    if scoring_policy == "legacy":
        default_alpha = 0.15
        default_beta = 0.10
    else:
        default_alpha = 0.05
        default_beta = 0.0

    alpha = default_alpha if hallucination_damping_alpha is None else hallucination_damping_alpha
    beta = default_beta if hallucination_subtractive_beta is None else hallucination_subtractive_beta

    domain_alpha_multiplier = {
        "engineering": 1.15,
        "math": 1.10,
        "natural_sciences": 1.05,
        "social_sciences": 1.00,
        "business": 0.95,
        "humanities": 0.90,
    }.get(detected_domain, 1.0)

    effective_alpha = alpha * domain_alpha_multiplier

    # ---- MANUAL WEIGHTED SCORING (explicit baseline) ----
    base_score = combine_scores(rubric, agreement=None)
    manual_weighted_score = (
        (0.8 * base_score
         + 0.2 * signals["section_coverage_pct"])
        - beta * (2 ** signals["suspected_hallucination_rate"])
    )

    # ---- BALANCED HYBRID FINAL SCORING (recommended) ----
    # S = 0.7*C + 0.3*M
    # S_final = S * (1 - 0.15*H)
    comprehensive_score = comprehensive_result["final_score"]
    blended_score = 0.7 * comprehensive_score + 0.3 * manual_weighted_score
    raw_damping = 1 - effective_alpha * signals["suspected_hallucination_rate"]
    min_damping_factor = 0.75
    max_damping_factor = 1.0
    hallucination_damping = max(min_damping_factor, min(max_damping_factor, raw_damping))
    raw_quality_score = blended_score
    risk_adjusted_score = blended_score * hallucination_damping
    final_score = risk_adjusted_score
    scorer_disagreement = abs(comprehensive_score - manual_weighted_score)

    # Save as json
    iteration_score_table = []
    for item in refinement_metadata.get("iteration_metrics", []):
        iteration_score_table.append(
            {
                "iteration": item.get("iteration"),
                "avg_rubric_score": item.get("avg_rubric_score"),
                "quality_score": item.get("quality_score"),
                "word_count": item.get("word_count"),
                "change_magnitude": item.get("change_magnitude"),
                "hallucination_rate": item.get("signals", {}).get("suspected_hallucination_rate"),
                "coverage": item.get("signals", {}).get("section_coverage_pct"),
                "glossary_recall": item.get("signals", {}).get("glossary_recall"),
            }
        )

    result = {
        "refined_summary": refined,
        "signals": signals,
        "rubric": rubric,
        "agreement": agree,
        "reference_metrics_used": False,
        "comprehensive_scoring": comprehensive_result,  # Detailed scoring breakdown
        "hybrid_scoring": {
            "comprehensive_score": comprehensive_score,
            "manual_weighted_score": manual_weighted_score,
            "blend_weights": {"comprehensive": 0.7, "manual": 0.3},
            "blended_score_pre_damping": blended_score,
            "hallucination_policy": {
                "scoring_policy": scoring_policy,
                "damping_alpha": alpha,
                "subtractive_beta": beta,
                "domain_alpha_multiplier": domain_alpha_multiplier,
                "effective_alpha": effective_alpha,
                "min_damping_factor": min_damping_factor,
                "max_damping_factor": max_damping_factor,
            },
            "hallucination_rate": signals["suspected_hallucination_rate"],
            "raw_damping_factor": raw_damping,
            "hallucination_damping_factor": hallucination_damping,
            "scorer_disagreement_delta": scorer_disagreement,
            "raw_quality_score": raw_quality_score,
            "risk_adjusted_score": risk_adjusted_score,
        },
        "leaderboard_scores": {
            "raw_quality_score": raw_quality_score,
            "risk_adjusted_score": risk_adjusted_score,
        },
        "iteration_score_table": iteration_score_table,
        "final_score_0to1": final_score,
        "pairwise_selection_enabled": use_pairwise_selection,
        "lecture_title": slides_dict.get("lecture_title", "Unknown Lecture"),
        "refinement_metadata": refinement_metadata,  # Add refinement details
    }

    write_json(os.path.join(out_dir, "result.json"), result)

    return result

