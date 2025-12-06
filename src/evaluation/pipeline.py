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
    judge_agreement_ensemble,
)
from src.models.refinement import iterative_refinement
from src.evaluation.scoring import combine_scores
from src.utils.filter_slides import filter_content_slides

#Evaluation pipeline for one lecture
def evaluate_summary(
    slide_path: str,
    initial_summary: str,
    human_reference: str,
    cfg_judge,
    cfg_refine,
    out_dir: str,
    target_words: int = 300,
    refine_iters: int = 3,
) -> Dict[str, Any]:


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

    # refinement internally handles filtering — we pass full slides
    refined = iterative_refinement(
        slides=slides_full,           
        initial_summary=initial_summary,
        cfg_judge=cfg_judge,
        cfg_refine=cfg_refine,
        iters=refine_iters,
        save_callback=save_callback,
    )

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
        runs=3
    )


    agree = judge_agreement_ensemble(
        human_reference,
        refined,
        cfg_judge,
        runs=3
    )


    #final score
    score = combine_scores(rubric, agree)

    #Save as json
    result = {
        "refined_summary": refined,
        "signals": signals,
        "rubric": rubric,
        "agreement": agree,
        "final_score_0to1": score,
        "lecture_title": slides_dict.get("lecture_title", "Unknown Lecture"),
    }

    write_json(os.path.join(out_dir, "result.json"), result)

    return result
