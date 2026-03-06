from typing import Dict, List, Callable, Tuple
import json

from src.models.llm_client import call_llm, LLMConfig
from src.models.judge import judge_rubric
from src.utils.chunking import slides_to_text
from src.utils.filter_slides import filter_content_slides
from src.utils.signals import compute_signals
from src.evaluation.pairwise import round_robin_pairwise
from src.models.lever_based_refinement import (
    LeverBasedRefinementController,
    RefinementState,
    compute_change_magnitude,
    create_lever_guided_refine_prompt,
)

# NLTK imports for meteor and tokenization
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

# Ensure required NLTK resources are available
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)



# Number of slides allowed for the REFINER model
REFINER_SLIDE_LIMIT = 9999   # Typically 8–15

# Toggle debug prints
DEBUG_REFINER = False


#Prompt for refinement model
REFINE_PROMPT = """
You are revising a student lecture summary using:
1) A **subset of the lecture slides** (content slides only)
2) A **judge's detailed feedback**

[Slides Provided]
{slides}

[Current Summary]
{summary}

[Judge Feedback]
{feedback}

TASK:
Rewrite the summary so that it:
- incorporates missing key ideas,
- fixes inaccuracies,
- improves clarity, structure, and flow,
- removes redundancy,
- stays close in length to the original (±20%),
- AND remains strictly grounded in the slides provided.

CRITICAL RULES:
- Use ONLY information from these slides.
- Do NOT include syllabus, instructor info, grading, or logistics.
- Do NOT reference topics not shown above.
- NO hallucinations.

Return ONLY the improved summary.
"""


# Limit slides for refinement model
def _limit_slides_for_refiner(slides: List[Dict]) -> List[Dict]:

    content_slides = filter_content_slides(slides)

    return slides
    #return content_slides[:REFINER_SLIDE_LIMIT]



# One step refinemnet
def refine_once(
    slides: List[Dict],
    summary: str,
    feedback: Dict,
    cfg_refine: LLMConfig,
    retry_limit: int = 2
) -> str:

    # Filter + truncate slides (Goal B + Option A)
    limited_slides = _limit_slides_for_refiner(slides)
    limited_slides_str = slides_to_text(limited_slides)

    # Prepare judge feedback
    feedback_text = json.dumps(feedback, indent=2)

    # Build prompt
    user_prompt = REFINE_PROMPT.format(
        slides=limited_slides_str,
        summary=summary,
        feedback=feedback_text
    )

    # Debug prints
    if DEBUG_REFINER:
        print("\n===== REFINER DEBUG =====")
        print("Slides used:", len(limited_slides))
        print("Slide text length:", len(limited_slides_str.split()))
        print("Summary length:", len(summary.split()))
        print("=========================\n")

    print("Calling REFINER... Summary length:", len(summary.split()))

    # Retry loop
    for attempt in range(retry_limit):
        raw_response = call_llm(
            system_prompt="You refine lecture summaries accurately and concisely.",
            user_prompt=user_prompt,
            cfg=cfg_refine,
            json_mode=False
        )

        try:
            print("RAW LLM RESPONSE (repr):", repr(raw_response))
        except UnicodeEncodeError:
            print("RAW LLM RESPONSE: [Unicode content that cannot be displayed]")

        refined = (raw_response or "").strip()

        if refined and len(refined.split()) >= 10:
            return refined

        print("   • Attempt:", attempt, " → output word count:", len(refined.split()))

    # Fallback: keep previous summary
    return summary

# Iterative refinement process
def iterative_refinement(
    slides: List[Dict],
    initial_summary: str,
    cfg_judge: LLMConfig,
    cfg_refine: LLMConfig,
    iters: int = 3,
    save_callback: Callable[[int, str], None] | None = None
) -> str:

    initial = initial_summary

    # Save initial summary
    if save_callback:
        save_callback(0, initial)

    S = initial

    for i in range(1, iters + 1):

        # 1. Judge all slides
        feedback = judge_rubric(slides, S, cfg_judge)

        # 2. Refiner uses content-only slides
        pairwise1 = S
        pairwise2 = refine_once(slides, S, feedback, cfg_refine)

        S = round_robin_pairwise(
            slides=slides,
            summaries={"prev": pairwise1, "refined": pairwise2},
            cfg_judge=cfg_judge,
            runs=3
        )["result_summary"]


        # 3. Save intermediate outputs
        if save_callback:
            save_callback(i, S)

    return S


# ============================================================================
# LEVER-BASED ITERATIVE REFINEMENT
# ============================================================================

def refine_once_with_guidance(
    slides: List[Dict],
    summary: str,
    judge_feedback: Dict,
    controller: LeverBasedRefinementController,
    state: RefinementState,
    cfg_refine: LLMConfig,
    retry_limit: int = 2
) -> str:
    """
    Refine summary using lever-based guidance in addition to judge feedback.
    """
    
    # Filter + truncate slides (Goal B + Option A)
    limited_slides = _limit_slides_for_refiner(slides)
    limited_slides_str = slides_to_text(limited_slides)

    # Build lever-guided prompt
    user_prompt = create_lever_guided_refine_prompt(
        slides_text=limited_slides_str,
        summary=summary,
        judge_feedback=judge_feedback,
        controller=controller,
        state=state,
    )

    if DEBUG_REFINER:
        print("\n===== LEVER-BASED REFINER DEBUG =====")
        print("Iteration:", state.iteration)
        print("Current avg rubric score:", state.avg_rubric_score)
        print("Word count:", state.word_count)
        print("Guidance:", controller.get_refinement_guidance(state))
        print("=====================================\n")

    print(f"[Iter {state.iteration}] Calling LEVER-BASED REFINER... Summary length: {len(summary.split())}")

    # Retry loop
    for attempt in range(retry_limit):
        raw_response = call_llm(
            system_prompt="You refine lecture summaries accurately, focusing on the provided lever-based guidance.",
            user_prompt=user_prompt,
            cfg=cfg_refine,
            json_mode=False
        )

        try:
            print(f"[Iter {state.iteration}] RAW LLM RESPONSE (repr):", repr(raw_response[:100]))
        except UnicodeEncodeError:
            print(f"[Iter {state.iteration}] RAW LLM RESPONSE: [Unicode content that cannot be displayed]")

        refined = (raw_response or "").strip()

        if refined and len(refined.split()) >= 10:
            return refined

        print(f"   • Attempt {attempt}: output word count: {len(refined.split())}")

    # Fallback: keep previous summary
    return summary


def iterative_refinement_lever_based(
    slides: List[Dict],
    initial_summary: str,
    cfg_judge: LLMConfig,
    cfg_refine: LLMConfig,
    save_callback: Callable[[int, str], None] | None = None,
    target_words: int = 350,
    min_avg_score: float = 4.0,
    min_change_threshold: float = 0.03,
    max_iterations: int = 12,
    min_iterations: int = 4,
    human_reference: str | None = None,
    min_agreement: float = 0.7,
) -> Tuple[str, Dict]:
    """
    Iterative refinement with lever-based guidance and domain-agnostic stopping.

    Refinement runs until a decision-table stop condition is reached
    (pass, borderline, stalled, convergence, or max-iterations), while
    always enforcing at least `min_iterations` iterations.
    
    Args:
        slides: List of slide dictionaries
        initial_summary: Starting summary text
        cfg_judge: LLM config for judge
        cfg_refine: LLM config for refiner
        save_callback: Optional callback(iteration, summary) to save intermediate results
        target_words: Target word count (default 350)
        min_avg_score: Minimum average rubric score to stop (default 4.0/5)
        min_change_threshold: Minimum change magnitude before stopping (default 0.03)
        max_iterations: Safety limit on iterations (default 12)
        min_iterations: Must run at least this many iterations (default 4)
        human_reference: Optional human summary used to compute agreement/meteor each iter
        min_agreement: Meteor score threshold (0..1) for quality+agreement stop
    
    Returns:
        (final_summary, refinement_metadata)
        where metadata includes:
        - iterations_completed
        - final_avg_score
        - final_word_count
        - stopping_reason
        - lever_history
        - final_rubric
        - final_agreement
    """

    # Initialize controller (pass agreement threshold too)
    controller = LeverBasedRefinementController(
        target_words=target_words,
        min_avg_score=min_avg_score,
        min_change_threshold=min_change_threshold,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        min_agreement=min_agreement,
    )

    # Initialize state
    prev_summary = initial_summary
    current_summary = initial_summary
    
    # Save initial summary
    if save_callback:
        save_callback(0, initial_summary)

    # Main loop
    iteration = 0
    all_lever_history = []
    all_signals_history = []
    all_quality_history = []
    stopping_reason = ""

    while iteration < max_iterations:
        iteration += 1
        
        # 1. Judge current summary
        print(f"\n[Iter {iteration}] Evaluating summary with judge...")
        judge_feedback = judge_rubric(slides, current_summary, cfg_judge)
        
        # Extract rubric scores
        rubric = {
            "coverage": judge_feedback.get("coverage", 3),
            "faithfulness": judge_feedback.get("faithfulness", 3),
            "organization": judge_feedback.get("organization", 3),
            "clarity": judge_feedback.get("clarity", 3),
            "style": judge_feedback.get("style", 3),
        }
        
        # Compute state metrics
        word_count = len(current_summary.split())
        avg_score = sum(rubric.values()) / len(rubric) if rubric else 0
        change_magnitude = compute_change_magnitude(prev_summary, current_summary)
        
        # compute agreement/meteor score if reference provided
        agreement = 0.0
        if human_reference:
            tokenized_ref = word_tokenize(human_reference)
            tokenized_curr = word_tokenize(current_summary)
            agreement = meteor_score([tokenized_ref], tokenized_curr)

        # Compute signals for signal-based and trend-aware stopping criteria
        signals = compute_signals(slides, current_summary, target_words=target_words)

        # Composite quality score used by decision-table stopping (0..1)
        rubric_norm = max(0.0, min(5.0, avg_score)) / 5.0
        coverage = signals.get("section_coverage_pct", 0.0)
        hallucination = signals.get("suspected_hallucination_rate", 1.0)
        blended_quality = 0.6 * rubric_norm + 0.2 * agreement + 0.2 * coverage
        quality_score = blended_quality * (1 - 0.15 * hallucination)

        all_signals_history.append(signals)
        all_quality_history.append(quality_score)

        # Create state object
        state = RefinementState(
            iteration=iteration,
            summary=current_summary,
            rubric=rubric,
            lever_history=all_lever_history.copy() + [rubric],
            word_count=word_count,
            change_magnitude=change_magnitude,
            avg_rubric_score=avg_score,
            target_words=target_words,
            agreement_score=agreement,
            signals=signals,
            signals_history=all_signals_history.copy(),
            quality_score=quality_score,
            quality_history=all_quality_history.copy(),
        )

        print(state.log_state())
        all_lever_history.append(rubric)
        
        # 2. Check stopping criteria
        should_stop, reason = controller.should_stop(state)
        if should_stop:
            stopping_reason = reason
            print(f"[STOPPING] {reason}")
            break
        
        # 3. Refine with lever-based guidance
        print(f"[Iter {iteration}] Refining with lever-based guidance...")
        prev_summary = current_summary
        
        refined_candidate = refine_once_with_guidance(
            slides=slides,
            summary=current_summary,
            judge_feedback=judge_feedback,
            controller=controller,
            state=state,
            cfg_refine=cfg_refine,
        )
        
        # 4. Pairwise comparison to select best
        print(f"[Iter {iteration}] Running pairwise comparison...")
        pairwise_result = round_robin_pairwise(
            slides=slides,
            summaries={"current": current_summary, "refined": refined_candidate},
            cfg_judge=cfg_judge,
            runs=3,
        )
        
        current_summary = pairwise_result["result_summary"]
        
        # 5. Save intermediate output
        if save_callback:
            save_callback(iteration, current_summary)

    # Create metadata
    metadata = {
        "iterations_completed": iteration,
        "final_avg_score": sum(rubric.values()) / len(rubric) if rubric else 0,
        "final_word_count": len(current_summary.split()),
        "stopping_reason": stopping_reason,
        "lever_history": all_lever_history,
        "quality_history": all_quality_history,
        "target_words": target_words,
        "final_rubric": rubric,
        "final_agreement": agreement,
        "final_quality_score": all_quality_history[-1] if all_quality_history else 0.0,
    }

    return current_summary, metadata

