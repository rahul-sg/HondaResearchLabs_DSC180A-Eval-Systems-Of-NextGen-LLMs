from typing import Dict, List, Callable
import json

from src.models.llm_client import call_llm, LLMConfig
from src.models.judge import judge_rubric
from src.utils.chunking import slides_to_text
from src.utils.filter_slides import filter_content_slides



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

        print("RAW LLM RESPONSE (repr):", repr(raw_response))

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

    S = initial_summary

    # Save initial summary
    if save_callback:
        save_callback(0, S)

    for i in range(1, iters + 1):

        # 1. Judge all slides
        feedback = judge_rubric(slides, S, cfg_judge)

        # 2. Refiner uses content-only slides
        S = refine_once(slides, S, feedback, cfg_refine)

        # 3. Save intermediate outputs
        if save_callback:
            save_callback(i, S)

    return S
