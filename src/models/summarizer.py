from typing import Dict, List

from src.utils.io import load_slides
from src.utils.chunking import slides_to_text
from src.utils.filter_slides import filter_content_slides
from src.models.llm_client import call_llm, LLMConfig


# Prompt for summarization model
SUMMARIZER_PROMPT = """
You are a lecture summarization assistant trained to produce clear,
accurate, and academically appropriate summaries.

TASK:
Summarize the following lecture content into a coherent 250–350 word summary
that:
- captures all major topics,
- preserves factual accuracy,
- avoids hallucinations,
- organizes information logically,
- is easy for a student to understand.

IMPORTANT:
- Only use information from the slides provided.
- Ignore syllabus, logistics, or administrative content.

Return ONLY the summary text.
"""

# Single call to summarization model
def _call_summarizer_once(slides_text: str, cfg: LLMConfig) -> str:
    user_prompt = f"{SUMMARIZER_PROMPT}\n\n[Slides]\n{slides_text}"

    summary = call_llm(
        system_prompt="You summarize lecture slides accurately for students.",
        user_prompt=user_prompt,
        cfg=cfg,
        json_mode=False,
    )

    return (summary or "").strip()


# Initial summary
def generate_initial_summary(
    pdf_path: str,
    cfg_summarizer: LLMConfig,
    retry_limit: int = 2,
) -> str:

    # 1. Parse entire slide deck (PDF → structured slides)
    slides_dict = load_slides(pdf_path)
    all_slides: List[Dict] = slides_dict["slides"]

    # 2. Remove administrative/syllabus slides
    content_slides = filter_content_slides(all_slides)

    # 3. Convert to text
    slides_text = slides_to_text(content_slides)

    # 4. Call the LLM with simple retry logic
    for attempt in range(retry_limit):
        summary = _call_summarizer_once(slides_text, cfg_summarizer)
        # basic sanity check: require at least ~120 words
        if summary and len(summary.split()) >= 120:
            return summary

    # If summarizer fails
    raise RuntimeError(
        "Summarizer failed to produce a sufficiently long summary "
        f"after {retry_limit} attempts using model {cfg_summarizer.model}."
    )
