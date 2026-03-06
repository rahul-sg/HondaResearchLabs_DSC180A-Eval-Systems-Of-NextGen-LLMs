from typing import Dict, Any, List
import json
import random
import re

from src.models.llm_client import call_llm, parse_json_or_throw, LLMConfig
from src.utils.chunking import slides_to_text
from src.evaluation.NormalSchema import get_domain_schema


#prompts
RUBRIC_PROMPT = """You are an extremely critical teaching assistant. Evaluate a student-facing summary of a lecture with high standards.

IMPORTANT:
- Prioritize FAITHFULNESS over style. Penalize inventions or unsupported claims heavily.
- Require explicit justification from slides for each major claim.
- Identify at least 2 specific weaknesses or areas for improvement, even if the summary is good.
- Do NOT give perfect scores unless the summary is truly exceptional.
- Cite 1–2 phrases from the slides as evidence for both strengths and issues.

Score each dimension on 1.0–5.0 scale with DECIMALS:
SCORING ANCHORS:
  1.0-2.0: Major problems (significant omissions, inaccuracies, incoherence)
  2.1-3.0: Needs work (some correct content but substantial gaps or unclear organization)
  3.1-4.0: Good (mostly correct and complete, minor issues or unclear sections)
  4.1-4.7: Very good (accurate, well-organized, only minor refinements needed)
  4.8-5.0: Exceptional (comprehensive, accurate, well-written, no meaningful flaws)

1) coverage (includes main ideas and scope)
2) faithfulness (accurate to slides, no unsupported claims)
3) organization (logical flow and structure)
4) clarity (easy to understand, well-explained)
5) style (precise terminology, appropriate nuance)

Also provide:
- overall_1to10 (1–10 integer)
- two_strengths: [ "...", "..." ]
- two_issues: ["...", "..."]
- faithfulness_evidence: ["...", "..."]

Return ONLY valid JSON with FLOAT scores (not integers):
{
  "coverage": float,
  "faithfulness": float,
  "organization": float,
  "clarity": float,
  "style": float,
  "overall_1to10": int,
  "two_strengths": ["...", "..."],
  "two_issues": ["...", "..."],
  "faithfulness_evidence": ["...", "..."]
}
"""

DOMAIN_RUBRIC_PROMPT = """You are an extremely critical teaching assistant specializing in {domain} evaluation. Evaluate a student-facing summary of a {domain} lecture with high standards.

DOMAIN-SPECIFIC CRITERIA:
{domain_criteria}

IMPORTANT:
- Prioritize FAITHFULNESS over style. Penalize inventions or unsupported claims heavily.
- Require explicit justification from slides for each major claim.
- Identify at least 2 specific weaknesses or areas for improvement, even if the summary is good.
- Do NOT give perfect scores unless the summary is truly exceptional and matches all domain criteria perfectly.
- Cite 1–2 phrases from the slides as evidence for both strengths and issues.
- Apply {domain} standards rigorously.

Score each dimension on 1.0–5.0 scale with DECIMALS:
SCORING ANCHORS:
  1.0-2.0: Major problems (significant omissions, inaccuracies, incoherence, domain violations)
  2.1-3.0: Needs work (some correct content but substantial gaps or unclear organization)
  3.1-4.0: Good (mostly correct and complete, minor issues or unclear sections)
  4.1-4.7: Very good (accurate, well-organized, domain-compliant, only minor refinements needed)
  4.8-5.0: Exceptional (comprehensive, accurate, well-written, domain criteria fully met, no meaningful flaws)

1) coverage: {coverage_def}
2) faithfulness: {faithfulness_def}
3) organization: {organization_def}
4) clarity: {clarity_def}
5) style: {style_def}

Also provide:
- overall_1to10 (1–10 integer)
- two_strengths: [ "...", "..." ]
- two_issues: ["...", "..."]
- faithfulness_evidence: ["...", "..."]

Return ONLY valid JSON with FLOAT scores (not integers):
{{
  "coverage": float,
  "faithfulness": float,
  "organization": float,
  "clarity": float,
  "style": float,
  "overall_1to10": int,
  "two_strengths": ["...", "..."],
  "two_issues": ["...", "..."],
  "faithfulness_evidence": ["...", "..."]
}}
"""

AGREEMENT_PROMPT = """You are grading agreement between a reference summary and a model summary.

Focus ONLY on:
- essential fact overlap
- missing key points
- inaccuracies added

Return ONLY valid JSON:
{
  "agreement_1to5": int,
  "missing_key_points": ["...", "..."],
  "added_inaccuracies": ["...", "..."]
}
"""

PAIRWISE_PROMPT = """You are comparing two summaries written for the SAME lecture.

Pick which is better overall for students.

Return ONLY JSON:
{
  "winner": "A" or "B",
  "reason": "..."
}
"""


def detect_domain(slides: List[Dict], cfg: LLMConfig) -> str:
    """Detect the academic domain from lecture slides."""
    slide_text = slides_to_text(slides)

    # Simple keyword-based detection first
    text_lower = slide_text.lower()

    # Domain keywords
    domain_keywords = {
        "math": ["theorem", "proof", "equation", "mathematical", "calculus", "algebra", "geometry", "statistics", "probability"],
        "humanities": ["literature", "philosophy", "history", "art", "culture", "society", "politics", "ethics", "theory"],
        "natural_sciences": ["biology", "chemistry", "physics", "experiment", "hypothesis", "scientific method", "molecular", "cellular"],
        "business": ["economics", "finance", "marketing", "management", "strategy", "market", "profit", "revenue", "corporate"]
    }

    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        domain_scores[domain] = score

    # Return domain with highest score, default to "humanities" if tie
    max_score = max(domain_scores.values())
    if max_score == 0:
        return "humanities"  # default

    candidates = [d for d, s in domain_scores.items() if s == max_score]
    return candidates[0]  # take first if tie


def judge_rubric(slides: List[Dict], summary: str, cfg: LLMConfig, use_domain_aware: bool = True) -> Dict[str, Any]:
    """Reference-free Rubric Judge with optional domain-aware evaluation."""

    if not use_domain_aware:
        # Use generic rubric
        slide_text = slides_to_text(slides)
        user_msg = f"[Slides]\n{slide_text}\n\n[Summary]\n{summary}\n\nReturn ONLY JSON."

        raw = call_llm(
            system_prompt=RUBRIC_PROMPT,
            user_prompt=user_msg,
            cfg=cfg,
            json_mode=True
        )
        return parse_json_or_throw(raw)

    # Domain-aware evaluation
    domain = detect_domain(slides, cfg)
    schema = get_domain_schema(domain)

    # Build domain-specific prompt
    domain_criteria = f"This is a {domain.upper()} lecture. Apply {domain}-specific evaluation standards."

    prompt = DOMAIN_RUBRIC_PROMPT.format(
        domain=domain,
        domain_criteria=domain_criteria,
        coverage_def=schema["dimensions"]["coverage"]["definition"],
        faithfulness_def=schema["dimensions"]["faithfulness"]["definition"],
        organization_def=schema["dimensions"]["organization"]["definition"],
        clarity_def=schema["dimensions"]["clarity"]["definition"],
        style_def=schema["dimensions"]["style"]["definition"]
    )

    slide_text = slides_to_text(slides)
    user_msg = f"[Slides]\n{slide_text}\n\n[Summary]\n{summary}\n\nReturn ONLY JSON."

    raw = call_llm(
        system_prompt=prompt,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )

    result = parse_json_or_throw(raw)
    result["detected_domain"] = domain  # Add domain info to result
    return result



#Agreement judge
def judge_agreement(reference: str, summary: str, cfg: LLMConfig) -> Dict[str, Any]:

    user_msg = f"[Reference]\n{reference}\n\n[Model Summary]\n{summary}\nReturn ONLY JSON."

    raw = call_llm(
        system_prompt=AGREEMENT_PROMPT,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )
    return parse_json_or_throw(raw)



# Pairwise judge (A vs B)
def judge_pairwise(slides: List[Dict], A: str, B: str, cfg: LLMConfig) -> Dict[str, Any]:

    slide_text = slides_to_text(slides)
    user_msg = (
        f"[Slides]\n{slide_text}\n\n"
        f"[Summary A]\n{A}\n\n"
        f"[Summary B]\n{B}\n\n"
        f"Return ONLY JSON."
    )

    raw = call_llm(
        system_prompt=PAIRWISE_PROMPT,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )

    data = parse_json_or_throw(raw)

    # If hallucinated, force correction
    winner = data.get("winner", "").strip()
    if winner not in ("A", "B"):
        winner = random.choice(["A", "B"])
        data["winner"] = winner

    return data


#Average score of multiple rubric judges
def judge_rubric_ensemble(slides, summary, cfg: LLMConfig, runs: int = 3, use_domain_aware: bool = True) -> Dict[str, Any]:

    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )
        outs.append(judge_rubric(slides, summary, cfg_r, use_domain_aware=use_domain_aware))

    keys = ["coverage", "faithfulness", "organization", "clarity", "style", "overall_1to10"]
    avg = {k: int(round(sum(o[k] for o in outs) / len(outs))) for k in keys}

    # Carry narrative fields from first run
    avg["two_strengths"] = outs[0]["two_strengths"]
    avg["two_issues"] = outs[0]["two_issues"]
    avg["faithfulness_evidence"] = outs[0]["faithfulness_evidence"]

    # Include detected domain from first run
    if "detected_domain" in outs[0]:
        avg["detected_domain"] = outs[0]["detected_domain"]

    return avg


def judge_agreement_ensemble(reference, summary, cfg: LLMConfig, runs: int = 3) -> Dict[str, Any]:
    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )
        outs.append(judge_agreement(reference, summary, cfg_r))

    score = int(round(sum(o["agreement_1to5"] for o in outs) / len(outs)))

    return {
        "agreement_1to5": score,
        "missing_key_points": outs[0]["missing_key_points"],
        "added_inaccuracies": outs[0]["added_inaccuracies"]
    }


def pairwise_judge_ensemble(slides, A, B, cfg: LLMConfig, runs: int = 5) -> Dict[str, Any]:
    wins = {"A": 0, "B": 0}
    reasons = []

    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )

        result = judge_pairwise(slides, A, B, cfg_r)
        winner = result["winner"]
        wins[winner] += 1
        reasons.append(result.get("reason", ""))

    final = "A" if wins["A"] >= wins["B"] else "B"

    return {
        "winner": final,
        "wins": wins,
        "reasons_sample": reasons[:2]
    }
