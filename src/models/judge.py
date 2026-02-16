from typing import Dict, Any, List
import json
import random

from src.models.llm_client import call_llm, parse_json_or_throw, LLMConfig
from src.utils.chunking import slides_to_text


#prompts
RUBRIC_PROMPT = """
You are a strict evaluation engine for lecture summarization.

Your job:
1) Compute ALL metrics exactly when possible.
2) Do NOT guess metric values.
3) If a metric cannot be computed exactly (e.g., no reference provided), return "N/A".
4) Faithfulness is the highest priority.
5) Output ONLY valid JSON. No explanations outside JSON.

--------------------------------------------------
TOKENIZATION RULE (MANDATORY)

- Lowercase all text.
- Tokens are sequences of characters a–z and 0–9.
- All other characters are separators.

--------------------------------------------------
INPUTS

SOURCE:
<<<
{source_text}
>>>

SUMMARY:
<<<
{summary_text}
>>>

REFERENCE (if available; otherwise empty):
<<<
{reference_text}
>>>

--------------------------------------------------
STEP 1 — TOKEN COUNTS (exact)

Count:
- source_tokens
- summary_tokens
- reference_tokens (0 if empty)

--------------------------------------------------
STEP 2 — ACCURACY METRICS (Reference-Based)

Only compute if reference_tokens > 0.

A) ROUGE-1 exact
Define unigram overlap as multiset overlap:
overlap = sum_t min(count_summary[t], count_reference[t])

precision = overlap / summary_tokens
recall = overlap / reference_tokens
f1 = 2 * precision * recall / (precision + recall)

B) BLEU-1 exact
Use unigram precision:
bleu1 = overlap / summary_tokens

C) METEOR proxy
Compute harmonic mean of unigram precision and recall
meteor_proxy = f1  (since full METEOR requires stemming/synonyms)

D) BERTScore proxy
If no embedding computation is possible, return "N/A".

If no reference exists:
Return "N/A" for all accuracy metrics.

--------------------------------------------------
STEP 3 — FAITHFULNESS (Source-Based)

A) SummaC Proxy (0–1)

1. Split SUMMARY into sentences using . ! ?
2. For each sentence, classify:
   - Supported
   - Partial
   - Not Supported
3. Score:
(1.0*S + 0.5*P + 0.0*N) / total_sentences

B) QAFactEval Proxy (0–1)

1. Generate 5 specific factual questions implied by SUMMARY.
2. Provide:
   - expected_answer (from SUMMARY)
   - source_answer (from SOURCE only or "not found")
   - verdict: Consistent / Inconsistent
3. Score = (#Consistent) / 5

--------------------------------------------------
STEP 4 — COMPRESSION

compression_ratio = summary_tokens / source_tokens

--------------------------------------------------
STEP 5 — EXTRACTIVENESS

Left-to-right greedy maximal span matching:

- Scan SUMMARY tokens.
- At each position, find longest contiguous span appearing in SOURCE.
- If multiple matches equal length, choose earliest in SOURCE.
- Record span lengths l1..lk.

extractive_coverage = sum(li) / summary_tokens
density = sum(li^2) / summary_tokens

--------------------------------------------------
STEP 6 — RUBRIC SCORES (1–5 integers)

Use metric results to guide scoring.

Coverage (content completeness):
1 = misses major lecture concepts
3 = covers core ideas
5 = covers all major ideas proportionally

Faithfulness:
1 = multiple unsupported claims
3 = minor unsupported claims
5 = fully supported

Organization:
1 = incoherent
3 = somewhat structured
5 = logically structured

Clarity:
1 = confusing
3 = understandable
5 = very clear

Style:
1 = awkward
3 = acceptable
5 = polished

overall_1to10:
1–3 poor
4–6 adequate
7–8 strong
9–10 excellent

Faithfulness must strongly influence overall score.

--------------------------------------------------
RETURN ONLY VALID JSON:

{
  "metrics": {
    "token_counts": {
      "source": int,
      "summary": int,
      "reference": int
    },
    "accuracy": {
      "rouge1_f1_exact": float | "N/A",
      "bleu1_exact": float | "N/A",
      "meteor_proxy": float | "N/A",
      "bertscore_proxy": float | "N/A"
    },
    "faithfulness": {
      "summac_proxy_score": float,
      "qafacteval_proxy_score": float
    },
    "compression_ratio": float,
    "extractiveness": {
      "coverage": float,
      "density": float
    }
  },
  "rubric_scores": {
    "coverage": int,
    "faithfulness": int,
    "organization": int,
    "clarity": int,
    "style": int,
    "overall_1to10": int
  },
  "two_strengths": ["...", "..."],
  "two_issues": ["...", "..."],
  "faithfulness_evidence": [
    {
      "sentence": "...",
      "label": "Supported|Partial|Not",
      "evidence_from_source": "..."
    }
  ],
  "qa_details": [
    {
      "q": "...",
      "expected_answer": "...",
      "source_answer": "...",
      "verdict": "Consistent|Inconsistent"
    }
  ]
}
"""



AGREEMENT_PROMPT = """
You are grading agreement between a REFERENCE summary and a MODEL summary for the same lecture.

Focus ONLY on:
- essential fact overlap
- missing key points
- inaccuracies added

Prioritize factual correctness over wording.

Return ONLY valid JSON:
{
  "agreement_1to5": int,
  "missing_key_points": ["...", "..."],
  "added_inaccuracies": ["...", "..."]
}
"""


PAIRWISE_PROMPT = """
You are comparing two summaries written for the SAME lecture.

Priorities (highest to lowest):
1) faithfulness to SOURCE
2) coverage of key points
3) clarity/organization for students
4) style

Pick which is better overall for students. Be specific and cite 1–2 short phrases from SOURCE.

Return ONLY JSON:
{
  "winner": "A" or "B",
  "reason": "..."
}
"""



#Refernce-free Rubric Judge
def judge_rubric(slides: List[Dict], summary: str, cfg: LLMConfig) -> Dict[str, Any]:

    slide_text = slides_to_text(slides)
    user_msg = f"[Slides]\n{slide_text}\n\n[Summary]\n{summary}\n\nReturn ONLY JSON."

    raw = call_llm(
        system_prompt=RUBRIC_PROMPT,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )
    return parse_json_or_throw(raw)



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
def judge_rubric_ensemble(slides, summary, cfg: LLMConfig, runs: int = 3) -> Dict[str, Any]:

    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )
        outs.append(judge_rubric(slides, summary, cfg_r))

    keys = ["coverage", "faithfulness", "organization", "clarity", "style", "overall_1to10"]
    avg = {k: int(round(sum(o[k] for o in outs) / len(outs))) for k in keys}

    # Carry narrative fields from first run
    avg["two_strengths"] = outs[0]["two_strengths"]
    avg["two_issues"] = outs[0]["two_issues"]
    avg["faithfulness_evidence"] = outs[0]["faithfulness_evidence"]

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


def judge_pairwise_ensemble(slides, A, B, cfg: LLMConfig, runs: int = 5) -> Dict[str, Any]:
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
