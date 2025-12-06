from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json, re, random, os
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same directory as this script
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_slides(slides: List[Dict], chunk_size: int) -> List[List[Dict]]:
    # Split slides (list of dicts) into fixed-size chunks by COUNT.Mostly used for quick tests.
    return [slides[i:i + chunk_size] for i in range(0, len(slides), chunk_size)]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def chunk_slides_by_tokens(
    slides: List[Dict],
    max_tokens: int = 1500,
    text_key: str = "content"
) -> List[List[Dict]]:
    # Chunk slides so each chunk stays under ~max_tokens (approx).

    chunks: List[List[Dict]] = []
    current, cur_tok = [], 0
    for s in slides:
        t = estimate_tokens(str(s.get(text_key, "")))
        if current and cur_tok + t > max_tokens:
            chunks.append(current)
            current, cur_tok = [], 0
        current.append(s)
        cur_tok += t
    if current:
        chunks.append(current)
    return chunks


def _slides_to_str(
    slides: List[Dict],
    max_chunks: int = 3,
    max_tokens: int = 1500
) -> str:
    # Render slides into a text block for the judge,
    chunks = chunk_slides_by_tokens(slides, max_tokens=max_tokens)
    chunks = chunks[:max_chunks]
    out = []
    for ci, ch in enumerate(chunks, 1):
        for i, s in enumerate(ch, 1):
            out.append(f"[Chunk {ci} • Slide {i}] {s.get('title','')}\n{s.get('content','')}")
    return "\n\n".join(out)


def _extract_sections(
    slides: List[Dict],
    title_key: str = "title",
    content_key: str = "content"
) -> List[Tuple[str, str]]:
    
    #Turn slides into a list of (title, content) tuples.
    out = []
    for s in slides:
        title = str(s.get(title_key, "")).strip()
        content = str(s.get(content_key, "")).strip()
        out.append((title, content))
    return out


def _top_keywords_per_section(
    sections: List[Tuple[str, str]],
    k: int = 5
) -> List[List[str]]:
    #Get top TF-IDF terms 

    docs = [t + "\n" + c for (t, c) in sections]
    if len(docs) == 0:
        return [[]]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    top_terms = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            top_terms.append([])
            continue
        idx = np.argsort(row.toarray()[0])[-k:]
        top_terms.append([t for t in terms[idx] if t])
    return top_terms


def _build_glossary(sections: List[Tuple[str, str]]) -> List[str]:
    
    # Crude glossary = title tokens + **bold** + `code` + ALLCAPS tokens.
    # Lowercased for matching. Implements Glossary Recall metric.
    
    terms = set()
    for (title, content) in sections:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title):
            terms.add(tok.lower())
        for bold in re.findall(r"\*\*(.+?)\*\*", content):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", bold):
                terms.add(tok.lower())
        for code in re.findall(r"`(.+?)`", content):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", code):
                terms.add(tok.lower())
        for cap in re.findall(r"\b[A-Z]{3,}\b", content):
            terms.add(cap.lower())
    return list(terms)


def _sentence_split(text: str) -> List[str]:
    # Simple sentence splitter using punctuation boundaries.
    
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def simple_signals(
    slides: List[Dict],
    summary: str,
    target_words: int = 300
) -> Dict[str, float]:
    sections = _extract_sections(slides)

    # 1) Length Error
    wc = max(1, len(summary.split()))
    length_error = abs(wc - target_words) / float(target_words)

    # 2) Section Coverage Percentage
    topk = _top_keywords_per_section(sections, k=5)
    covered = 0
    summary_lc = summary.lower()
    for kws in topk:
        if not kws:
            continue
        hit = any(k in summary_lc for k in kws)
        covered += 1 if hit else 0
    section_coverage_pct = 0.0 if len(topk) == 0 else covered / len(topk)

    # 3) Glossary Recall
    glossary = _build_glossary(sections)
    if glossary:
        hits = sum(1 for g in glossary if g in summary_lc)
        glossary_recall = hits / len(glossary)
    else:
        glossary_recall = 0.0

    # 4) Suspected Hallucination Rate
    slide_sentences = []
    for (_, content) in sections:
        slide_sentences.extend(_sentence_split(content))
    slide_sentences = [s for s in slide_sentences if len(s.split()) >= 4]
    summary_sentences = _sentence_split(summary)

    suspected = 0
    if slide_sentences and summary_sentences:
        vec = TfidfVectorizer(stop_words="english", max_features=8000)
        vec.fit(slide_sentences + summary_sentences)
        slide_mat = vec.transform(slide_sentences)
        for s in summary_sentences:
            q = vec.transform([s])
            sims = cosine_similarity(q, slide_mat).ravel()
            if (sims >= 0.25).sum() < 1:
                suspected += 1
        suspected_hallucination_rate = suspected / len(summary_sentences)
    else:
        suspected_hallucination_rate = 0.0

    return {
        "length_error": float(length_error),
        "section_coverage_pct": float(section_coverage_pct),
        "glossary_recall": float(glossary_recall),
        "suspected_hallucination_rate": float(suspected_hallucination_rate),
    }

@dataclass
class LLMConfig:
    provider: str = "openai"         # kept for compatibility
    model: str = "gpt-5-mini"        # default; override per use
    temperature: Optional[float] = None  # None => don't send temperature
    max_tokens: int = 512
    seed: Optional[int] = None       # used for ensembles


_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def call_llm(system: str, user: str, cfg: LLMConfig, json_mode: bool = False) -> str:
    client = _get_client()

    args: Dict[str, Any] = {
        "model": cfg.model,
        "max_completion_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    # Only send temperature if explicitly set (some models like gpt-5-mini
    # do NOT support non-default temperatures).
    if cfg.temperature is not None:
        args["temperature"] = cfg.temperature

    # Seed is optional; only set if provided.
    if cfg.seed is not None:
        args["seed"] = cfg.seed

    if json_mode:
        # Forces the model to emit syntactically valid JSON.
        args["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**args)
    return resp.choices[0].message.content

_RUBRIC_PROMPT = """You are a strict teaching assistant. Evaluate a student-facing summary of a lecture.

IMPORTANT: Prioritize FAITHFULNESS over style. Do NOT reward eloquence if facts are unsupported by slides.
When judging faithfulness, cite 1–2 concrete phrases from the slides that support or contradict the summary.

SCORE the summary on the following 5 dimensions from 1 (poor) to 5 (excellent):
1) Coverage (major topics included; breadth of core points covered)
2) Faithfulness (all claims supported by the slides; no contradictions or hallucinations)
3) Organization (logical flow, signposting, paragraph cohesion)
4) Pedagogical clarity (definitions/examples help a student learn)
5) Style/Conciseness (clear, concise; within target length; no fluff)

Return ONLY JSON with keys:
{
  "coverage": int,
  "faithfulness": int,
  "organization": int,
  "clarity": int,
  "style": int,
  "overall_1to10": int,
  "two_strengths": ["...", "..."],
  "two_issues": ["...", "..."],
  "faithfulness_evidence": ["slide quote or paraphrase", "slide quote or paraphrase"]
}
"""

_AGREEMENT_PROMPT = """You are grading agreement with a reference answer.
Judge overlap of essential facts and omissions. Ignore stylistic differences.

Return ONLY JSON:
{
  "agreement_1to5": int,
  "missing_key_points": ["...", "..."],
  "added_inaccuracies": ["...", "..."]
}
"""

_PAIRWISE_PROMPT = """You are selecting the better study summary for students.

Choose the better overall summary for students to study for THIS lecture.
Return ONLY JSON: {"winner": "A"|"B", "reason": "..."}
"""


#force json output from LLM
def _force_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError(f"LLM did not return valid JSON:\n{text}")

_REFINE_PROMPT = """
You are improving a lecture summary based on feedback from a judge.

[Slides]
{slides}

[Current Summary]
{summary}

[Judge Feedback]
{feedback}

TASK:
Rewrite the summary so that:
- all factual errors are fixed,
- coverage of core concepts improves,
- clarity and organization improve,
- unnecessary or redundant sentences are removed,
- the summary becomes more information-dense while staying roughly the same length.

IMPORTANT:
- Do NOT add hallucinated details not present in the slides.
- Prefer fusing related ideas into concise, content-rich sentences rather than expanding length.

Return ONLY the improved summary text, nothing else.
"""

#Refine summary from judge feeback
def refine_summary_once(
    slides: List[Dict],
    summary: str,
    feedback: str,
    cfg_refine: LLMConfig
) -> str:
    slides_str = _slides_to_str(slides, max_chunks=3, max_tokens=1500)
    user = _REFINE_PROMPT.format(slides=slides_str, summary=summary, feedback=feedback)
    raw = call_llm("You revise lecture summaries for students.", user, cfg_refine, json_mode=False)
    return raw.strip()


# Compare summary using rubric
def judge_scores(slides: List[Dict], summary: str, cfg_judge: LLMConfig) -> Dict[str, Any]:
    slides_str = _slides_to_str(slides, max_chunks=3, max_tokens=1500)
    user_msg = f"[Slides]\n{slides_str}\n\n[Summary]\n{summary}\n\nReturn ONLY JSON."
    raw = call_llm(_RUBRIC_PROMPT, user_msg, cfg_judge, json_mode=True)
    data = _force_json(raw)
    data["_meta"] = {
        "model": cfg_judge.model,
        "temperature": cfg_judge.temperature,
        "seed": cfg_judge.seed,
        "system_prompt_hash": hash(_RUBRIC_PROMPT),
        "user_len": len(user_msg),
    }
    return data

#Compare to handwritten human summary
def judge_agreement(reference: str, summary: str, cfg_judge: LLMConfig) -> Dict[str, Any]:
    user_msg = f"[Reference]\n{reference}\n\n[Model summary]\n{summary}\n\nReturn ONLY JSON."
    raw = call_llm(_AGREEMENT_PROMPT, user_msg, cfg_judge, json_mode=True)
    data = _force_json(raw)
    data["_meta"] = {
        "model": cfg_judge.model,
        "temperature": cfg_judge.temperature,
        "seed": cfg_judge.seed,
        "system_prompt_hash": hash(_AGREEMENT_PROMPT),
        "user_len": len(user_msg),
    }
    return data

#choose the better of two summaries
def pairwise_judge(slides: List[Dict], A: str, B: str, cfg_judge: LLMConfig) -> Dict[str, Any]:
    slides_str = _slides_to_str(slides, max_chunks=3, max_tokens=1500)
    user_msg = f"[Slides]\n{slides_str}\n\n[Summary A]\n{A}\n\n[Summary B]\n{B}\n\nReturn ONLY JSON."
    raw = call_llm(_PAIRWISE_PROMPT, user_msg, cfg_judge, json_mode=True)
    data = _force_json(raw)

    # Force the winner to be "A" or "B" if the model free-forms
    if data.get("winner") not in ("A", "B"):
        wtxt = json.dumps(data).lower()
        if "summary a" in wtxt or '"a"' in wtxt:
            data["winner"] = "A"
        elif "summary b" in wtxt or '"b"' in wtxt:
            data["winner"] = "B"
        else:
            data["winner"] = random.choice(["A", "B"])
            data["reason"] = (data.get("reason") or "") + " (tie-broken randomly)"

    data["_meta"] = {
        "model": cfg_judge.model,
        "temperature": cfg_judge.temperature,
        "seed": cfg_judge.seed,
        "system_prompt_hash": hash(_PAIRWISE_PROMPT),
        "user_len": len(user_msg),
    }
    return data


def iterative_refine_summary(
    slides: List[Dict],
    initial_summary: str,
    cfg_judge: LLMConfig,
    cfg_refine: LLMConfig,
    iters: int = 3
) -> str:
    
    #Implements S0 -> Judge -> S1 -> Judge -> S2 -> Judge -> S3.
    
    S = initial_summary
    for i in range(iters):
        fb = judge_scores(slides, S, cfg_judge)   # feedback dictionary
        fb_text = json.dumps({
            "coverage": fb.get("coverage"),
            "faithfulness": fb.get("faithfulness"),
            "organization": fb.get("organization"),
            "clarity": fb.get("clarity"),
            "style": fb.get("style"),
            "two_strengths": fb.get("two_strengths"),
            "two_issues": fb.get("two_issues"),
        }, indent=2)

        S = refine_summary_once(slides, S, fb_text, cfg_refine)

    return S



def judge_scores_ensemble(
    slides: List[Dict],
    summary: str,
    cfg_judge: LLMConfig,
    runs: int = 3
) -> Dict[str, Any]:
    
    #Call the rubric judge multiple times 
    outs = []
    base_seed = cfg_judge.seed or 0
    for r in range(runs):
        cfg_r = LLMConfig(
            provider=cfg_judge.provider,
            model=cfg_judge.model,
            temperature=cfg_judge.temperature,
            max_tokens=cfg_judge.max_tokens,
            seed=base_seed + r,
        )
        outs.append(judge_scores(slides, summary, cfg_r))

    meanable = ["coverage", "faithfulness", "organization", "clarity", "style", "overall_1to10"]
    avg = {k: int(round(np.mean([o.get(k, 0) for o in outs]))) for k in meanable}
    avg["two_strengths"] = outs[0].get("two_strengths", [])
    avg["two_issues"] = outs[0].get("two_issues", [])
    avg["faithfulness_evidence"] = outs[0].get("faithfulness_evidence", [])
    avg["_stdev_overall"] = float(np.std([o.get("overall_1to10", 0) for o in outs], ddof=1))
    return avg


#iterate and get average agreement score
def judge_agreement_ensemble(
    reference: str,
    summary: str,
    cfg_judge: LLMConfig,
    runs: int = 3
) -> Dict[str, Any]:
    
    outs = []
    base_seed = cfg_judge.seed or 0
    for r in range(runs):
        cfg_r = LLMConfig(
            provider=cfg_judge.provider,
            model=cfg_judge.model,
            temperature=cfg_judge.temperature,
            max_tokens=cfg_judge.max_tokens,
            seed=base_seed + r,
        )
        outs.append(judge_agreement(reference, summary, cfg_r))

    avg = {"agreement_1to5": int(round(np.mean([o.get("agreement_1to5", 0) for o in outs])))}
    avg["missing_key_points"] = outs[0].get("missing_key_points", [])
    avg["added_inaccuracies"] = outs[0].get("added_inaccuracies", [])
    avg["_stdev_agreement"] = float(np.std([o.get("agreement_1to5", 0) for o in outs], ddof=1))
    return avg


#Multiple pariwise votes to choose best summary
def pairwise_judge_ensemble(
    slides: List[Dict],
    A: str,
    B: str,
    cfg_judge: LLMConfig,
    runs: int = 5
) -> Dict[str, Any]:
    wins = {"A": 0, "B": 0}
    reasons = []
    base_seed = cfg_judge.seed or 0

    for r in range(runs):
        cfg_r = LLMConfig(
            provider=cfg_judge.provider,
            model=cfg_judge.model,
            temperature=cfg_judge.temperature,
            max_tokens=cfg_judge.max_tokens,
            seed=base_seed + r,
        )
        if r % 2 == 0:
            res = pairwise_judge(slides, A, B, cfg_r)
            w = res.get("winner", "A")
            wins[w] += 1
            reasons.append(res.get("reason", ""))
        else:
            res = pairwise_judge(slides, B, A, cfg_r)
            w = res.get("winner", "A")           # winner relative to swapped order
            w = "A" if w == "B" else "B"        # map back to original A/B names
            wins[w] += 1
            reasons.append(res.get("reason", ""))

    final_winner = "A" if wins["A"] >= wins["B"] else "B"
    return {"winner": final_winner, "wins": wins, "reasons_sample": reasons[:2]}


def pairwise_guided_refinement(
    slides: List[Dict],
    summary_candidates: Dict[str, str],
    cfg_judge: LLMConfig,
    cfg_refine: LLMConfig,
    rounds: int = 2
) -> Dict[str, str]:
    curr = summary_candidates

    for rd in range(rounds):
        names = list(curr.keys())
        scores = {n: 0 for n in names}

        # pairwise comparisons
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = names[i], names[j]
                res = pairwise_judge_ensemble(slides, curr[A], curr[B], cfg_judge, runs=3)
                winner = A if res["winner"] == "A" else B
                scores[winner] += 1

        # Keep top 50%
        sorted_names = sorted(scores, key=lambda n: scores[n], reverse=True)
        keep = sorted_names[:max(1, len(sorted_names) // 2)]

        # Refine survivors
        new_set: Dict[str, str] = {}
        for k in keep:
            fb = judge_scores(slides, curr[k], cfg_judge)
            fb_text = json.dumps(fb, indent=2)
            refined = refine_summary_once(slides, curr[k], fb_text, cfg_refine)
            new_set[k] = refined

        curr = new_set

    return curr


_REFINE_AGREEMENT_PROMPT = """
You are improving a lecture summary to better align with a reference (human-written) summary.

[Slides]
{slides}

[Current Summary]
{summary}

[Reference Summary]
{reference}

[Agreement Feedback]
{feedback}

Improve the summary WITHOUT:
- copying sentences exactly,
- introducing hallucinations not grounded in the slides.

Focus on:
- keeping all factual content aligned with the slides,
- matching reference-level coverage,
- improving clarity and structure.

Return ONLY the refined summary text.
"""

def refine_summary_toward_reference(
    slides: List[Dict],
    summary: str,
    reference: str,
    feedback: str,
    cfg_refine: LLMConfig
) -> str:
    slides_str = _slides_to_str(slides, max_chunks=3, max_tokens=1500)
    user = _REFINE_AGREEMENT_PROMPT.format(
        slides=slides_str,
        summary=summary,
        reference=reference,
        feedback=feedback
    )
    raw = call_llm("You refine summaries toward reference targets.", user, cfg_refine, json_mode=False)
    return raw.strip()


def agreement_calibrated_refinement(
    slides: List[Dict],
    initial_summary: str,
    reference: str,
    cfg_judge: LLMConfig,
    cfg_refine: LLMConfig,
    iters: int = 2
) -> str:

    S = initial_summary
    for i in range(iters):
        fb = judge_agreement(reference, S, cfg_judge)
        fb_text = json.dumps(fb, indent=2)
        S = refine_summary_toward_reference(slides, S, reference, fb_text, cfg_refine)
    return S


#Combine rubric and agreement into scalar
def llm_score(
    rubric: Dict[str, Any],
    agree: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    w = weights or {"coverage": 1, "faithfulness": 2, "organization": 1, "clarity": 1, "style": 1}
    denom = sum(w.values()) * 5.0
    r = (
        w["coverage"] * int(rubric.get("coverage", 0)) +
        w["faithfulness"] * int(rubric.get("faithfulness", 0)) +
        w["organization"] * int(rubric.get("organization", 0)) +
        w["clarity"] * int(rubric.get("clarity", 0)) +
        w["style"] * int(rubric.get("style", 0))
    ) / denom  # 0..1

    a = max(0, min(5, int(agree.get("agreement_1to5", 0)))) / 5.0
    return float(0.5 * r + 0.5 * a)


#Evaluation pipeline for one summary
#S0 -> S1 -> S2 -> S3 via judge feedback + revision.
def evaluate_one_summary(
    slides: List[Dict],
    model_summary: str,
    human_reference: str,
    cfg_judge: LLMConfig,
    cfg_refine: Optional[LLMConfig] = None,
    target_words: int = 300,
    refine_iters: int = 0,
) -> Dict[str, Any]:
    if cfg_refine is None:
        cfg_refine = cfg_judge

    # Optional: Iterative refinement
    if refine_iters > 0:
        model_summary = iterative_refine_summary(slides, model_summary, cfg_judge, cfg_refine, iters=refine_iters)

    sig = simple_signals(slides, model_summary, target_words=target_words)
    rubric = judge_scores_ensemble(slides, model_summary, cfg_judge, runs=3)
    agree = judge_agreement_ensemble(human_reference, model_summary, cfg_judge, runs=3)
    score = llm_score(rubric, agree)

    return {
        "refined_summary": model_summary,
        "signals": sig,
        "rubric": rubric,
        "agreement": agree,
        "final_score_0to1": score,
    }


# Compare all summaries using the ensemble judge
def round_robin_pairwise(
    slides: List[Dict],
    summaries: Dict[str, str],
    cfg_judge: LLMConfig
) -> Dict[str, Any]:
    names = list(summaries.keys())
    wins = {n: 0 for n in names}
    matches = []
    total_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            total_pairs += 1
            Aname, Bname = names[i], names[j]
            res = pairwise_judge_ensemble(slides, summaries[Aname], summaries[Bname], cfg_judge, runs=5)
            win_name = Aname if res["winner"] == "A" else Bname
            wins[win_name] += 1
            matches.append({
                "A": Aname, "B": Bname,
                "winner": win_name,
                "wins_detail": res["wins"],
                "reasons_sample": res["reasons_sample"],
            })

    win_rate = {k: (v / max(1, total_pairs)) for k, v in wins.items()}
    return {"wins": wins, "win_rate": win_rate, "matches": matches}



#Sanity check
def sanity_direction_of_info(slides: List[Dict], summary: str, cfg_judge: LLMConfig) -> Dict[str, Any]:
    r_full = judge_scores_ensemble(slides, summary, cfg_judge, runs=2)
    r_blind = judge_scores_ensemble([], summary, cfg_judge, runs=2)
    return {
        "faithfulness_full": r_full["faithfulness"],
        "faithfulness_blind": r_blind["faithfulness"],
        "passed": r_full["faithfulness"] > r_blind["faithfulness"]
    }



if __name__ == "__main__":
    # Example slides and summaries
    slides = [
        {"title": "Gradient Descent", "content": "Update parameters opposite the gradient to minimize loss."},
        {"title": "Learning Rate", "content": "Too high diverges; too low slows convergence; schedules can help."},
        {"title": "Stopping Criteria", "content": "Use validation loss, gradient norm, or max steps."},
    ]
    human_ref = "Covers update rule, impact of learning rate, and stopping criteria."
    model_sum = "Update parameters using the negative gradient. High LR diverges. Stop by validation or gradient."

    # Judge: gpt-5-chat-latest (JSON, temperature allowed)
    cfg_judge = LLMConfig(
        provider="openai",
        model="gpt-5-chat-latest",
        temperature=0.0,   # deterministic judge
        max_tokens=512,
        seed=7,
    )

    # Refiner: gpt-5-mini (no custom temperature allowed)
    cfg_refine = LLMConfig(
        provider="openai",
        model="gpt-5-mini",
        temperature=None,  # don't send temperature
        max_tokens=512,
        seed=7,
    )

    print("Signals (no LLM needed):", simple_signals(slides, model_sum, target_words=80))

    res = evaluate_one_summary(
        slides,
        model_sum,
        human_ref,
        cfg_judge,
        cfg_refine,
        refine_iters=3   # activates the iterative refinement loop
    )

    print("\n=== Final Scalar Score (0–1) ===")
    print(res["final_score_0to1"])

    print("\n=== Refined Summary ===")
    print(res["refined_summary"])
