#!/usr/bin/env python3
"""
iterative_refinement_test.py

Simple simulation of an iterative, lever-based rubric refinement loop.
This script does not call any LLM; it simulates summary edits and updates
rubric parts (levers) up/down/stay based on how the summary changes.

Run locally to see iteration traces and stopping behaviour.
"""
import argparse
import glob
import math
import os
import random
import re
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def change_metric(prev: str, curr: str) -> float:
    """Return a change magnitude in [0,1] where 0 means identical."""
    return 1.0 - jaccard_similarity(prev, curr)


KEYWORD_POOLS = {
    "coverage": [
        "includes main ideas",
        "covers core concepts",
        "mentions key examples",
    ],
    "clarity": [
        "is easy to follow",
        "simple and clear",
        "well structured",
    ],
    "accuracy": [
        "factually correct",
        "no incorrect claims",
        "references correct definitions",
    ],
    "conciseness": [
        "brief and focused",
        "no unnecessary detail",
        "succinct summary",
    ],
}


def simulate_refine(summary: str, rubric: Dict[str, int]) -> str:
    """Simulate refinement by adding/removing short phrases tied to rubric levers.

    For parts where the rubric is low, add phrases from the corresponding pool.
    For parts where the rubric is high, remove matching phrases (simulate pruning).
    """
    s = summary.strip()
    # break into pseudo-sentences by splitting on '.' (very simple)
    sentences = [seg.strip() for seg in s.split('.') if seg.strip()]

    # For each rubric part, decide to add or remove a phrase
    for part, score in rubric.items():
        pool = KEYWORD_POOLS.get(part, [])
        if not pool:
            continue

        if score <= 1:
            # low score -> add a short phrase to improve this part
            phrase = random.choice(pool)
            sentences.append(phrase)
        elif score >= 4:
            # high score -> try to remove one sentence containing any pool token
            tokens = set(" ".join(pool).split())
            removed = False
            for i, sent in enumerate(list(sentences)):
                if tokens & set(tokenize(sent)):
                    sentences.pop(i)
                    removed = True
                    break
            if not removed and sentences:
                # if nothing matched, optionally trim last sentence
                sentences = sentences[:-1]
        else:
            # mid scores -> small tweak: replace a sentence with a pool phrase
            if sentences:
                i = random.randrange(len(sentences))
                sentences[i] = random.choice(pool)

    # shuffle a bit to simulate reordering
    random.shuffle(sentences)
    return ". ".join(sentences) + ("." if sentences else "")


def presence_by_part(text: str) -> Dict[str, float]:
    """Return a score per rubric part indicating presence of related tokens (0..1)."""
    tokens = set(tokenize(text))
    out = {}
    for part, pool in KEYWORD_POOLS.items():
        pool_tokens = set(" ".join(pool).split())
        if not pool_tokens:
            out[part] = 0.0
        else:
            out[part] = len(tokens & pool_tokens) / len(pool_tokens)
    return out


def lever_update(rubric: Dict[str, int], prev_text: str, curr_text: str, step=1,
                 min_v=0, max_v=5) -> Dict[str, int]:
    """Update rubric levers up/down/stay based on presence deltas per part."""
    prev = presence_by_part(prev_text)
    curr = presence_by_part(curr_text)
    new = rubric.copy()
    for part in rubric:
        delta = curr.get(part, 0.0) - prev.get(part, 0.0)
        if delta > 0.05:
            new[part] = min(max_v, rubric[part] + step)
        elif delta < -0.05:
            new[part] = max(min_v, rubric[part] - step)
        else:
            # no meaningful change -> keep same
            new[part] = rubric[part]
    return new


def run_simulation(init_summary: str, max_iterations: int = 10, stop_threshold: float = 0.05):
    # initialize a mid-range rubric
    rubric = {k: 2 for k in KEYWORD_POOLS.keys()}
    cur = init_summary
    history: List[Tuple[int, str, Dict[str, int], float]] = []

    for it in range(1, max_iterations + 1):
        cand = simulate_refine(cur, rubric)
        change = change_metric(cur, cand)

        new_rubric = lever_update(rubric, cur, cand)

        history.append((it, cand, new_rubric, change))

        print(f"Iteration {it} | change={change:.3f} | rubric={new_rubric}")
        print("Summary:", cand)
        print("---")

        # stopping criterion: when candidate changed very little from previous
        if change <= stop_threshold:
            print(f"Stopping: change {change:.3f} <= stop_threshold {stop_threshold}")
            break

        # update state for next iter
        cur = cand
        rubric = new_rubric

    print("Final iteration:")
    if history:
        it, cand, new_rubric, change = history[-1]
        print(f"iter={it} change={change:.3f} rubric={new_rubric}")
        print(cand)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--stop-threshold", type=float, default=0.05,
                        help="stop when change magnitude <= threshold (0..1)")
    parser.add_argument("--summary-path", type=str, default="data/summaries/model_s0",
                        help="path to a summary file or directory containing summaries (.txt)")
    args = parser.parse_args()

    summary_path = args.summary_path

    # if a single file was provided, run once; if a directory, run for each .txt
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            init_summary = f.read()
        print(f"Running simulation for: {summary_path}")
        run_simulation(init_summary, max_iterations=args.max_iterations,
                       stop_threshold=args.stop_threshold)
    elif os.path.isdir(summary_path):
        pattern = os.path.join(summary_path, "*.txt")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No .txt summary files found in directory: {summary_path}")
            return
        for fn in files:
            print("\n==============================")
            print(f"File: {fn}")
            with open(fn, "r", encoding="utf-8") as f:
                init_summary = f.read()
            run_simulation(init_summary, max_iterations=args.max_iterations,
                           stop_threshold=args.stop_threshold)
    else:
        print(f"Path not found: {summary_path}")


if __name__ == "__main__":
    main()
