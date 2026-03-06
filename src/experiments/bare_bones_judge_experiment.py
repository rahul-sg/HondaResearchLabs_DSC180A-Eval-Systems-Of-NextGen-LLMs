import json
import sys
from pathlib import Path
from typing import Dict, List

from src.models.judge import judge_rubric_ensemble, judge_agreement_ensemble
from src.models.llm_client import LLMConfig
from src.evaluation.scoring import combine_scores
from src.utils.io import load_slides


ROOT = Path(__file__).resolve().parents[2]


def _discover_lecture_ids() -> List[str]:
    slides_dir = ROOT / "data" / "slides"
    lecture_ids = sorted(p.stem for p in slides_dir.glob("lecture*.pdf"))
    if not lecture_ids:
        raise FileNotFoundError(f"No lecture PDFs found in {slides_dir}")
    return lecture_ids


def _load_text(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} is empty: {path}")
    return text


def _evaluate_summary(slides: List[Dict], reference: str, summary: str, cfg_judge: LLMConfig) -> Dict:
    rubric = judge_rubric_ensemble(slides, summary, cfg_judge, runs=3, use_domain_aware=True)
    agreement = judge_agreement_ensemble(reference, summary, cfg_judge, runs=3)
    final_score = combine_scores(rubric, agreement)

    return {
        "rubric": rubric,
        "agreement": agreement,
        "final_score_0to1": final_score,
        "word_count": len(summary.split()),
    }


def _run_lecture(lecture_id: str, cfg_judge: LLMConfig, out_root: Path) -> Dict:
    slide_path = ROOT / "data" / "slides" / f"{lecture_id}.pdf"
    reference_path = ROOT / "data" / "references" / f"{lecture_id}_reference.txt"
    s0_path = ROOT / "data" / "summaries" / "model_s0" / f"{lecture_id}.txt"
    refined_path = ROOT / "data" / "summaries" / "refined_iterations" / lecture_id / "final.txt"

    slides = load_slides(str(slide_path))["slides"]
    reference = _load_text(reference_path, "Reference")
    s0_summary = _load_text(s0_path, "S0 summary")

    lecture_out = out_root / lecture_id
    lecture_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[BARE-BONES] Evaluating S0 for {lecture_id}...")
    s0_eval = _evaluate_summary(slides, reference, s0_summary, cfg_judge)

    result = {
        "lecture_id": lecture_id,
        "s0": s0_eval,
    }

    if refined_path.exists():
        refined_summary = _load_text(refined_path, "Refined summary")
        print(f"[BARE-BONES] Evaluating refined final for {lecture_id}...")
        refined_eval = _evaluate_summary(slides, reference, refined_summary, cfg_judge)
        delta = refined_eval["final_score_0to1"] - s0_eval["final_score_0to1"]
        result["refined"] = refined_eval
        result["delta_refined_minus_s0"] = delta

    (lecture_out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"BARE-BONES JUDGE RESULT: {lecture_id}",
        f"S0 final score: {s0_eval['final_score_0to1']:.4f}",
        f"S0 word count: {s0_eval['word_count']}",
    ]
    if "refined" in result:
        lines.extend([
            f"Refined final score: {result['refined']['final_score_0to1']:.4f}",
            f"Refined word count: {result['refined']['word_count']}",
            f"Delta (refined - s0): {result['delta_refined_minus_s0']:+.4f}",
        ])

    (lecture_out / "result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[BARE-BONES] Saved: {lecture_out / 'result.json'}")

    return result


def _write_aggregate(results: List[Dict], out_root: Path) -> None:
    s0_scores = [r["s0"]["final_score_0to1"] for r in results]

    refined_rows = [r for r in results if "refined" in r]
    refined_scores = [r["refined"]["final_score_0to1"] for r in refined_rows]
    deltas = [r["delta_refined_minus_s0"] for r in refined_rows]

    aggregate = {
        "num_lectures": len(results),
        "avg_s0_score": sum(s0_scores) / len(s0_scores) if s0_scores else 0.0,
        "num_refined_available": len(refined_rows),
        "avg_refined_score": (sum(refined_scores) / len(refined_scores)) if refined_scores else None,
        "avg_delta_refined_minus_s0": (sum(deltas) / len(deltas)) if deltas else None,
        "per_lecture": results,
    }

    json_path = out_root / "all_lectures_summary.json"
    txt_path = out_root / "all_lectures_summary.txt"

    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "BARE-BONES JUDGE SUMMARY (ALL LECTURES)",
        f"Lectures evaluated: {aggregate['num_lectures']}",
        f"Avg S0 score: {aggregate['avg_s0_score']:.4f}",
        f"Refined available: {aggregate['num_refined_available']}",
    ]
    if aggregate["avg_refined_score"] is not None:
        lines.append(f"Avg refined score: {aggregate['avg_refined_score']:.4f}")
        lines.append(f"Avg delta (refined - s0): {aggregate['avg_delta_refined_minus_s0']:+.4f}")

    lines.append("")
    lines.append("Per-lecture results:")
    for row in results:
        base = f"- {row['lecture_id']}: s0={row['s0']['final_score_0to1']:.4f}"
        if "refined" in row:
            base += (
                f", refined={row['refined']['final_score_0to1']:.4f}, "
                f"delta={row['delta_refined_minus_s0']:+.4f}"
            )
        lines.append(base)

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n===== BARE-BONES AGGREGATE SAVED =====")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")


def main() -> None:
    lecture_arg = sys.argv[1].strip() if len(sys.argv) > 1 else "all"

    if lecture_arg.lower() == "all":
        lecture_ids = _discover_lecture_ids()
    else:
        lecture_ids = [lecture_arg]

    out_root = ROOT / "data" / "summaries" / "bare_bones_experiment"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_judge = LLMConfig(
        model="gpt-5-chat-latest",
        max_completion_tokens=512,
    )

    results = []
    for lecture_id in lecture_ids:
        results.append(_run_lecture(lecture_id, cfg_judge, out_root))

    if len(results) > 1:
        _write_aggregate(results, out_root)


if __name__ == "__main__":
    main()
