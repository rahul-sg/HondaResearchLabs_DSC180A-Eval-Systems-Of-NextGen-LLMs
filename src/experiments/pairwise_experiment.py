import json
import sys
from pathlib import Path
from typing import Dict, List

from src.evaluation.pipeline import evaluate_summary
from src.models.llm_client import LLMConfig
from src.models.summarizer import generate_initial_summary


ROOT = Path(__file__).resolve().parents[2]


def _discover_lecture_ids() -> List[str]:
    slides_dir = ROOT / "data" / "slides"
    lecture_ids = sorted(p.stem for p in slides_dir.glob("lecture*.pdf"))
    if not lecture_ids:
        raise FileNotFoundError(f"No lecture PDFs found in {slides_dir}")
    return lecture_ids


def _load_or_generate_s0(lecture_id: str, force_regen: bool) -> str:
    initial_summary_path = ROOT / "data" / "summaries" / "model_s0" / f"{lecture_id}.txt"

    if initial_summary_path.exists() and not force_regen:
        with open(initial_summary_path, "r", encoding="utf-8") as f:
            s0 = f.read().strip()
        if len(s0.split()) >= 50:
            print(f"[S0] Reusing existing S0 at {initial_summary_path}")
            return s0
        print("[S0] Existing S0 too short; regenerating...")

    print("[S0] Generating new S0 with gpt-5-chat-latest...")
    cfg_summarizer = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=700)
    slide_path = ROOT / "data" / "slides" / f"{lecture_id}.pdf"
    s0 = generate_initial_summary(str(slide_path), cfg_summarizer)

    initial_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(initial_summary_path, "w", encoding="utf-8") as f:
        f.write(s0)

    print(f"[S0] Saved new S0 to {initial_summary_path}")
    return s0


def _run_one(
    lecture_id: str,
    initial_summary: str,
    out_dir: Path,
    use_pairwise_selection: bool,
) -> dict:
    cfg_judge = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=512)
    cfg_refine = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=800)

    out_dir.mkdir(parents=True, exist_ok=True)

    return evaluate_summary(
        slide_path=str(ROOT / "data" / "slides" / f"{lecture_id}.pdf"),
        initial_summary=initial_summary,
        human_reference=None,
        cfg_judge=cfg_judge,
        cfg_refine=cfg_refine,
        out_dir=str(out_dir),
        target_words=350,
        use_lever_based=True,
        min_avg_score=4.0,
        min_change_threshold=0.03,
        max_iterations=12,
        min_iterations=4,
        min_agreement=0.7,
        use_pairwise_selection=use_pairwise_selection,
    )


def _extract_metrics(result: Dict) -> Dict:
    metadata = result.get("refinement_metadata", {})
    return {
        "final_score_0to1": result.get("final_score_0to1", 0.0),
        "iterations_completed": metadata.get("iterations_completed", 0),
        "stopping_reason": metadata.get("stopping_reason", ""),
    }


def _run_lecture_ablation(lecture_id: str, force_regen: bool) -> Dict:
    slide_path = ROOT / "data" / "slides" / f"{lecture_id}.pdf"

    if not slide_path.exists():
        raise FileNotFoundError(f"Slides not found: {slide_path}")

    initial_summary = _load_or_generate_s0(lecture_id, force_regen)

    out_root = ROOT / "data" / "summaries" / "pairwise_experiment" / lecture_id
    with_dir = out_root / "with_pairwise"
    without_dir = out_root / "without_pairwise"

    print(f"\n[EXPERIMENT] Running WITH pairwise selection for {lecture_id}...")
    with_result = _run_one(
        lecture_id=lecture_id,
        initial_summary=initial_summary,
        out_dir=with_dir,
        use_pairwise_selection=True,
    )

    print(f"\n[EXPERIMENT] Running WITHOUT pairwise selection for {lecture_id}...")
    without_result = _run_one(
        lecture_id=lecture_id,
        initial_summary=initial_summary,
        out_dir=without_dir,
        use_pairwise_selection=False,
    )

    with_json = out_root / "with_pairwise.json"
    without_json = out_root / "without_pairwise.json"
    with_json.write_text(json.dumps(with_result, indent=2), encoding="utf-8")
    without_json.write_text(json.dumps(without_result, indent=2), encoding="utf-8")

    with_metrics = _extract_metrics(with_result)
    without_metrics = _extract_metrics(without_result)
    delta = with_metrics["final_score_0to1"] - without_metrics["final_score_0to1"]

    print("\n===== QUICK COMPARISON =====")
    print(f"WITH pairwise final score:    {with_metrics['final_score_0to1']:.4f}")
    print(f"WITHOUT pairwise final score: {without_metrics['final_score_0to1']:.4f}")
    print(f"Delta (WITH - WITHOUT):       {delta:+.4f}")
    print(f"Saved: {with_json}")
    print(f"Saved: {without_json}")

    return {
        "lecture_id": lecture_id,
        "with_pairwise": with_metrics,
        "without_pairwise": without_metrics,
        "delta_with_minus_without": delta,
    }


def _write_aggregate_report(results: List[Dict]) -> None:
    out_root = ROOT / "data" / "summaries" / "pairwise_experiment"
    out_root.mkdir(parents=True, exist_ok=True)

    deltas = [r["delta_with_minus_without"] for r in results]
    with_scores = [r["with_pairwise"]["final_score_0to1"] for r in results]
    without_scores = [r["without_pairwise"]["final_score_0to1"] for r in results]

    with_better = sum(1 for d in deltas if d > 0)
    without_better = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - with_better - without_better

    aggregate = {
        "num_lectures": len(results),
        "avg_with_pairwise_score": sum(with_scores) / len(with_scores),
        "avg_without_pairwise_score": sum(without_scores) / len(without_scores),
        "avg_delta_with_minus_without": sum(deltas) / len(deltas),
        "with_pairwise_better_count": with_better,
        "without_pairwise_better_count": without_better,
        "tie_count": ties,
        "per_lecture": results,
    }

    aggregate_json = out_root / "all_lectures_summary.json"
    aggregate_txt = out_root / "all_lectures_summary.txt"
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "PAIRWISE ABLATION SUMMARY (ALL LECTURES)",
        f"Lectures evaluated: {aggregate['num_lectures']}",
        f"Avg WITH pairwise score:    {aggregate['avg_with_pairwise_score']:.4f}",
        f"Avg WITHOUT pairwise score: {aggregate['avg_without_pairwise_score']:.4f}",
        f"Avg delta (WITH - WITHOUT): {aggregate['avg_delta_with_minus_without']:+.4f}",
        f"WITH better: {aggregate['with_pairwise_better_count']}",
        f"WITHOUT better: {aggregate['without_pairwise_better_count']}",
        f"Ties: {aggregate['tie_count']}",
        "",
        "Per-lecture results:",
    ]

    for row in results:
        lines.append(
            f"- {row['lecture_id']}: with={row['with_pairwise']['final_score_0to1']:.4f}, "
            f"without={row['without_pairwise']['final_score_0to1']:.4f}, "
            f"delta={row['delta_with_minus_without']:+.4f}, "
            f"iters(with/without)={row['with_pairwise']['iterations_completed']}/{row['without_pairwise']['iterations_completed']}"
        )

    aggregate_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n===== AGGREGATE SUMMARY SAVED =====")
    print(f"JSON: {aggregate_json}")
    print(f"TXT:  {aggregate_txt}")


def main():
    lecture_id = sys.argv[1].strip() if len(sys.argv) > 1 else "lecture1"
    force_regen = (sys.argv[2].strip().lower() == "yes") if len(sys.argv) > 2 else False

    if lecture_id.lower() == "all":
        lecture_ids = _discover_lecture_ids()
    else:
        lecture_ids = [lecture_id]

    all_results = []
    for lid in lecture_ids:
        all_results.append(_run_lecture_ablation(lid, force_regen))

    if len(all_results) > 1:
        _write_aggregate_report(all_results)


if __name__ == "__main__":
    main()
