import json
import sys
from pathlib import Path
from statistics import mean, pstdev

from src.evaluation.pipeline import evaluate_summary
from src.models.llm_client import LLMConfig
from src.models.summarizer import generate_initial_summary


ROOT = Path(__file__).resolve().parents[2]


def _discover_lecture_ids() -> list[str]:
    slides_dir = ROOT / "data" / "slides"
    lecture_ids = sorted(p.stem for p in slides_dir.glob("lecture*.pdf"))
    if not lecture_ids:
        raise FileNotFoundError(f"No lecture PDFs found in {slides_dir}")
    return lecture_ids


def _load_or_generate_s0(lecture_id: str, force_regen: bool) -> str:
    initial_summary_path = ROOT / "data" / "summaries" / "model_s0" / f"{lecture_id}.txt"

    if initial_summary_path.exists() and not force_regen:
        s0 = initial_summary_path.read_text(encoding="utf-8").strip()
        if len(s0.split()) >= 50:
            return s0

    cfg_summarizer = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=700)
    slide_path = ROOT / "data" / "slides" / f"{lecture_id}.pdf"
    s0 = generate_initial_summary(str(slide_path), cfg_summarizer)
    initial_summary_path.parent.mkdir(parents=True, exist_ok=True)
    initial_summary_path.write_text(s0, encoding="utf-8")
    return s0


def _run_seed(lecture_id: str, seed: int, initial_summary: str) -> dict:
    out_dir = ROOT / "data" / "summaries" / "seed_robustness" / lecture_id / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_judge = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=512, seed=seed)
    cfg_refine = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=800, seed=seed)

    result = evaluate_summary(
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
        scoring_policy="tuned",
    )

    return {
        "seed": seed,
        "final_score_0to1": result.get("final_score_0to1", 0.0),
        "raw_quality_score": result.get("leaderboard_scores", {}).get("raw_quality_score", 0.0),
        "risk_adjusted_score": result.get("leaderboard_scores", {}).get("risk_adjusted_score", 0.0),
        "iterations_completed": result.get("refinement_metadata", {}).get("iterations_completed", 0),
        "stopping_reason": result.get("refinement_metadata", {}).get("stopping_reason", ""),
    }


def _aggregate(seed_rows: list[dict]) -> dict:
    final_scores = [r["final_score_0to1"] for r in seed_rows]
    raw_scores = [r["raw_quality_score"] for r in seed_rows]
    risk_scores = [r["risk_adjusted_score"] for r in seed_rows]
    iterations = [r["iterations_completed"] for r in seed_rows]

    return {
        "num_runs": len(seed_rows),
        "mean_final_score": mean(final_scores),
        "std_final_score": pstdev(final_scores) if len(final_scores) > 1 else 0.0,
        "mean_raw_quality_score": mean(raw_scores),
        "std_raw_quality_score": pstdev(raw_scores) if len(raw_scores) > 1 else 0.0,
        "mean_risk_adjusted_score": mean(risk_scores),
        "std_risk_adjusted_score": pstdev(risk_scores) if len(risk_scores) > 1 else 0.0,
        "mean_iterations": mean(iterations),
    }


def main():
    lecture_id = sys.argv[1].strip() if len(sys.argv) > 1 else "all"
    seeds_csv = sys.argv[2].strip() if len(sys.argv) > 2 else "11,22,33"
    force_regen = (sys.argv[3].strip().lower() == "yes") if len(sys.argv) > 3 else False

    seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip()]
    lecture_ids = _discover_lecture_ids() if lecture_id.lower() == "all" else [lecture_id]

    out_root = ROOT / "outputs" / "seed_robustness"
    out_root.mkdir(parents=True, exist_ok=True)

    per_lecture = []
    for lid in lecture_ids:
        print(f"\n[ROBUSTNESS] {lid} | seeds={seeds}")
        initial_summary = _load_or_generate_s0(lid, force_regen)
        runs = [_run_seed(lid, s, initial_summary) for s in seeds]
        summary = _aggregate(runs)
        row = {
            "lecture_id": lid,
            "seeds": seeds,
            "runs": runs,
            "summary": summary,
        }
        per_lecture.append(row)

    all_means = [r["summary"]["mean_final_score"] for r in per_lecture]
    overall = {
        "num_lectures": len(per_lecture),
        "seeds": seeds,
        "grand_mean_final_score": mean(all_means) if all_means else 0.0,
        "grand_std_across_lectures": pstdev(all_means) if len(all_means) > 1 else 0.0,
    }

    report = {
        "overall": overall,
        "per_lecture": per_lecture,
    }

    out_json = out_root / "multi_seed_summary.json"
    out_txt = out_root / "multi_seed_summary.txt"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "MULTI-SEED ROBUSTNESS SUMMARY",
        f"Seeds: {seeds}",
        f"Lectures: {overall['num_lectures']}",
        f"Grand mean final score: {overall['grand_mean_final_score']:.4f}",
        f"Grand std across lectures: {overall['grand_std_across_lectures']:.4f}",
        "",
        "Per lecture:",
    ]
    for r in per_lecture:
        s = r["summary"]
        lines.append(
            f"- {r['lecture_id']}: final={s['mean_final_score']:.4f}±{s['std_final_score']:.4f}, "
            f"raw={s['mean_raw_quality_score']:.4f}±{s['std_raw_quality_score']:.4f}, "
            f"risk={s['mean_risk_adjusted_score']:.4f}±{s['std_risk_adjusted_score']:.4f}, "
            f"iters={s['mean_iterations']:.2f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved TXT:  {out_txt}")


if __name__ == "__main__":
    main()
