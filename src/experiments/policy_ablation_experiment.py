import json
import sys
from pathlib import Path

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


def _run_policy(lecture_id: str, initial_summary: str, policy: str) -> dict:
    out_dir = ROOT / "data" / "summaries" / "policy_ablation" / lecture_id / policy
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_judge = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=512)
    cfg_refine = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=800)

    if policy == "legacy":
        alpha = 0.15
        beta = 0.10
    else:
        alpha = 0.05
        beta = 0.0

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
        scoring_policy=policy,
        hallucination_damping_alpha=alpha,
        hallucination_subtractive_beta=beta,
    )

    return {
        "final_score_0to1": result.get("final_score_0to1", 0.0),
        "raw_quality_score": result.get("leaderboard_scores", {}).get("raw_quality_score", 0.0),
        "risk_adjusted_score": result.get("leaderboard_scores", {}).get("risk_adjusted_score", 0.0),
        "iterations_completed": result.get("refinement_metadata", {}).get("iterations_completed", 0),
        "stopping_reason": result.get("refinement_metadata", {}).get("stopping_reason", ""),
    }


def main():
    lecture_id = sys.argv[1].strip() if len(sys.argv) > 1 else "all"
    force_regen = (sys.argv[2].strip().lower() == "yes") if len(sys.argv) > 2 else False

    lecture_ids = _discover_lecture_ids() if lecture_id.lower() == "all" else [lecture_id]

    rows = []
    for lid in lecture_ids:
        print(f"\n[POLICY ABLATION] {lid}")
        s0 = _load_or_generate_s0(lid, force_regen)
        tuned = _run_policy(lid, s0, "tuned")
        legacy = _run_policy(lid, s0, "legacy")
        rows.append(
            {
                "lecture_id": lid,
                "tuned": tuned,
                "legacy": legacy,
                "delta_tuned_minus_legacy": tuned["final_score_0to1"] - legacy["final_score_0to1"],
            }
        )

    tuned_mean = sum(r["tuned"]["final_score_0to1"] for r in rows) / len(rows)
    legacy_mean = sum(r["legacy"]["final_score_0to1"] for r in rows) / len(rows)
    deltas = [r["delta_tuned_minus_legacy"] for r in rows]
    tuned_better = sum(1 for d in deltas if d > 0)
    legacy_better = sum(1 for d in deltas if d < 0)
    ties = len(rows) - tuned_better - legacy_better

    report = {
        "num_lectures": len(rows),
        "avg_tuned_score": tuned_mean,
        "avg_legacy_score": legacy_mean,
        "avg_delta_tuned_minus_legacy": sum(deltas) / len(deltas),
        "tuned_better_count": tuned_better,
        "legacy_better_count": legacy_better,
        "tie_count": ties,
        "per_lecture": rows,
    }

    out_root = ROOT / "outputs" / "policy_ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "tuned_vs_legacy_summary.json"
    out_txt = out_root / "tuned_vs_legacy_summary.txt"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "TUNED VS LEGACY POLICY ABLATION",
        f"Lectures evaluated: {report['num_lectures']}",
        f"Avg tuned score:  {report['avg_tuned_score']:.4f}",
        f"Avg legacy score: {report['avg_legacy_score']:.4f}",
        f"Avg delta (tuned - legacy): {report['avg_delta_tuned_minus_legacy']:+.4f}",
        f"Tuned better: {report['tuned_better_count']}",
        f"Legacy better: {report['legacy_better_count']}",
        f"Ties: {report['tie_count']}",
        "",
        "Per lecture:",
    ]
    for row in rows:
        lines.append(
            f"- {row['lecture_id']}: tuned={row['tuned']['final_score_0to1']:.4f}, "
            f"legacy={row['legacy']['final_score_0to1']:.4f}, "
            f"delta={row['delta_tuned_minus_legacy']:+.4f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved TXT:  {out_txt}")


if __name__ == "__main__":
    main()
