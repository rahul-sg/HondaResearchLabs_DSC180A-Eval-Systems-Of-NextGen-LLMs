import argparse
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


def _iteration_sort_key(name: str):
    if name.startswith("iter_"):
        try:
            return (0, int(name.split("_")[1]))
        except (IndexError, ValueError):
            return (0, 999)
    if name == "final":
        return (1, 999)
    return (2, 999)


def _parse_stop_state(stopping_reason: str) -> str:
    if not stopping_reason:
        return "unknown"
    return stopping_reason.split(":", 1)[0].strip().lower()


def load_iteration_summaries(iter_dir: Path):
    texts = {}
    for fname in iter_dir.glob("*.txt"):
        texts[fname.stem] = fname.read_text(encoding="utf-8").strip()
    return texts


def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(1536)

    from openai import OpenAI

    client = OpenAI()
    emb = client.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(emb.data[0].embedding)


def compute_semantic_drift(iter_texts):
    names = sorted(iter_texts.keys(), key=_iteration_sort_key)
    embeddings = [get_embedding(iter_texts[n]) for n in names]

    if not embeddings:
        return [], []

    s0_emb = embeddings[0]
    sims = [cosine_similarity([s0_emb], [emb])[0][0] for emb in embeddings]
    drifts = [1 - sim for sim in sims]
    return names, drifts


def load_cohort_records(example_root: Path):
    records = []
    if not example_root.exists():
        return records

    for lecture_dir in sorted([p for p in example_root.glob("lecture*") if p.is_dir()]):
        result_path = lecture_dir / "result.json"
        pairwise_path = lecture_dir / "pairwise_s0_vs_refined.json"

        if not result_path.exists():
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = result.get("refinement_metadata", {})
        stop_reason = metadata.get("stopping_reason", "")

        pairwise_refined = np.nan
        pairwise_s0 = np.nan
        if pairwise_path.exists():
            pairwise = json.loads(pairwise_path.read_text(encoding="utf-8"))
            pairwise_refined = pairwise.get("win_rate", {}).get("gpt5_refined", np.nan)
            pairwise_s0 = pairwise.get("win_rate", {}).get("gpt5_S0", np.nan)

        lever_history = metadata.get("lever_history", [])
        faithfulness_delta = np.nan
        faithfulness_start = np.nan
        faithfulness_end = np.nan
        if lever_history:
            faithfulness_start = float(lever_history[0].get("faithfulness", 0.0))
            faithfulness_end = float(lever_history[-1].get("faithfulness", 0.0))
            faithfulness_delta = faithfulness_end - faithfulness_start

        records.append(
            {
                "lecture": lecture_dir.name,
                "final_score": float(result.get("final_score_0to1", 0.0)),
                "stop_state": _parse_stop_state(stop_reason),
                "iterations": int(metadata.get("iterations_completed", 0)),
                "pairwise_refined_win_rate": pairwise_refined,
                "pairwise_s0_win_rate": pairwise_s0,
                "faithfulness_delta": faithfulness_delta,
                "faithfulness_start": faithfulness_start,
                "faithfulness_end": faithfulness_end,
            }
        )

    return records


def _draw_single_lecture_section(fig, gs, iter_steps, word_counts, quality_history, signals,
                                 final_score, comp_score, manual_score, disagreement_delta,
                                 rubric, include_drift, iter_texts, stop_state,
                                 iterations_completed, agreement_score, detected_domain,
                                 stopping_reason):
    ax_len = fig.add_subplot(gs[0, 0])
    ax_len.plot(iter_steps, word_counts, marker="o", linewidth=2)
    ax_len.set_title("Summary Length Across Iterations", fontsize=15)
    ax_len.set_xlabel("Iteration")
    ax_len.set_ylabel("Word Count")
    ax_len.grid(True)
    ax_len.tick_params(axis="both", labelsize=11)

    ax_quality = fig.add_subplot(gs[0, 1])
    if quality_history:
        qx = list(range(1, len(quality_history) + 1))
        ax_quality.plot(qx, quality_history, marker="o", linewidth=2, color="#1f77b4")
        ax_quality.set_xticks(qx)
        ax_quality.set_ylim(0, 1)
        ax_quality.set_title("Quality Score Trajectory", fontsize=15)
        ax_quality.set_xlabel("Iteration")
        ax_quality.set_ylabel("Quality Score")
    else:
        ax_quality.text(0.5, 0.5, "No quality history in metadata", ha="center", va="center")
        ax_quality.set_title("Quality Score Trajectory", fontsize=15)
    ax_quality.grid(True)
    ax_quality.tick_params(axis="both", labelsize=11)

    ax_signals = fig.add_subplot(gs[1, 0])
    sig_names = ["Length Error", "Coverage", "Glossary Recall", "Hallucination"]
    sig_vals = [
        signals.get("length_error", 0.0),
        signals.get("section_coverage_pct", 0.0),
        signals.get("glossary_recall", 0.0),
        signals.get("suspected_hallucination_rate", 0.0),
    ]
    ax_signals.bar(sig_names, sig_vals, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"])
    ax_signals.set_ylim(0, 1)
    ax_signals.set_title("Final Deterministic Signals", fontsize=15)
    ax_signals.grid(axis="y")
    ax_signals.tick_params(axis="both", labelsize=11)

    ax_scores = fig.add_subplot(gs[1, 1])
    ax_scores.bar(["Final", "Comprehensive", "Manual"], [final_score, comp_score, manual_score],
                  color=["#2ca02c", "#1f77b4", "#ff7f0e"])
    ax_scores.set_ylim(0, 1)
    ax_scores.set_title(f"Score Components (Δ C-M: {disagreement_delta:.3f})", fontsize=15)
    ax_scores.grid(axis="y")
    ax_scores.tick_params(axis="both", labelsize=11)

    ax_radar = fig.add_subplot(gs[2, 0], polar=True)
    categories = ["Coverage", "Faithfulness", "Organization", "Clarity", "Style"]
    values = [
        float(rubric.get("coverage", 0)),
        float(rubric.get("faithfulness", 0)),
        float(rubric.get("organization", 0)),
        float(rubric.get("clarity", 0)),
        float(rubric.get("style", 0)),
    ]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values))
    ax_radar.plot(angles, values, linewidth=2)
    ax_radar.fill(angles, values, alpha=0.3)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 5)
    ax_radar.set_title("Final Rubric Radar", pad=18, fontsize=15)
    ax_radar.tick_params(axis="both", labelsize=11)

    ax_diag = fig.add_subplot(gs[2, 1])
    if include_drift:
        try:
            names, drifts = compute_semantic_drift(iter_texts)
            ax_diag.plot(names, drifts, marker="o", color="#9467bd", linewidth=2)
            ax_diag.set_title("Semantic Drift (1 - cosine to first iteration)")
            ax_diag.set_xlabel("Iteration")
            ax_diag.set_ylabel("Drift")
            ax_diag.grid(True)
        except Exception as exc:
            ax_diag.text(0.5, 0.5, f"Drift unavailable: {exc}", ha="center", va="center", wrap=True)
            ax_diag.set_title("Semantic Drift")
            ax_diag.axis("off")
    else:
        wrapped_reason = textwrap.fill(stopping_reason or "N/A", width=52)
        diagnostics = [
            f"Stop state: {stop_state}",
            f"Iterations completed: {iterations_completed}",
            f"Agreement (1-5): {agreement_score:.2f}",
            f"Detected domain: {detected_domain}",
            "Stopping reason:",
            wrapped_reason,
        ]
        ax_diag.axis("off")
        ax_diag.set_title("Run Diagnostics", fontsize=15)
        ax_diag.text(0.02, 0.98, "\n".join(diagnostics), va="top", ha="left", fontsize=12.5)


def _draw_cohort_section(fig, gs, records, start_row: int):
    lectures = [r["lecture"] for r in records]
    final_scores = [r["final_score"] for r in records]
    stop_states = [r["stop_state"] for r in records]
    refined_wins = [r["pairwise_refined_win_rate"] for r in records]
    s0_wins = [r["pairwise_s0_win_rate"] for r in records]

    stop_colors = {
        "pass": "#2ca02c",
        "borderline": "#1f77b4",
        "stalled": "#ff7f0e",
        "max_iters": "#d62728",
        "unknown": "#7f7f7f",
    }
    bar_colors = [stop_colors.get(s, "#7f7f7f") for s in stop_states]

    ax_table = fig.add_subplot(gs[start_row, 0])
    ax_table.axis("off")
    table_cols = ["Lecture", "Final", "Stop", "Iters", "Pairwise Win"]
    table_rows = []
    for r in records:
        win = r["pairwise_refined_win_rate"]
        win_txt = f"{win:.2f}" if not np.isnan(win) else "N/A"
        table_rows.append([r["lecture"], f"{r['final_score']:.3f}", r["stop_state"], str(r["iterations"]), win_txt])
    table = ax_table.table(cellText=table_rows, colLabels=table_cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.22)
    ax_table.set_title("Cross-Lecture Comparison", pad=10, fontsize=15)

    ax_scores = fig.add_subplot(gs[start_row, 1])
    ax_scores.bar(lectures, final_scores, color=bar_colors)
    ax_scores.set_ylim(0, 1)
    ax_scores.set_title("Final Score by Lecture", fontsize=15)
    ax_scores.set_xlabel("Lecture")
    ax_scores.set_ylabel("Final Score (0-1)")
    ax_scores.grid(axis="y")
    ax_scores.tick_params(axis="both", labelsize=11)

    ax_ablation = fig.add_subplot(gs[start_row + 1, 0])
    x = np.arange(len(lectures))
    width = 0.36
    ax_ablation.bar(x - width / 2, np.nan_to_num(s0_wins, nan=0.0), width, label="S0 Baseline", color="#7f7f7f")
    ax_ablation.bar(x + width / 2, np.nan_to_num(refined_wins, nan=0.0), width, label="Lever/Hybrid", color="#2ca02c")
    ax_ablation.set_xticks(x)
    ax_ablation.set_xticklabels(lectures)
    ax_ablation.set_ylim(0, 1)
    ax_ablation.set_title("Ablation Mini-Chart (Pairwise Win Rate)", fontsize=15)
    ax_ablation.set_xlabel("Lecture")
    ax_ablation.set_ylabel("Win Rate")
    ax_ablation.legend(loc="upper right")
    ax_ablation.grid(axis="y")
    ax_ablation.tick_params(axis="both", labelsize=11)

    ax_conv = fig.add_subplot(gs[start_row + 1, 1])
    unique_states = sorted(set(stop_states))
    counts = [stop_states.count(s) for s in unique_states]
    colors = [stop_colors.get(s, "#7f7f7f") for s in unique_states]
    ax_conv.pie(counts, labels=unique_states, autopct="%1.0f%%", startangle=90, colors=colors)
    ax_conv.set_title("Convergence Panel: Stop-State Distribution", fontsize=15)

    avg_final = float(np.mean(final_scores)) if final_scores else 0.0
    avg_iters = float(np.mean([r["iterations"] for r in records])) if records else 0.0
    valid_refined = [v for v in refined_wins if not np.isnan(v)]
    avg_refined = float(np.mean(valid_refined)) if valid_refined else 0.0

    valid_faith = [r for r in records if not np.isnan(r["faithfulness_delta"])]
    if valid_faith:
        top = max(valid_faith, key=lambda r: r["faithfulness_delta"])
        callout = (
            f"Error-analysis callout: {top['lecture']} faithfulness improved "
            f"{top['faithfulness_start']:.2f} → {top['faithfulness_end']:.2f} "
            f"(Δ={top['faithfulness_delta']:+.2f})."
        )
    else:
        callout = "Error-analysis callout: faithfulness trend data unavailable."

    takeaways = (
        f"Takeaways:\n"
        f"• Avg final score across {len(records)} lectures: {avg_final:.3f}\n"
        f"• Avg refinement depth: {avg_iters:.1f} iterations\n"
        f"• Lever/Hybrid avg pairwise win rate vs S0: {avg_refined:.2%}\n"
        f"• {callout}"
    )
    fig.text(0.02, 0.008, takeaways, fontsize=10.8, va="bottom", ha="left")


def plot_dashboard(iter_dir, result_json_path, include_drift=False, cohort_records=None, merged=False):
    iter_dir = Path(iter_dir)
    result_json_path = Path(result_json_path)

    iter_texts = load_iteration_summaries(iter_dir)
    iter_lengths = {k: len(v.split()) for k, v in iter_texts.items()}
    iter_steps = sorted(iter_lengths.keys(), key=_iteration_sort_key)
    word_counts = [iter_lengths[k] for k in iter_steps]

    result = json.loads(result_json_path.read_text(encoding="utf-8"))
    signals = result.get("signals", {})
    rubric = result.get("rubric", {})
    agreement_score = result.get("agreement", {}).get("agreement_1to5", 0.0)
    final_score = result.get("final_score_0to1", 0.0)

    comprehensive = result.get("comprehensive_scoring", {})
    comp_score = comprehensive.get("final_score", 0.0)
    hybrid = result.get("hybrid_scoring", {})
    manual_score = hybrid.get("manual_weighted_score", 0.0)
    disagreement_delta = hybrid.get("scorer_disagreement_delta", abs(comp_score - manual_score))

    metadata = result.get("refinement_metadata", {})
    quality_history = metadata.get("quality_history", [])
    stopping_reason = metadata.get("stopping_reason", "")
    stop_state = _parse_stop_state(stopping_reason)
    iterations_completed = metadata.get("iterations_completed", 0)
    detected_domain = comprehensive.get("detected_domain", "unknown")

    has_cohort = bool(cohort_records)

    if merged:
        if has_cohort:
            fig = plt.figure(figsize=(20, 27))
            gs = fig.add_gridspec(
                6,
                2,
                hspace=0.92,
                wspace=0.30,
                height_ratios=[1.25, 1.25, 1.60, 0.45, 1.40, 1.40],
            )
            fig.suptitle("Evaluation Dashboard (Current Pipeline + Cohort Insights)", fontsize=24, weight="bold", y=0.992)
            ax_section = fig.add_subplot(gs[3, :])
            ax_section.axis("off")
            ax_section.text(0.5, 0.5, "Cohort Insights (example_run)", fontsize=20, weight="bold", ha="center", va="center")
        else:
            fig = plt.figure(figsize=(16, 15))
            gs = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.28)
            fig.suptitle("Evaluation Dashboard (Current Pipeline)", fontsize=20, weight="bold", y=0.98)

        _draw_single_lecture_section(
            fig, gs,
            iter_steps, word_counts, quality_history, signals,
            final_score, comp_score, manual_score, disagreement_delta,
            rubric, include_drift, iter_texts,
            stop_state, iterations_completed, agreement_score, detected_domain,
            stopping_reason,
        )

        if has_cohort:
            _draw_cohort_section(fig, gs, cohort_records, start_row=4)
            plt.tight_layout(rect=[0, 0.065, 1, 0.975])
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.965])

        plt.show()
        return

    # Default: cleaner split windows
    fig1 = plt.figure(figsize=(16, 15))
    gs1 = fig1.add_gridspec(3, 2, hspace=0.48, wspace=0.28)
    fig1.suptitle("Evaluation Dashboard (Current Pipeline)", fontsize=21, weight="bold", y=0.985)
    _draw_single_lecture_section(
        fig1, gs1,
        iter_steps, word_counts, quality_history, signals,
        final_score, comp_score, manual_score, disagreement_delta,
        rubric, include_drift, iter_texts,
        stop_state, iterations_completed, agreement_score, detected_domain,
        stopping_reason,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    if has_cohort:
        fig2 = plt.figure(figsize=(18, 13))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.48, wspace=0.30)
        fig2.suptitle("Cohort Insights (example_run)", fontsize=22, weight="bold", y=0.985)
        _draw_cohort_section(fig2, gs2, cohort_records, start_row=0)
        plt.tight_layout(rect=[0, 0.055, 1, 0.965])
        plt.show()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Render static evaluation dashboard")
    parser.add_argument("lecture_id", nargs="?", default="lecture1")
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Compute semantic drift using OpenAI embeddings (slower, API cost)",
    )
    parser.add_argument(
        "--no-cohort",
        action="store_true",
        help="Disable cohort insights view from example_run data",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Render single merged figure (default uses cleaner split windows)",
    )
    args = parser.parse_args()

    iter_dir = f"data/summaries/refined_iterations/{args.lecture_id}"
    result_json = f"{iter_dir}/result.json"

    cohort_records = None
    if not args.no_cohort:
        example_root = Path("example_run")
        cohort_records = load_cohort_records(example_root) if example_root.exists() else []

    print(f"\nLoading dashboard for {args.lecture_id}")
    print(f"  Iter dir: {iter_dir}")
    print(f"  Result JSON: {result_json}")
    print(f"  Drift enabled: {args.drift}")
    print(f"  Cohort enabled: {not args.no_cohort}\n")

    plot_dashboard(
        iter_dir=iter_dir,
        result_json_path=result_json,
        include_drift=args.drift,
        cohort_records=cohort_records,
        merged=args.merged,
    )


if __name__ == "__main__":
    main()
