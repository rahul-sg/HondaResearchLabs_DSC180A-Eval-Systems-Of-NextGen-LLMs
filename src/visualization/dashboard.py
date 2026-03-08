import argparse
import json
import re
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from dotenv import load_dotenv


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


def _safe_float(val, default=np.nan):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_selected_iteration(stopping_reason: str):
    if not stopping_reason:
        return None
    m = re.search(r"final_selection:\s*best_of_last_\d+\s*\(iter\s+(\d+)", stopping_reason)
    if m:
        return int(m.group(1))
    return None


def load_iteration_summaries(iter_dir: Path):
    texts = {}
    for fname in iter_dir.glob("*.txt"):
        texts[fname.stem] = fname.read_text(encoding="utf-8").strip()
    return texts


def _extract_iteration_metrics(result: dict):
    metadata = result.get("refinement_metadata", {})
    if isinstance(metadata.get("iteration_metrics"), list) and metadata["iteration_metrics"]:
        return metadata["iteration_metrics"]
    if isinstance(result.get("iteration_score_table"), list) and result["iteration_score_table"]:
        return result["iteration_score_table"]
    return []


def _iteration_lookup(iteration_metrics):
    lookup = {}
    for row in iteration_metrics:
        it = row.get("iteration")
        if it is not None:
            lookup[int(it)] = row
    return lookup


def _draw_numeric_heatmap(ax, matrix, row_labels, col_labels, title, cmap="viridis", center_zero=False, fmt="{:.2f}"):
    matrix = np.array(matrix, dtype=float)

    if matrix.size == 0:
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=10)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    finite_vals = matrix[np.isfinite(matrix)]
    if finite_vals.size == 0:
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=10)
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
        return

    if center_zero:
        vabs = max(abs(np.nanmin(finite_vals)), abs(np.nanmax(finite_vals)))
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs) if vabs > 0 else None
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=13, pad=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=9, color="black")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def load_cohort_records(cohort_root: Path):
    records = []
    if not cohort_root.exists():
        return records

    rubric_dims = ["coverage", "faithfulness", "organization", "clarity", "style"]

    for lecture_dir in sorted([p for p in cohort_root.glob("lecture*") if p.is_dir()]):
        result_path = lecture_dir / "result.json"
        if not result_path.exists():
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = result.get("refinement_metadata", {})
        stop_reason = metadata.get("stopping_reason", "")
        iteration_metrics = _extract_iteration_metrics(result)
        iteration_map = _iteration_lookup(iteration_metrics)

        selected_iter = _extract_selected_iteration(stop_reason)
        if selected_iter is None and iteration_metrics:
            selected_iter = int(iteration_metrics[-1].get("iteration", len(iteration_metrics)))

        first_iter_row = iteration_map.get(1, iteration_metrics[0] if iteration_metrics else {})
        selected_iter_row = iteration_map.get(selected_iter, iteration_metrics[-1] if iteration_metrics else {})

        first_quality = _safe_float(first_iter_row.get("quality_score", np.nan))
        selected_quality = _safe_float(selected_iter_row.get("quality_score", np.nan))

        first_signals = first_iter_row.get("signals", {}) if isinstance(first_iter_row.get("signals", {}), dict) else {}
        selected_signals = selected_iter_row.get("signals", {}) if isinstance(selected_iter_row.get("signals", {}), dict) else {}

        first_rubric = first_iter_row.get("rubric", {}) if isinstance(first_iter_row.get("rubric", {}), dict) else {}
        selected_rubric = selected_iter_row.get("rubric", {}) if isinstance(selected_iter_row.get("rubric", {}), dict) else {}

        detected_domain = result.get("comprehensive_scoring", {}).get("detected_domain", "unknown")

        record = {
            "lecture": lecture_dir.name,
            "selected_iteration": selected_iter,
            "selected_quality": selected_quality,
            "first_quality": first_quality,
            "quality_delta": (
                selected_quality - first_quality
                if not np.isnan(first_quality) and not np.isnan(selected_quality)
                else np.nan
            ),
            "iterations": int(metadata.get("iterations_completed", 0)),
            "stop_state": _parse_stop_state(stop_reason),
            "detected_domain": str(detected_domain).lower(),
            "length_error": _safe_float(
                selected_signals.get("length_error", result.get("signals", {}).get("length_error", np.nan))
            ),
            "section_coverage_pct": _safe_float(
                selected_signals.get("section_coverage_pct", result.get("signals", {}).get("section_coverage_pct", np.nan))
            ),
            "glossary_recall": _safe_float(
                selected_signals.get("glossary_recall", result.get("signals", {}).get("glossary_recall", np.nan))
            ),
            "suspected_hallucination_rate": _safe_float(
                selected_signals.get(
                    "suspected_hallucination_rate",
                    result.get("signals", {}).get("suspected_hallucination_rate", np.nan),
                )
            ),
        }

        for d in rubric_dims:
            first_val = _safe_float(first_rubric.get(d, np.nan))
            selected_val = _safe_float(selected_rubric.get(d, np.nan))
            record[f"{d}_first"] = first_val
            record[f"{d}_selected"] = selected_val
            record[f"{d}_delta"] = (
                selected_val - first_val
                if not np.isnan(first_val) and not np.isnan(selected_val)
                else np.nan
            )

        records.append(record)

    return records


def _draw_rubric_small_multiples(fig, parent_spec, records):
    dims = [
        ("coverage", "Coverage"),
        ("faithfulness", "Faithfulness"),
        ("organization", "Organization"),
    ]

    subgs = parent_spec.subgridspec(1, 3, wspace=0.45)

    lectures = [r["lecture"] for r in records]
    x = np.arange(len(lectures))
    width = 0.36

    for idx, (dim_key, dim_label) in enumerate(dims):
        ax = fig.add_subplot(subgs[0, idx])

        first_vals = np.array(
            [_safe_float(r.get(f"{dim_key}_first", np.nan), 0.0) for r in records],
            dtype=float,
        )
        selected_vals = np.array(
            [_safe_float(r.get(f"{dim_key}_selected", np.nan), 0.0) for r in records],
            dtype=float,
        )

        ax.bar(x - width / 2, first_vals, width, label="First")
        ax.bar(x + width / 2, selected_vals, width, label="Selected")

        ax.set_title(dim_label, fontsize=10, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(lectures, rotation=32, ha="right", fontsize=8)
        ax.set_ylim(3, 5)
        ax.set_yticks([3, 4, 5])
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        if idx == 0:
            ax.set_ylabel("Rubric Score", fontsize=10)
        else:
            ax.set_yticklabels([])

        if idx == 1:
            ax.legend(loc="upper center", fontsize=8, frameon=True)


def _draw_quality_waterfall(ax, records):
    valid = [r for r in records if np.isfinite(_safe_float(r.get("quality_delta", np.nan)))]

    if not valid:
        ax.axis("off")
        ax.set_title("Quality Delta Waterfall", fontsize=15)
        ax.text(0.5, 0.5, "No quality delta data", ha="center", va="center")
        return

    lectures = [r["lecture"] for r in valid]
    deltas = np.array([r["quality_delta"] for r in valid], dtype=float)
    x = np.arange(len(lectures))

    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    ax.bar(x, deltas, color=colors)

    for xi, d in zip(x, deltas):
        ax.text(
            xi,
            d + (0.002 if d >= 0 else -0.002),
            f"{d:+.3f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=9,
        )

    dmin = float(np.min(deltas))
    dmax = float(np.max(deltas))
    span = max(dmax - dmin, 0.01)
    pad = max(0.005, 0.25 * span)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(lectures, rotation=35, ha="right")
    ax.set_ylim(dmin - pad, dmax + pad)
    ax.set_ylabel("Selected Quality - First Quality")
    ax.set_title("Quality Delta Waterfall by Lecture", fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def _build_individual_record(result: dict):
    metadata = result.get("refinement_metadata", {})
    stop_reason = metadata.get("stopping_reason", "")
    iteration_metrics = _extract_iteration_metrics(result)
    iteration_map = _iteration_lookup(iteration_metrics)

    selected_iter = _extract_selected_iteration(stop_reason)
    if selected_iter is None and iteration_metrics:
        selected_iter = int(iteration_metrics[-1].get("iteration", len(iteration_metrics)))

    first_row = iteration_map.get(1, iteration_metrics[0] if iteration_metrics else {})
    selected_row = iteration_map.get(selected_iter, iteration_metrics[-1] if iteration_metrics else {})

    first_signals = first_row.get("signals", {}) if isinstance(first_row.get("signals", {}), dict) else {}
    selected_signals = selected_row.get("signals", {}) if isinstance(selected_row.get("signals", {}), dict) else {}

    first_quality = _safe_float(first_row.get("quality_score", np.nan))
    selected_quality = _safe_float(selected_row.get("quality_score", np.nan))
    target_words = _safe_float(metadata.get("target_words", np.nan))

    hybrid = result.get("hybrid_scoring", {})
    comprehensive = result.get("comprehensive_scoring", {})

    raw_quality = _safe_float(hybrid.get("raw_quality_score", np.nan))
    risk_adjusted = _safe_float(hybrid.get("risk_adjusted_score", np.nan))
    comprehensive_score = _safe_float(comprehensive.get("final_score", np.nan))
    manual_score = _safe_float(hybrid.get("manual_weighted_score", np.nan))
    final_score = _safe_float(result.get("final_score_0to1", np.nan))
    damping_factor = _safe_float(hybrid.get("hallucination_damping_factor", np.nan))
    halluc_rate = _safe_float(hybrid.get("hallucination_rate", np.nan))
    damping_drop = (
        raw_quality - risk_adjusted
        if np.isfinite(raw_quality) and np.isfinite(risk_adjusted)
        else np.nan
    )

    return {
        "selected_iteration": selected_iter,
        "first_quality": first_quality,
        "selected_quality": selected_quality,
        "quality_delta": (
            selected_quality - first_quality
            if np.isfinite(first_quality) and np.isfinite(selected_quality)
            else np.nan
        ),
        "first_signals": {
            "length_error": _safe_float(first_signals.get("length_error", np.nan)),
            "section_coverage_pct": _safe_float(first_signals.get("section_coverage_pct", np.nan)),
            "glossary_recall": _safe_float(first_signals.get("glossary_recall", np.nan)),
            "suspected_hallucination_rate": _safe_float(first_signals.get("suspected_hallucination_rate", np.nan)),
        },
        "selected_signals": {
            "length_error": _safe_float(selected_signals.get("length_error", result.get("signals", {}).get("length_error", np.nan))),
            "section_coverage_pct": _safe_float(
                selected_signals.get("section_coverage_pct", result.get("signals", {}).get("section_coverage_pct", np.nan))
            ),
            "glossary_recall": _safe_float(
                selected_signals.get("glossary_recall", result.get("signals", {}).get("glossary_recall", np.nan))
            ),
            "suspected_hallucination_rate": _safe_float(
                selected_signals.get(
                    "suspected_hallucination_rate",
                    result.get("signals", {}).get("suspected_hallucination_rate", np.nan),
                )
            ),
        },
        "iteration_metrics": iteration_metrics,
        "target_words": target_words,
        "score_components": {
            "comprehensive": comprehensive_score,
            "manual": manual_score,
            "raw_quality": raw_quality,
            "risk_adjusted": risk_adjusted,
            "final": final_score,
            "damping_factor": damping_factor,
            "damping_drop": damping_drop,
            "hallucination_rate": halluc_rate,
        },
    }


def _draw_length_target_chart(ax, iteration_metrics, selected_iter, target_words):
    if not iteration_metrics:
        ax.axis("off")
        ax.set_title("Summary Length vs Target", fontsize=15, pad=10)
        ax.text(0.5, 0.5, "No iteration metrics", ha="center", va="center")
        return

    xs = [int(row.get("iteration", i + 1)) for i, row in enumerate(iteration_metrics)]
    ys = [_safe_float(row.get("word_count", np.nan)) for row in iteration_metrics]

    ax.plot(xs, ys, marker="o", linewidth=2, label="Word Count")

    if np.isfinite(target_words):
        ax.axhline(
            target_words,
            linestyle="--",
            linewidth=1.5,
            label=f"Target = {int(target_words)}",
        )

    if selected_iter in xs:
        idx = xs.index(selected_iter)
        ax.scatter([xs[idx]], [ys[idx]], s=90, zorder=3)

    ymin = np.nanmin(ys)
    ymax = np.nanmax(ys)
    pad = max(8, 0.08 * (ymax - ymin if ymax > ymin else 20))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title("Summary Length vs Target", fontsize=15, pad=10)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Word Count", fontsize=11)
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9, frameon=True)


def _draw_quality_signal_trend(ax, iteration_metrics, selected_iter):
    if not iteration_metrics:
        ax.axis("off")
        ax.set_title("Quality and Hallucination Across Iterations", fontsize=15, pad=10)
        ax.text(0.5, 0.5, "No iteration metrics", ha="center", va="center")
        return

    xs = [int(row.get("iteration", i + 1)) for i, row in enumerate(iteration_metrics)]
    q = [_safe_float(row.get("quality_score", np.nan)) for row in iteration_metrics]

    halluc = []
    for row in iteration_metrics:
        if isinstance(row.get("signals"), dict):
            halluc.append(_safe_float(row["signals"].get("suspected_hallucination_rate", np.nan)))
        else:
            halluc.append(_safe_float(row.get("hallucination_rate", np.nan)))

    ax.plot(xs, q, marker="o", linewidth=2, label="Quality Score")
    ax.plot(xs, halluc, marker="s", linewidth=2, linestyle="--", label="Hallucination Rate")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Score / Rate", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True)

    if selected_iter in xs:
        idx = xs.index(selected_iter)
        ax.axvline(xs[idx], linestyle=":", linewidth=1.5)

    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_title("Quality and Hallucination Across Iterations", fontsize=15, pad=10)


def _draw_score_equation_panel(ax, score_components):
    ax.axis("off")
    ax.set_title("Score Calculation Walkthrough", fontsize=16, pad=28)

    comp = score_components.get("comprehensive", np.nan)
    manual = score_components.get("manual", np.nan)
    raw = score_components.get("raw_quality", np.nan)
    risk = score_components.get("risk_adjusted", np.nan)
    final = score_components.get("final", np.nan)
    hall = score_components.get("hallucination_rate", np.nan)
    damp = score_components.get("damping_factor", np.nan)
    drop = score_components.get("damping_drop", np.nan)

    lines = []

    lines.append(f"C = comprehensive score = {comp:.3f}" if np.isfinite(comp) else "C = comprehensive score = N/A")
    lines.append(f"M = manual weighted score = {manual:.3f}" if np.isfinite(manual) else "M = manual weighted score = N/A")

    lines.append("")
    lines.append("Blend step:")
    lines.append("Raw = 0.7·C + 0.3·M")

    if np.isfinite(comp) and np.isfinite(manual) and np.isfinite(raw):
        lines.append(f"Raw = 0.7({comp:.3f}) + 0.3({manual:.3f})")
        lines.append(f"Raw = {raw:.3f}")
    else:
        lines.append("Raw = N/A")

    lines.append("")
    lines.append("Risk adjustment:")

    lines.append(f"Hallucination rate = {hall:.3f}" if np.isfinite(hall) else "Hallucination rate = N/A")
    lines.append(f"Damping factor = {damp:.3f}" if np.isfinite(damp) else "Damping factor = N/A")
    lines.append("Risk Adjusted = Raw × damping")

    if np.isfinite(raw) and np.isfinite(damp) and np.isfinite(risk):
        lines.append(f"Risk Adjusted = {raw:.3f} × {damp:.3f}")
        lines.append(f"Risk Adjusted = {risk:.3f}")
    else:
        lines.append("Risk Adjusted = N/A")

    if np.isfinite(drop):
        lines.append("")
        lines.append(f"Damping penalty = Raw - Risk Adjusted = {drop:.3f}")

    if np.isfinite(final):
        lines.append("")
        lines.append(f"Final stored score = {final:.3f}")

    ax.text(
        0.03,
        0.82,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.25,
    )


def _draw_run_diagnostics(ax, stop_state, iterations_completed, detected_domain, stopping_reason, selected_iter, quality_delta):
    wrapped_reason = textwrap.fill(stopping_reason or "N/A", width=42)
    qdelta_text = f"{quality_delta:+.3f}" if np.isfinite(quality_delta) else "N/A"
    diagnostics = [
        f"Stop state: {stop_state}",
        f"Iterations completed: {iterations_completed}",
        f"Detected domain: {detected_domain}",
        f"Selected iteration: {selected_iter}",
        f"Selected quality delta: {qdelta_text}",
        "Stopping reason:",
        wrapped_reason,
    ]
    ax.axis("off")
    ax.set_title("Run Diagnostics", fontsize=16, pad=28)
    ax.text(
        0.02,
        0.82,
        "\n".join(diagnostics),
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.25,
    )


def _draw_single_lecture_section(
    fig,
    gs,
    result,
    iter_texts,
    stop_state,
    iterations_completed,
    detected_domain,
    stopping_reason,
):
    record = _build_individual_record(result)
    iteration_metrics = record["iteration_metrics"]
    selected_iter = record["selected_iteration"]

    ax_len = fig.add_subplot(gs[0, 0])
    _draw_length_target_chart(ax_len, iteration_metrics, selected_iter, record["target_words"])

    ax_quality = fig.add_subplot(gs[0, 1])
    _draw_quality_signal_trend(ax_quality, iteration_metrics, selected_iter)

    ax_scores = fig.add_subplot(gs[1:, 0])
    _draw_score_equation_panel(ax_scores, record["score_components"])

    ax_diag = fig.add_subplot(gs[1:, 1])
    _draw_run_diagnostics(
        ax_diag,
        stop_state=stop_state,
        iterations_completed=iterations_completed,
        detected_domain=detected_domain,
        stopping_reason=stopping_reason,
        selected_iter=selected_iter,
        quality_delta=record["quality_delta"],
    )


def _draw_cohort_section(fig, gs, records, start_row: int):
    if not records:
        ax = fig.add_subplot(gs[start_row:, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "No cohort records found", ha="center", va="center", fontsize=16)
        return

    lectures = [r["lecture"] for r in records]

    signal_cols = [
        "length_error",
        "section_coverage_pct",
        "glossary_recall",
        "suspected_hallucination_rate",
    ]
    signal_labels = [
        "Length Error",
        "Coverage",
        "Glossary Recall",
        "Hallucination",
    ]
    signal_matrix = [[r.get(col, np.nan) for col in signal_cols] for r in records]

    ax_signal = fig.add_subplot(gs[start_row, 0])
    _draw_numeric_heatmap(
        ax_signal,
        signal_matrix,
        row_labels=lectures,
        col_labels=signal_labels,
        title="Selected Deterministic Signals",
        cmap="YlGnBu",
        center_zero=False,
        fmt="{:.2f}",
    )

    _draw_rubric_small_multiples(fig, gs[start_row, 1], records)

    ax_q = fig.add_subplot(gs[start_row + 1, 0])
    _draw_quality_waterfall(ax_q, records)

    ax_text = fig.add_subplot(gs[start_row + 1, 1])
    ax_text.axis("off")

    selected_qualities = np.array([_safe_float(r.get("selected_quality", np.nan)) for r in records], dtype=float)
    iterations = np.array([r.get("iterations", np.nan) for r in records], dtype=float)
    deltas = np.array([_safe_float(r.get("quality_delta", np.nan)) for r in records], dtype=float)

    avg_selected_quality = float(np.nanmean(selected_qualities)) if selected_qualities.size else np.nan
    avg_iters = float(np.nanmean(iterations)) if iterations.size else np.nan
    avg_delta = float(np.nanmean(deltas)) if deltas.size else np.nan

    stop_states = [r.get("stop_state", "unknown") for r in records]
    dominant_stop = max(set(stop_states), key=stop_states.count)

    domains = [r.get("detected_domain", "unknown") for r in records]
    domain_counts = {}
    for d in domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1
    domain_summary = ", ".join([f"{k}: {v}" for k, v in sorted(domain_counts.items())])

    best_selected_record = max(records, key=lambda r: _safe_float(r.get("selected_quality", -np.inf), -np.inf))
    best_lecture = best_selected_record["lecture"]
    best_selected_quality = _safe_float(best_selected_record.get("selected_quality", np.nan))
    best_selected_iter = best_selected_record.get("selected_iteration", "N/A")

    best_delta_record = max(
        [r for r in records if np.isfinite(_safe_float(r.get("quality_delta", np.nan)))],
        key=lambda r: r["quality_delta"],
        default=None,
    )
    best_delta_msg = "Unavailable"
    if best_delta_record is not None:
        best_delta_msg = (
            f"{best_delta_record['lecture']} "
            f"(iter {best_delta_record.get('selected_iteration', 'N/A')}, "
            f"{best_delta_record['quality_delta']:+.3f})"
        )

    summary = (
        f"Cohort Summary\n\n"
        f"Lectures analyzed: {len(records)}\n"
        f"Avg selected quality: {avg_selected_quality:.3f}\n"
        f"Avg refinement depth: {avg_iters:.1f} iterations\n"
        f"Avg selected quality delta: {avg_delta:+.3f}\n"
        f"Dominant stop state: {dominant_stop}\n"
        f"Domain mix: {domain_summary}\n\n"
        f"Top selected quality:\n"
        f"{best_lecture} (iter {best_selected_iter}, {best_selected_quality:.3f})\n\n"
        f"Largest selected quality gain:\n"
        f"{best_delta_msg}"
    )

    ax_text.text(0.02, 0.98, summary, va="top", ha="left", fontsize=12)


def plot_dashboard(iter_dir, result_json_path, cohort_records=None):
    iter_dir = Path(iter_dir)
    result_json_path = Path(result_json_path)

    iter_texts = load_iteration_summaries(iter_dir)
    result = json.loads(result_json_path.read_text(encoding="utf-8"))

    comprehensive = result.get("comprehensive_scoring", {})
    metadata = result.get("refinement_metadata", {})
    stopping_reason = metadata.get("stopping_reason", "")
    stop_state = _parse_stop_state(stopping_reason)
    iterations_completed = metadata.get("iterations_completed", 0)
    detected_domain = comprehensive.get("detected_domain", "unknown")

    fig1 = plt.figure(figsize=(18, 14))
    gs1 = fig1.add_gridspec(3, 2, hspace=0.88, wspace=0.28)
    fig1.suptitle("Evaluation Dashboard (Current Pipeline)", fontsize=21, weight="bold", y=0.985)
    _draw_single_lecture_section(
        fig1,
        gs1,
        result=result,
        iter_texts=iter_texts,
        stop_state=stop_state,
        iterations_completed=iterations_completed,
        detected_domain=detected_domain,
        stopping_reason=stopping_reason,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    if cohort_records:
        fig2 = plt.figure(figsize=(21, 14))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.60, wspace=0.42)
        fig2.suptitle("Cohort Insights (selected iterations from refined runs)", fontsize=20, weight="bold", y=0.97)
        _draw_cohort_section(fig2, gs2, cohort_records, start_row=0)
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.show()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Render static evaluation dashboard")
    parser.add_argument("lecture_id", nargs="?", default="lecture1")
    parser.add_argument(
        "--no-cohort",
        action="store_true",
        help="Disable cohort insights view from cohort data",
    )
    args = parser.parse_args()

    iter_dir = f"data/summaries/refined_iterations/{args.lecture_id}"
    result_json = f"{iter_dir}/result.json"

    cohort_records = None
    if not args.no_cohort:
        cohort_root = Path("data/summaries/refined_iterations")
        cohort_records = load_cohort_records(cohort_root) if cohort_root.exists() else []

    print(f"\nLoading dashboard for {args.lecture_id}")
    print(f"  Iter dir: {iter_dir}")
    print(f"  Result JSON: {result_json}")
    print(f"  Cohort enabled: {not args.no_cohort}\n")

    plot_dashboard(
        iter_dir=iter_dir,
        result_json_path=result_json,
        cohort_records=cohort_records,
    )


if __name__ == "__main__":
    main()