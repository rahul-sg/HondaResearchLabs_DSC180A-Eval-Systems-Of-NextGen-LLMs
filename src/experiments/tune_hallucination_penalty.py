import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "data" / "summaries" / "refined_iterations"
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "hallucination_tuning" / "report.json"
DEFAULT_HUMAN_LABELS_PATH = ROOT / "outputs" / "hallucination_tuning" / "human_faithfulness_labels.json"


@dataclass
class Record:
    sample_id: str
    lecture_id: str
    detected_domain: str
    comprehensive_score: float
    manual_base_without_penalty: float
    hallucination_rate: float
    current_final_score: float
    faithfulness_1to5: float
    human_faithfulness_1to5: float | None = None


def _safe_float(value, fallback=0.0):
    try:
        return float(value)
    except Exception:
        return fallback


def _load_human_labels(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: dict[str, float] = {}
    for lecture_id, value in data.items():
        labels[str(lecture_id)] = _safe_float(value, fallback=0.0)
    return labels


def load_records(results_dir: Path, human_labels_path: Path | None = None) -> list[Record]:
    human_labels = _load_human_labels(human_labels_path)
    records: list[Record] = []
    for lecture_dir in sorted(results_dir.iterdir()):
        if not lecture_dir.is_dir():
            continue

        result_path = lecture_dir / "result.json"
        if not result_path.exists():
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hybrid = data.get("hybrid_scoring", {})
        comprehensive = _safe_float(hybrid.get("comprehensive_score"))
        current_final = _safe_float(data.get("final_score_0to1"))
        hallucination = _safe_float(data.get("signals", {}).get("suspected_hallucination_rate"), fallback=1.0)
        detected_domain = str(data.get("rubric", {}).get("detected_domain", "humanities"))

        policy = hybrid.get("hallucination_policy", {})
        recorded_beta = _safe_float(policy.get("subtractive_beta"), fallback=0.0)

        # Recover manual score BEFORE subtractive hallucination penalty.
        # Current implementation: manual = (0.8*base + 0.2*coverage) - beta*(2^H)
        current_manual = _safe_float(hybrid.get("manual_weighted_score"))
        manual_base_no_penalty = current_manual + recorded_beta * (2 ** hallucination)

        faithfulness = _safe_float(data.get("rubric", {}).get("faithfulness"), fallback=0.0)

        records.append(
            Record(
                lecture_id=lecture_dir.name,
                sample_id=lecture_dir.name,
                detected_domain=detected_domain,
                comprehensive_score=comprehensive,
                manual_base_without_penalty=manual_base_no_penalty,
                hallucination_rate=hallucination,
                current_final_score=current_final,
                faithfulness_1to5=faithfulness,
                human_faithfulness_1to5=human_labels.get(lecture_dir.name),
            )
        )

    return records


def _looks_like_calibration_dataset(path: Path) -> bool:
    if not path.exists() or path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, list) or not data:
        return False
    sample = data[0]
    needed = {
        "sample_id",
        "lecture_id",
        "human_faithfulness_1to5",
        "comprehensive_score",
        "manual_base_without_penalty",
        "hallucination_rate",
    }
    return isinstance(sample, dict) and needed.issubset(sample.keys())


def load_calibration_records(path: Path) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    for row in data:
        records.append(
            Record(
                sample_id=str(row.get("sample_id", row.get("lecture_id", "unknown"))),
                lecture_id=str(row.get("lecture_id", "unknown")),
                detected_domain=str(row.get("detected_domain", "humanities")),
                comprehensive_score=_safe_float(row.get("comprehensive_score")),
                manual_base_without_penalty=_safe_float(row.get("manual_base_without_penalty")),
                hallucination_rate=_safe_float(row.get("hallucination_rate"), fallback=1.0),
                current_final_score=0.0,
                faithfulness_1to5=_safe_float(row.get("human_faithfulness_1to5"), fallback=0.0),
                human_faithfulness_1to5=_safe_float(row.get("human_faithfulness_1to5"), fallback=0.0),
            )
        )
    return records


def spearman_rank_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        out = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                out[indexed[k][0]] = avg_rank
            i = j + 1
        return out

    rx = ranks(xs)
    ry = ranks(ys)
    mx = mean(rx)
    my = mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def score_policy(records: list[Record], alpha: float, beta: float, use_human_labels: bool = False) -> dict:
    finals = []
    comps = []
    halls = []
    faithfulness = []
    penalties = []

    domain_alpha_multiplier_map = {
        "engineering": 1.15,
        "math": 1.10,
        "natural_sciences": 1.05,
        "social_sciences": 1.00,
        "business": 0.95,
        "humanities": 0.90,
    }
    min_damping_factor = 0.75
    max_damping_factor = 1.0

    for r in records:
        manual = r.manual_base_without_penalty - beta * (2 ** r.hallucination_rate)
        blended = 0.7 * r.comprehensive_score + 0.3 * manual
        mult = domain_alpha_multiplier_map.get(r.detected_domain, 1.0)
        effective_alpha = alpha * mult
        raw_damp = 1.0 - effective_alpha * r.hallucination_rate
        damp = max(min_damping_factor, min(max_damping_factor, raw_damp))
        final = blended * damp

        finals.append(final)
        comps.append(r.comprehensive_score)
        halls.append(r.hallucination_rate)
        if use_human_labels and r.human_faithfulness_1to5 is not None:
            faithfulness.append(r.human_faithfulness_1to5)
        else:
            faithfulness.append(r.faithfulness_1to5)
        penalties.append(max(0.0, blended - final))

    mean_final = mean(finals) if finals else 0.0
    mean_penalty = mean(penalties) if penalties else 0.0
    spearman_comp = spearman_rank_corr(finals, comps)
    spearman_faith = spearman_rank_corr(finals, faithfulness)
    # More negative is better: high hallucination -> lower score.
    spearman_hall = spearman_rank_corr(finals, halls)

    if use_human_labels:
        objective = (0.35 * spearman_comp) + (0.50 * spearman_faith) + (0.15 * (-spearman_hall))
    else:
        # Composite objective (proxy, not ground truth):
        # keep alignment with rubric while preserving anti-hallucination monotonicity
        objective = (0.45 * spearman_comp) + (0.35 * spearman_faith) + (0.20 * (-spearman_hall))

    return {
        "alpha": alpha,
        "beta": beta,
        "objective": objective,
        "mean_final_score": mean_final,
        "mean_penalty": mean_penalty,
        "spearman_vs_comprehensive": spearman_comp,
        "spearman_vs_faithfulness": spearman_faith,
        "spearman_vs_hallucination": spearman_hall,
        "using_human_labels": use_human_labels,
    }


def run_grid(records: list[Record], use_human_labels: bool = False) -> list[dict]:
    alphas = [round(x, 3) for x in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]]
    betas = [round(x, 3) for x in [0.0, 0.025, 0.05, 0.075, 0.10, 0.125]]

    out = []
    for alpha in alphas:
        for beta in betas:
            out.append(score_policy(records, alpha=alpha, beta=beta, use_human_labels=use_human_labels))
    out.sort(key=lambda d: d["objective"], reverse=True)
    return out


def summarize(records: list[Record], ranked: list[dict], use_human_labels: bool = False) -> dict:
    current_policy = score_policy(records, alpha=0.15, beta=0.10, use_human_labels=use_human_labels)
    best = ranked[0] if ranked else current_policy

    recommendation = {
        "recommended_alpha": best["alpha"],
        "recommended_beta": best["beta"],
        "rationale": (
            "Maximizes objective using human faithfulness labels."
            if use_human_labels
            else "Maximizes proxy objective balancing rubric alignment and anti-hallucination monotonicity."
        ),
    }

    labeled_count = sum(1 for r in records if r.human_faithfulness_1to5 is not None)

    return {
        "n_samples": len(records),
        "n_lectures": len({r.lecture_id for r in records}),
        "n_human_labeled_samples": labeled_count,
        "using_human_labels": use_human_labels,
        "lecture_ids": sorted({r.lecture_id for r in records}),
        "current_policy": current_policy,
        "recommended_policy": best,
        "recommendation": recommendation,
        "top_10_policies": ranked[:10],
    }


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HUMAN_LABELS_PATH
    if _looks_like_calibration_dataset(input_path):
        records = load_calibration_records(input_path)
        labels_path = input_path
    else:
        labels_path = input_path
        records = load_records(DEFAULT_RESULTS_DIR, human_labels_path=labels_path)

    if not records:
        raise RuntimeError(f"No result.json files found under: {DEFAULT_RESULTS_DIR}")

    labeled_count = sum(1 for r in records if r.human_faithfulness_1to5 is not None)
    use_human_labels = labeled_count >= 3

    ranked = run_grid(records, use_human_labels=use_human_labels)
    report = summarize(records, ranked, use_human_labels=use_human_labels)

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("===== HALLUCINATION PENALTY TUNING =====")
    print(f"Samples analyzed: {report['n_samples']}")
    print(f"Lectures analyzed: {report['n_lectures']}")
    print(f"Human-labeled samples: {report['n_human_labeled_samples']}")
    print(f"Current policy (alpha=0.15, beta=0.10): objective={report['current_policy']['objective']:.4f}")
    print(
        "Recommended policy "
        f"(alpha={report['recommended_policy']['alpha']}, beta={report['recommended_policy']['beta']}): "
        f"objective={report['recommended_policy']['objective']:.4f}"
    )
    print(f"Report saved to: {DEFAULT_OUTPUT_PATH}")
    if not use_human_labels:
        print(
            "Tip: run full labeled calibration with:\n"
            "1) python -m src.experiments.prepare_faithfulness_labeling_set\n"
            "2) fill human_faithfulness_1to5 in outputs/hallucination_tuning/human_labeling_candidates.jsonl\n"
            "3) python -m src.experiments.build_human_calibration_dataset\n"
            "4) python -m src.experiments.tune_hallucination_penalty outputs/hallucination_tuning/human_calibration_dataset.json"
        )


if __name__ == "__main__":
    main()
