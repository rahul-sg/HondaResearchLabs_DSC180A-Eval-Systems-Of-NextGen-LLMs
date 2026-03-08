import json
import sys
from pathlib import Path

from src.evaluation.scoring import combine_scores, compute_comprehensive_score
from src.models.judge import judge_rubric_ensemble
from src.models.llm_client import LLMConfig
from src.utils.filter_slides import filter_content_slides
from src.utils.io import load_slides
from src.utils.signals import compute_signals


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_PATH = ROOT / "outputs" / "hallucination_tuning" / "human_labeling_candidates.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "hallucination_tuning" / "human_calibration_dataset.json"


def _safe_float(value, fallback=None):
    try:
        return float(value)
    except Exception:
        return fallback


def _load_labeled_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = _safe_float(row.get("human_faithfulness_1to5"), fallback=None)
            if label is None:
                continue
            if not (1.0 <= label <= 5.0):
                continue
            rows.append(row)
    return rows


def _score_row(row: dict, cfg_judge: LLMConfig) -> dict:
    lecture_id = row["lecture_id"]
    sample_id = row["sample_id"]
    source = row.get("source", "unknown")
    summary = row["summary"]
    human_faith = float(row["human_faithfulness_1to5"])

    slide_path = ROOT / "data" / "slides" / f"{lecture_id}.pdf"
    if not slide_path.exists():
        raise FileNotFoundError(f"Missing slides for {lecture_id}: {slide_path}")

    slides_dict = load_slides(str(slide_path))
    slides_full = slides_dict["slides"]
    slides_content = filter_content_slides(slides_full)

    signals = compute_signals(slides_content, summary, target_words=350)
    rubric = judge_rubric_ensemble(
        slides_full,
        summary,
        cfg_judge,
        runs=1,
        use_domain_aware=True,
    )
    comprehensive = compute_comprehensive_score(rubric_result=rubric, agreement_result=None, meteor_score=None)

    base_score = combine_scores(rubric, agreement=None)
    manual_base_without_penalty = (0.8 * base_score) + (0.2 * signals["section_coverage_pct"])

    return {
        "sample_id": sample_id,
        "lecture_id": lecture_id,
        "source": source,
        "human_faithfulness_1to5": human_faith,
        "detected_domain": comprehensive.get("detected_domain", rubric.get("detected_domain", "unknown")),
        "comprehensive_score": float(comprehensive["final_score"]),
        "manual_base_without_penalty": float(manual_base_without_penalty),
        "hallucination_rate": float(signals["suspected_hallucination_rate"]),
        "section_coverage_pct": float(signals["section_coverage_pct"]),
        "raw_rubric": {
            "coverage": rubric.get("coverage", 0),
            "faithfulness": rubric.get("faithfulness", 0),
            "organization": rubric.get("organization", 0),
            "clarity": rubric.get("clarity", 0),
            "style": rubric.get("style", 0),
        },
    }


def main():
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LABELS_PATH
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH

    labeled_rows = _load_labeled_rows(labels_path)
    if not labeled_rows:
        raise RuntimeError(
            "No labeled rows found. Fill `human_faithfulness_1to5` in the labeling JSONL first."
        )

    cfg_judge = LLMConfig(model="gpt-5-chat-latest", max_completion_tokens=512)

    dataset = []
    for idx, row in enumerate(labeled_rows, 1):
        print(f"Scoring labeled sample {idx}/{len(labeled_rows)}: {row.get('sample_id')}")
        dataset.append(_score_row(row, cfg_judge))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("===== BUILD HUMAN CALIBRATION DATASET =====")
    print(f"Labeled samples processed: {len(dataset)}")
    print(f"Output: {out_path}")
    print("Next: python -m src.experiments.tune_hallucination_penalty <path_to_this_output_json>")


if __name__ == "__main__":
    main()
