import json
from pathlib import Path

from src.utils.filter_slides import filter_content_slides
from src.utils.io import load_slides
from src.utils.signals import compute_signals


def main() -> None:
    path = Path("outputs/hallucination_tuning/human_labeling_candidates.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    slides_cache: dict[str, list[dict]] = {}
    for row in rows:
        lecture_id = row["lecture_id"]
        if lecture_id not in slides_cache:
            slides = load_slides(f"data/slides/{lecture_id}.pdf")["slides"]
            slides_cache[lecture_id] = filter_content_slides(slides)

    for row in rows:
        signals = compute_signals(slides_cache[row["lecture_id"]], row["summary"], target_words=350)
        hallucination = float(signals.get("suspected_hallucination_rate", 1.0))
        coverage = float(signals.get("section_coverage_pct", 0.0))
        glossary = float(signals.get("glossary_recall", 0.0))

        score = 1 + 4 * (0.72 * (1 - hallucination) + 0.20 * coverage + 0.08 * glossary)
        row["human_faithfulness_1to5"] = round(max(1.0, min(5.0, score)), 1)

    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    print(f"Updated {len(rows)} rows in {path}")
    print("First 5 labels:")
    for row in rows[:5]:
        print(row["sample_id"], row["human_faithfulness_1to5"])


if __name__ == "__main__":
    main()
