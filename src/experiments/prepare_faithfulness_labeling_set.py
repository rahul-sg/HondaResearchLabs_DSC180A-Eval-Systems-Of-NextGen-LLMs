import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "outputs" / "hallucination_tuning" / "human_labeling_candidates.jsonl"


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _make_variants(text: str) -> list[tuple[str, str]]:
    sents = _sentences(text)
    variants: list[tuple[str, str]] = []

    if len(sents) >= 6:
        variants.append(("variant_truncate_head_60pct", " ".join(sents[: max(1, int(0.6 * len(sents)))])))
        variants.append(("variant_drop_middle_30pct", " ".join(sents[:2] + sents[-2:])))
    elif len(sents) >= 3:
        variants.append(("variant_truncate_head_70pct", " ".join(sents[: max(1, int(0.7 * len(sents)))])))

    if len(text.split()) > 220:
        words = text.split()
        variants.append(("variant_word_truncate_75pct", " ".join(words[: int(0.75 * len(words))])))

    return [(name, _normalize_text(v)) for name, v in variants if _normalize_text(v)]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def collect_candidates() -> list[dict]:
    candidates: list[dict] = []

    # S0 candidates
    for p in sorted((ROOT / "data" / "summaries" / "model_s0").glob("lecture*.txt")):
        lecture_id = p.stem
        text = _normalize_text(_read_text(p))
        if text:
            candidates.append({
                "sample_id": f"{lecture_id}::s0",
                "lecture_id": lecture_id,
                "source": "model_s0",
                "summary": text,
                "human_faithfulness_1to5": None,
            })

    # Refined finals + optional iteration files
    refined_root = ROOT / "data" / "summaries" / "refined_iterations"
    for lecture_dir in sorted(refined_root.glob("lecture*")):
        if not lecture_dir.is_dir():
            continue
        lecture_id = lecture_dir.name

        final_path = lecture_dir / "final.txt"
        if final_path.exists():
            text = _normalize_text(_read_text(final_path))
            if text:
                candidates.append({
                    "sample_id": f"{lecture_id}::final",
                    "lecture_id": lecture_id,
                    "source": "refined_final",
                    "summary": text,
                    "human_faithfulness_1to5": None,
                })

                for variant_name, variant_text in _make_variants(text):
                    candidates.append({
                        "sample_id": f"{lecture_id}::{variant_name}",
                        "lecture_id": lecture_id,
                        "source": variant_name,
                        "summary": variant_text,
                        "human_faithfulness_1to5": None,
                    })

        for iter_path in sorted(lecture_dir.glob("iter_*.txt")):
            text = _normalize_text(_read_text(iter_path))
            if text:
                candidates.append({
                    "sample_id": f"{lecture_id}::{iter_path.stem}",
                    "lecture_id": lecture_id,
                    "source": "refined_iteration",
                    "summary": text,
                    "human_faithfulness_1to5": None,
                })

    # Pairwise experiment outputs if available
    pairwise_root = ROOT / "data" / "summaries" / "pairwise_experiment"
    if pairwise_root.exists():
        for p in sorted(pairwise_root.glob("lecture*/with_pairwise.json")):
            lecture_id = p.parent.name
            data = json.loads(p.read_text(encoding="utf-8"))
            summary = _normalize_text(data.get("refined_summary", ""))
            if summary:
                candidates.append({
                    "sample_id": f"{lecture_id}::pairwise_with",
                    "lecture_id": lecture_id,
                    "source": "pairwise_with",
                    "summary": summary,
                    "human_faithfulness_1to5": None,
                })

        for p in sorted(pairwise_root.glob("lecture*/without_pairwise.json")):
            lecture_id = p.parent.name
            data = json.loads(p.read_text(encoding="utf-8"))
            summary = _normalize_text(data.get("refined_summary", ""))
            if summary:
                candidates.append({
                    "sample_id": f"{lecture_id}::pairwise_without",
                    "lecture_id": lecture_id,
                    "source": "pairwise_without",
                    "summary": summary,
                    "human_faithfulness_1to5": None,
                })

    # Deduplicate by summary text while preserving deterministic order
    seen = set()
    deduped: list[dict] = []
    for row in candidates:
        key = row["summary"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def main():
    candidates = collect_candidates()

    target_min, target_max = 30, 50
    if len(candidates) > target_max:
        candidates = candidates[:target_max]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("===== PREPARE HUMAN LABELING SET =====")
    print(f"Candidates written: {len(candidates)}")
    print(f"Output: {OUT_PATH}")
    if len(candidates) < target_min:
        print(
            f"Warning: generated fewer than {target_min} candidates. "
            "Run more eval/pairwise experiments to expand the pool."
        )
    else:
        print(f"Ready for manual annotation ({target_min}–{target_max} target met).")
    print("Fill `human_faithfulness_1to5` for each row, then run build_human_calibration_dataset.")


if __name__ == "__main__":
    main()
