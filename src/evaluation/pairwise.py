from typing import Dict, Any, List
import random
from pathlib import Path

from src.models.judge import pairwise_judge_ensemble


# Perform round-robin pairwise comparison of summaries using ensemble judge
def round_robin_pairwise(
    slides: List[Dict],
    summaries: Dict[str, str],
    cfg_judge,
    runs: int = 5,
) -> Dict[str, Any]:


    names = list(summaries.keys())
    wins = {n: 0 for n in names}
    matches = []
    total_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nameA = names[i]
            nameB = names[j]

            total_pairs += 1

            A_text = summaries[nameA]
            B_text = summaries[nameB]

            # Ensemble A/B judge
            result = pairwise_judge_ensemble(
                slides=slides,
                A=A_text,
                B=B_text,
                cfg=cfg_judge,
                runs=runs,
            )

            # Determine winner
            winner_side = result["winner"]  # "A" or "B"
            winner_name = nameA if winner_side == "A" else nameB

            wins[winner_name] += 1

            matches.append({
                "A": nameA,
                "B": nameB,
                "winner": winner_name,
                "wins_detail": result["wins"],
                "reasons_sample": result["reasons_sample"],
            })

    # Normalize win rates
    win_rate = {
        name: wins[name] / max(1, total_pairs)
        for name in names
    }

    overall_winner = max(wins, key=wins.get)
    overall_text = summaries[overall_winner]
    print(f"Overall winner: {overall_text[:60]}")
    output = f"Overall winner: {overall_text}"
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairwise_overall_winners.log"

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(output + "\n")


    return {
        "wins": wins,
        "win_rate": win_rate,
        "matches": matches,
        "result_summary": overall_text,
        "overall_winner": overall_winner
    }
