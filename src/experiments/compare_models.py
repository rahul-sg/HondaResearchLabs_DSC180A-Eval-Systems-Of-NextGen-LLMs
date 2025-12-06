import os
from pathlib import Path
from src.utils.io import load_slides, write_json
from src.evaluation.pairwise import round_robin_pairwise
from src.models.llm_client import LLMConfig


def main():
    #lecture select
    lecture_id = "lecture1"

    SLIDES_PATH = f"data/slides/{lecture_id}.pdf"
    OUT_DIR = f"outputs/{lecture_id}/model_comparison"
    os.makedirs(OUT_DIR, exist_ok=True)

    #load
    slide_data = load_slides(SLIDES_PATH)
    slides = slide_data["slides"]

    
    summaries = {
        "gpt5": "This is GPT-5’s summary. Replace with real content.",
        "gpt4o": "This is GPT-4o’s summary.",
        "llama3": "This is LLaMA-3’s summary.",
    }

    cfg_judge = LLMConfig(model="gpt-5-chat-latest")

    #test
    results = round_robin_pairwise(
        slides=slides,
        summaries=summaries,
        cfg_judge=cfg_judge,
        runs=5,   # ensemble for stability
    )

    #create json
    write_json(os.path.join(OUT_DIR, "pairwise_results.json"), results)

    #print summary
    print("\n===== MODEL COMPARISON RESULTS =====")
    print("Wins:", results["wins"])
    print("Win Rates:", results["win_rate"])
    print("\nMatches (sample):")
    for m in results["matches"][:5]:
        print(m)

    print(f"\nFull results saved to: {OUT_DIR}/pairwise_results.json")


if __name__ == "__main__":
    main()
