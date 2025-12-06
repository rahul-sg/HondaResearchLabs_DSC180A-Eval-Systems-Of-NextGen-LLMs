import os
from pathlib import Path

from src.utils.io import load_slides, write_iteration_summary, write_final_summary
from src.models.refinement import iterative_refinement
from src.models.llm_client import LLMConfig

def main():

    #test lecture
    lecture_id = "lecture1"

    SLIDES_PATH = f"data/slides/{lecture_id}.pdf"
    OUT_DIR = f"outputs/{lecture_id}/refine_demo"

    os.makedirs(OUT_DIR, exist_ok=True)

    #load
    slide_data = load_slides(SLIDES_PATH)
    slides = slide_data["slides"]

    #test summary
    initial_summary = (
        "This is an example initial summary. Replace with your true S0 from a "
        "summarization model. The refinement loop will iteratively improve it."
    )


    cfg_judge = LLMConfig(model="gpt-5-chat-latest")  # JSON-robust judge
    cfg_refine = LLMConfig(model="gpt-5-mini")        # cheap refiner

    #callback to each iteration
    def save_callback(iter_idx: int, summary_text: str):
        write_iteration_summary(OUT_DIR, iter_idx, summary_text)

    #refine
    refined_summary = iterative_refinement(
        slides=slides,
        initial_summary=initial_summary,
        cfg_judge=cfg_judge,
        cfg_refine=cfg_refine,
        iters=3,                      # number of refinement steps
        save_callback=save_callback,
    )

    # Save final
    write_final_summary(OUT_DIR, refined_summary)


    print("\n===== REFINEMENT COMPLETE =====")
    print("Final summary:\n")
    print(refined_summary)
    print(f"\nAll iterations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
