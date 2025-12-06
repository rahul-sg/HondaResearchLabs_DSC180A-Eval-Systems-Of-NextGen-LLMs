import os
from pathlib import Path
from src.utils.io import load_slides
from src.models.llm_client import LLMConfig
from src.models.judge import judge_rubric_ensemble



#direction of information sanity check
def direction_of_information_test(slides, summary, cfg_judge):

    # Normal evaluation
    res_full = judge_rubric_ensemble(slides, summary, cfg_judge, runs=2)

    # Blind the judge entirely
    res_blind = judge_rubric_ensemble([], summary, cfg_judge, runs=2)

    return {
        "faithfulness_full": res_full["faithfulness"],
        "faithfulness_blind": res_blind["faithfulness"],
        "passed": res_full["faithfulness"] > res_blind["faithfulness"]
    }


def main():
    #test lecture
    lecture_id = "lecture1"

    SLIDES_PATH = f"data/slides/{lecture_id}.pdf"

    #load
    slide_data = load_slides(SLIDES_PATH)
    slides = slide_data["slides"]

    #example summary
    sample_summary = (
        "Gradient descent updates parameters opposite the gradient to reduce loss. "
        "A learning rate controls step size. Too large and training may diverge; too small and "
        "convergence slows dramatically. Stopping criteria include validation loss, gradient norms, "
        "and max steps."
    )

    #judge
    cfg_judge = LLMConfig(model="gpt-5-chat-latest")

    #run check
    result = direction_of_information_test(slides, sample_summary, cfg_judge)

    #print results
    print("\n===== SANITY CHECK: DIRECTION OF INFORMATION =====")
    print("Faithfulness (with slides): ", result["faithfulness_full"])
    print("Faithfulness (blind):       ", result["faithfulness_blind"])
    print("PASSED:", result["passed"])


if __name__ == "__main__":
    main()
