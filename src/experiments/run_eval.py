import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import json

from src.evaluation.pipeline import evaluate_summary
from src.models.llm_client import LLMConfig
from src.models.summarizer import generate_initial_summary
from src.evaluation.pairwise import round_robin_pairwise
from src.utils.io import load_slides


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

def main():
    #parse args
    if len(sys.argv) > 1:
        lecture_id = sys.argv[1].strip()
    else:
        lecture_id = "lecture1"

    if len(sys.argv) > 2:
        force_regen = sys.argv[2].strip().lower()
    else:
        force_regen = "no"   # default

    if len(sys.argv) > 3:
        use_lever_based = sys.argv[3].strip().lower() != "no"
    else:
        use_lever_based = True  # default: use lever-based refinement

    # optional stopping parameters: avg_score, change_thresh, max_iters, min_iters
    if len(sys.argv) > 4:
        min_avg_score = float(sys.argv[4])
    else:
        min_avg_score = 4.0

    if len(sys.argv) > 5:
        min_change_threshold = float(sys.argv[5])
    else:
        min_change_threshold = 0.03

    if len(sys.argv) > 6:
        max_iterations = int(sys.argv[6])
    else:
        max_iterations = 12

    if len(sys.argv) > 7:
        min_iterations = int(sys.argv[7])
    else:
        min_iterations = 4

    if len(sys.argv) > 8:
        min_agreement = float(sys.argv[8])
    else:
        min_agreement = 0.7

    if len(sys.argv) > 9:
        hallucination_policy = sys.argv[9].strip().lower()
    else:
        hallucination_policy = "tuned"

    if len(sys.argv) > 10:
        hallucination_alpha_override = float(sys.argv[10])
    else:
        hallucination_alpha_override = None

    if len(sys.argv) > 11:
        hallucination_beta_override = float(sys.argv[11])
    else:
        hallucination_beta_override = None

    if hallucination_policy == "tuned":
        hallucination_damping_alpha = 0.05
        hallucination_subtractive_beta = 0.0
    elif hallucination_policy == "legacy":
        hallucination_damping_alpha = 0.15
        hallucination_subtractive_beta = 0.10
    elif hallucination_policy == "human_tuned":
        hallucination_damping_alpha = 0.20
        hallucination_subtractive_beta = 0.125
    else:
        hallucination_policy = "tuned"
        hallucination_damping_alpha = 0.05
        hallucination_subtractive_beta = 0.0

    if hallucination_alpha_override is not None:
        hallucination_damping_alpha = hallucination_alpha_override
    if hallucination_beta_override is not None:
        hallucination_subtractive_beta = hallucination_beta_override

    print(f"\nLecture selected: {lecture_id}")
    print(f"Force regenerate S0: {force_regen}")
    print(f"Lever-based refinement: {use_lever_based}")
    print(f"Stopping params -> min_avg_score: {min_avg_score}, min_change_threshold: {min_change_threshold}, max_iters: {max_iterations}, min_iters: {min_iterations}, min_agreement(legacy): {min_agreement}")
    print(
        "Hallucination policy -> "
        f"{hallucination_policy} "
        f"(alpha={hallucination_damping_alpha}, beta={hallucination_subtractive_beta})"
    )
    if hallucination_alpha_override is not None or hallucination_beta_override is not None:
        print("Hallucination override source: CLI alpha/beta override")

    if lecture_id.lower() == "all":
        slides_dir = ROOT / "data" / "slides"
        lecture_ids = sorted(p.stem for p in slides_dir.glob("lecture*.pdf"))
        if not lecture_ids:
            raise FileNotFoundError(f"No lecture PDFs found in {slides_dir}")

        failed = []
        for lid in lecture_ids:
            print(f"\n[RUN_EVAL ALL] Running {lid}...")
            cmd = [
                sys.executable,
                "-m",
                "src.experiments.run_eval",
                lid,
                force_regen,
                "yes" if use_lever_based else "no",
                str(min_avg_score),
                str(min_change_threshold),
                str(max_iterations),
                str(min_iterations),
                str(min_agreement),
                hallucination_policy,
            ]
            if hallucination_alpha_override is not None:
                cmd.append(str(hallucination_alpha_override))
            if hallucination_beta_override is not None:
                cmd.append(str(hallucination_beta_override))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failed.append((lid, result.returncode))

        if failed:
            print("\n[RUN_EVAL ALL] Failures:")
            for lid, code in failed:
                print(f"- {lid}: exit code {code}")
            raise SystemExit(1)

        print(f"\n[RUN_EVAL ALL] Completed {len(lecture_ids)} lectures successfully.")
        return

    #paths
    SLIDES_PATH = f"data/slides/{lecture_id}.pdf"
    HUMAN_REF_PATH = f"data/references/{lecture_id}_reference.txt"
    INITIAL_SUMMARY_PATH = Path(f"data/summaries/model_s0/{lecture_id}.txt")
    OUT_DIR = Path(f"data/summaries/refined_iterations/{lecture_id}")

    # Validate input files
    if not Path(SLIDES_PATH).exists():
        raise FileNotFoundError(f" Lecture slides not found: {SLIDES_PATH}")

    if Path(HUMAN_REF_PATH).exists():
        print(f"Reference file found (legacy diagnostics optional): {HUMAN_REF_PATH}")
    else:
        print(f"No reference file found for {lecture_id}; running in reference-free mode.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous outputs
    print(f"Clearing previous evaluation runs for {lecture_id}...")
    for file in OUT_DIR.glob("*"):
        try:
            file.unlink()
        except Exception:
            print(f" Could not delete: {file}")

    # load optional reference (legacy only)
    human_reference = None
    if Path(HUMAN_REF_PATH).exists():
        with open(HUMAN_REF_PATH, "r", encoding="utf-8") as f:
            human_reference = f.read().strip()

    #generate summary 0
    regenerate = (force_regen == "yes")
    initial_summary = ""

    if INITIAL_SUMMARY_PATH.exists() and not regenerate:
        # Try loading existing S0
        with open(INITIAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            initial_summary = f.read().strip()

        if len(initial_summary.split()) < 50:
            print("[S0] Existing S0 too short — regenerating...")
            initial_summary = ""
        else:
            print(f"[S0] Reusing existing S0 at {INITIAL_SUMMARY_PATH}")

    else:
        if regenerate:
            print("[S0] Force regenerate is ON — creating new S0.")
        else:
            print("[S0] No S0 found — generating new S0.")

        initial_summary = ""

    # Generate new S0 if needed
    if not initial_summary:
        print("[S0] Generating new S0 with gpt-5-chat-latest...")

        cfg_summarizer = LLMConfig(
            model="gpt-5-chat-latest",
            max_completion_tokens=700,
        )

        initial_summary = generate_initial_summary(SLIDES_PATH, cfg_summarizer)

        INITIAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INITIAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(initial_summary)

        print(f"[S0] Saved new S0 → {INITIAL_SUMMARY_PATH}")

    # Judge and refine configs
    cfg_judge = LLMConfig(
        model="gpt-5-chat-latest",
        max_completion_tokens=512,
    )

    cfg_refine = LLMConfig(
        model="gpt-5-chat-latest",
        max_completion_tokens=800,
    )

    #Evaluate with lever-based or legacy refinement
    result = evaluate_summary(
        slide_path=SLIDES_PATH,
        initial_summary=initial_summary,
        human_reference=human_reference,
        cfg_judge=cfg_judge,
        cfg_refine=cfg_refine,
        out_dir=str(OUT_DIR),
        target_words=350,
        use_lever_based=use_lever_based,
        min_avg_score=min_avg_score,  # Stop when avg rubric reaches threshold
        min_change_threshold=min_change_threshold,  # convergence threshold
        max_iterations=max_iterations,  # Safety limit
        min_iterations=min_iterations,
        min_agreement=min_agreement,
        scoring_policy=hallucination_policy,
        hallucination_damping_alpha=hallucination_damping_alpha,
        hallucination_subtractive_beta=hallucination_subtractive_beta,
    )

    print("\nRunning pairwise comparison (GPT-5 S0 vs Refined)...")

    slides = load_slides(SLIDES_PATH)["slides"]

    pairwise_results = round_robin_pairwise(
        slides=slides,
        summaries={
            "gpt5_S0": initial_summary,
            "gpt5_refined": result["refined_summary"],
        },
        cfg_judge=cfg_judge,
        runs=5,   # ensemble for stability
    )

    pairwise_out = OUT_DIR / "pairwise_s0_vs_refined.json"
    with open(pairwise_out, "w", encoding="utf-8") as f:
        json.dump(pairwise_results, f, indent=2)

    print("Pairwise wins:", pairwise_results["wins"])
    print("Pairwise win rates:", pairwise_results["win_rate"])
    print(f"Pairwise results saved to: {pairwise_out}")


    #print final results
    print("\n===== FINAL EVALUATION RESULT =====")
    print("Final Score (0–1):", result["final_score_0to1"])
    if "leaderboard_scores" in result:
        leaderboard = result["leaderboard_scores"]
        print("Raw Quality Score:", leaderboard.get("raw_quality_score"))
        print("Risk-Adjusted Score:", leaderboard.get("risk_adjusted_score"))

    # Show comprehensive scoring breakdown
    if "comprehensive_scoring" in result:
        comp = result["comprehensive_scoring"]
        print("\n--- COMPREHENSIVE SCORING BREAKDOWN ---")
        print(f"Domain Detected: {comp.get('detected_domain', 'unknown').upper()}")
        mode = comp.get("mode", "unknown")
        print(f"Scoring Mode: {mode}")
        print(f"Domain-Aware Rubric Score: {comp['layer_scores']['domain_rubric']:.3f}")
        if mode == "reference_aware":
            print(f"NLP Agreement Score: {comp['layer_scores']['nlp_agreement']:.3f}")
            print(f"Semantic Similarity (METEOR): {comp['layer_scores']['semantic_similarity']:.3f}")
        else:
            print("Reference metrics: disabled (agreement/METEOR not used)")
        print(f"Layer Weights: Domain {comp['layer_weights']['domain_rubric']:.1f}, NLP {comp['layer_weights']['nlp_agreement']:.1f}, Semantic {comp['layer_weights']['semantic_similarity']:.1f}")
        print("Rubric Dimensions:", comp["rubric_breakdown"])

    try:
        print("\nRefined Summary:\n", result["refined_summary"])
    except UnicodeEncodeError:
        print("\nRefined Summary: [Unicode content that cannot be displayed]")
    print("\nSignals:", result["signals"])
    print("\nDetailed Rubric:", result["rubric"])
    print("\nAgreement Analysis:", result.get("agreement", {"used": False}))

    # Print refinement metadata
    if "refinement_metadata" in result:
        metadata = result["refinement_metadata"]
        print("\n===== REFINEMENT METADATA =====")
        print(f"Iterations completed: {metadata.get('iterations_completed')}")
        print(f"Final avg rubric score: {metadata.get('final_avg_score', 'N/A'):.2f}/5")
        print(f"Final word count: {metadata.get('final_word_count', 'N/A')}")
        print(f"Stopping reason: {metadata.get('stopping_reason', 'N/A')}")

    print("\nOutputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

