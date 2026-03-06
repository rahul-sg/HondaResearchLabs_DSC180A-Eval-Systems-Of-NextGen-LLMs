# DSC180 – Evaluation Strategies for Next-Generation AI Systems

**Capstone Project | UC San Diego | Spring 2026**

This repository implements an end-to-end pipeline for generating, refining, and evaluating large-language-model (LLM) summaries of university lecture slides.

Our capstone goal is to study how next-generation LLM systems can be evaluated and improved in a way that is reproducible, domain-aware, and practically useful for educational content. Instead of relying on a single metric or fixed iteration schedule, this project combines deterministic signals, LLM rubric judging, pairwise preference testing, and hybrid final scoring to better capture summary quality. The pipeline is designed for research and applied benchmarking: it records full intermediate artifacts, exposes stopping behavior, and supports cross-domain comparison across multiple UCSD lecture datasets.

Given a lecture PDF, the system:
1. Generates an initial summary (`S0`)
2. Iteratively refines it with lever-based guidance
3. Evaluates quality using deterministic signals + LLM judges
4. Produces reproducible artifacts (`iter_*.txt`, `final.txt`, `result.json`, pairwise outputs)

## Quick Start

```bash
# 1) Create environment
conda env create -f environment.yml
conda activate dsc180a-eval

# 2) Set API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 3) Run one lecture
python -m src.experiments.run_eval lecture1
```

No human reference file is required for the default pipeline mode.

Then review generated artifacts in:

```
data/summaries/refined_iterations/lecture1/
example_run/lecture1/
```

## Key Innovations

- **Lever-Based Iterative Refinement**: criterion-driven iteration instead of fixed loops
- **Domain-Aware Rubric Judging**: automatic domain detection with discipline-aware criteria
- **Hybrid Final Scoring**: combines comprehensive layered score with explicit weighted score
- **Trend-Aware Stopping**: decision-table stopping (`pass`, `borderline`, `stalled`) with plateau detection
- **Research-Grade Outputs**: full metadata and intermediate traces for reproducibility

## Dependencies

This project uses packages listed in `environment.yml` and `requirements.txt`, including:

- Python 3.10.20
- openai 2.26.0
- python-dotenv 1.2.2
- pymupdf 1.27.1
- nltk 3.9.3
- streamlit 1.55.0
- plotly 6.6.0
- numpy 2.2.6
- scipy 1.15.2
- scikit-learn 1.7.2
- matplotlib 3.10.8

> **NLTK resources:** On first run, the pipeline auto-downloads `punkt`, `punkt_tab`, `omw-1.4`, and `wordnet`. Ensure internet access for the initial execution.

## Project Structure

```
HondaResearchLabs_DSC180A-Eval-Systems-Of-NextGen-LLMs/
├── assets/
│   ├── 99p-logo.png
│   └── hdsi-white.png
├── data/
│   ├── references/                    # Optional legacy benchmark references
│   ├── slides/                        # Lecture PDFs
│   └── summaries/
│       ├── bare_bones_experiment/     # Judge-only experiment outputs
│       ├── model_s0/                  # Initial LLM summaries
│       ├── pairwise_experiment/       # Pairwise ablation outputs
│       └── refined_iterations/
│           └── lectureX/
│               ├── iter_0.txt ... iter_n.txt
│               ├── final.txt
│               ├── result.json
│               └── pairwise_s0_vs_refined.json
├── example_run/
│   ├── lecture1/
│   ├── lecture2/
│   ├── ...
│   └── lecture7/                      # Sample run artifacts for demo use
├── dev_notes/                         # Development notes and archived test scripts
├── src/
│   ├── evaluation/
│   │   ├── NormalSchema.py
│   │   ├── pairwise.py
│   │   ├── pipeline.py
│   │   └── scoring.py
│   ├── experiments/
│   │   ├── bare_bones_judge_experiment.py
│   │   ├── compare_models.py
│   │   ├── pairwise_experiment.py
│   │   ├── refine_demo.py
│   │   ├── run_eval.py
│   │   └── sanity_checks.py
│   ├── models/
│   │   ├── judge.py
│   │   ├── lever_based_refinement.py
│   │   ├── llm_client.py
│   │   ├── refinement.py
│   │   └── summarizer.py
│   ├── utils/
│   └── visualization/
├── papers/
│   ├── Q1/
│   └── Q2/
├── .env.example
├── environment.yml
├── requirements.txt
├── startup.sh
├── startup.ps1
└── README.md
```

## Paper

Our Quarter 1 report and Quarter 2 proposal are listed under `papers/Q1/` and `papers/Q2/`.

## Environment Setup

### Option A: Start-Up Script (recommended)

**Mac/Linux (bash)**
```bash
chmod +x startup.sh
source startup.sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File startup.ps1
```

### Option B: Manual Setup

```bash
conda env create -f environment.yml
conda activate dsc180a-eval
```

Create `.env` in project root:

```bash
OPENAI_API_KEY=your_key_here
```

Deactivate environment:

```bash
conda deactivate
```

## Running Evaluation

### Standard run

```bash
python -m src.experiments.run_eval lecture1
```

### Full CLI signature

```bash
python -m src.experiments.run_eval \
    lecture1 [force_regen] [use_lever_based] [min_avg_score] \
    [min_change_threshold] [max_iterations] [min_iterations] [min_agreement]
```

Parameters:
- `lecture1`: lecture id (`lectureN`)
- `force_regen`: optional (`yes`/`no`), default `no`
- `use_lever_based`: optional (`yes`/`no`), default `yes`
- `min_avg_score`: optional float, default `4.0`
- `min_change_threshold`: optional float, default `0.03`
- `max_iterations`: optional int, default `12`
- `min_iterations`: optional int, default `4`
- `min_agreement`: optional float, default `0.7` (**legacy/compat parameter**)

`run_eval` works without `data/references/lectureN_reference.txt` in default reference-free mode.

Outputs are written to:

```
data/summaries/refined_iterations/lectureN/
    iter_0.txt
    iter_1.txt
    ...
    final.txt
    result.json
    pairwise_s0_vs_refined.json
```

`iter_X` count varies depending on stopping behavior.

## Current Pipeline (What Happens Internally)

### 1) Generation + Refinement
- Generate/reuse `S0`
- Iteratively refine using rubric feedback + pairwise winner selection

### 2) Trend-Aware Stopping (domain-agnostic)
The lever-based controller uses a decision table with minimum-iteration enforcement:
- **pass**: strict quality/signal criteria met, or high quality plateau
- **borderline**: moderate quality + stable trajectory
- **stalled**: trend is not improving meaningfully
- **max_iters**: safety stop

This avoids both premature stopping and endless loops.

### 3) Evaluation Layers
- **Domain-aware rubric** (coverage, faithfulness, organization, clarity, style)
- **Deterministic signals** (`length_error`, `section_coverage_pct`, `glossary_recall`, `suspected_hallucination_rate`)
- **Pairwise preference** during refinement candidate selection

### 4) Final Scoring

Comprehensive layered score:

$$
C = domain_{rubric}
$$

Manual weighted score (explicit baseline):

$$
M = (0.8 \cdot base + 0.2 \cdot coverage) - 0.1 \cdot 2^{hallucination}
$$

Balanced hybrid final score:

$$
S = 0.7C + 0.3M
$$

$$
S_{final} = S(1 - 0.15h)
$$

`result.json` also logs both component scores and disagreement diagnostics.

## Example Sample Run

A complete sample run is provided in:

```
example_run/lecture1/
    iter_0.txt
    iter_1.txt
    iter_2.txt
    iter_3.txt
    final.txt
    result.json
    pairwise_s0_vs_refined.json
    s0_summary.txt
```

This directory is intended for quick demonstration.

## Result Schema (`result.json`)

Key fields:

```
refined_summary
signals
rubric
agreement
comprehensive_scoring
hybrid_scoring
refinement_metadata
final_score_0to1
lecture_title
```

`agreement` is retained for compatibility and is marked unused in reference-free mode.

Notable metadata fields include stopping reason, iteration history, lever history, and quality trajectory.

## Dashboards

Run after generating evaluation outputs.

### Static dashboard

```bash
python -m src.visualization.dashboard lecture1
```

By default, this now renders:
- a single-lecture dashboard (current pipeline metrics), and
- a cohort insights view sourced from `example_run/`.

Optional flags:

```bash
# Disable cohort insights panel
python -m src.visualization.dashboard lecture1 --no-cohort

# Enable semantic drift embedding plot (slower, API cost)
python -m src.visualization.dashboard lecture1 --drift

# Render merged single figure (default uses split windows)
python -m src.visualization.dashboard lecture1 --merged
```

### Interactive Streamlit dashboard

```bash
streamlit run src/visualization/interactive_dashboard.py
```

Includes summary trends, rubric visuals, score-component diagnostics, stop-state diagnostics, cohort insights (`example_run`), and full-text iteration views.

## Provided Dataset Coverage

The repository includes multiple lecture/reference pairs across domains (business, humanities, social sciences, computer science, and psychology) under:

- `data/slides/lectureN.pdf`
- `data/references/lectureN_reference.txt`
- `data/summaries/model_s0/lectureN.txt`
- `data/summaries/refined_iterations/lectureN/`

### Example Test Data

We have provided basic test results within the following domains:

- `data/summaries/refined_iterations/lecture1/` - UCSD MGT 45 (Financial & Managerial Accounting) [Dr. Andreya Pérez Silva, aperezsilva@ucsd.edu] - Week 1 Slides
- `data/summaries/refined_iterations/lecture2/` - UCSD MGT 45 (Financial & Managerial Accounting) [Dr. Andreya Pérez Silva, aperezsilva@ucsd.edu] - Week 2 Slides
- `data/summaries/refined_iterations/lecture3/` - UCSD LATI 10 (Reading North by South: Latin American Studies and the US Liberation Movements) [Dr. Amy Kennemore, akennemo@ucsd.edu] - Week 3 Slides
- `data/summaries/refined_iterations/lecture4/` - UCSD ANTH 2 (Human Origins) [Maria Carolina Marchetto, PhD, mcmarchetto@ucsd.edu] - Week 2 Slides
- `data/summaries/refined_iterations/lecture5/` - UCSD EDS/SOCI 117 (Language, Culture, and Education) [Gabrielle Jones, Ph.D., gajones@ucsd.edu] - Week 2 Wednesday Slides
- `data/summaries/refined_iterations/lecture6/` - UCSD DSC 100 (Introduction to Data Management) [Babak Salimi, bsalimi@ucsd.edu] - Week 3 Slides
- `data/summaries/refined_iterations/lecture7/` - UCSD COGS 14A (Intro to Research Methods) [Sarah C. Creel, screel@ucsd.edu] - Week 5 Slides

The LLM-generated initial summaries for each test set are stored here:

```
data/summaries/model_s0/lecture1.txt
data/summaries/model_s0/lecture2.txt
data/summaries/model_s0/lecture3.txt
data/summaries/model_s0/lecture4.txt
data/summaries/model_s0/lecture5.txt
data/summaries/model_s0/lecture6.txt
data/summaries/model_s0/lecture7.txt
```

And the human-written reference summaries for each test set are stored here:

```
data/references/lecture1_reference.txt
data/references/lecture2_reference.txt
data/references/lecture3_reference.txt
data/references/lecture4_reference.txt
data/references/lecture5_reference.txt
data/references/lecture6_reference.txt
data/references/lecture7_reference.txt
```

Note on pipeline evolution: the initial/legacy workflow emphasized human-reference-driven evaluation. The current default pipeline is reference-free and uses multi-signal + model-judge evaluation (deterministic signals, rubric judging, pairwise preference, and hybrid scoring). Human reference files are retained only for continuity and optional legacy diagnostics.

## Adding Your Own Lecture

To evaluate a new lecture end-to-end with the current CLI pipeline:

1. Add your slide deck as `data/slides/lectureN.pdf` (for the next available `N`).
2. (Optional, legacy) Add a reference summary as `data/references/lectureN_reference.txt` if you want benchmark continuity.
3. Run the evaluation:

```bash
python -m src.experiments.run_eval lectureN
```

4. Review outputs in:

```
data/summaries/refined_iterations/lectureN/
```

5. (Optional) Copy artifacts into:

```
example_run/lectureN/
```

The `example_run` artifacts are provided as presentation/demo examples.

## Future Directions

- Improve scorer calibration and disagreement-based confidence reporting
- Expand multimodal handling for charts, figures, and equations
- Add longitudinal/cohort analytics across larger lecture corpora
- Support human-in-the-loop editing and reviewer-facing diagnostics

## Authors

- Rahul Sengupta ([LinkedIn](https://www.linkedin.com/in/rahul-sg/))
- Akshay Medidi ([LinkedIn](https://www.linkedin.com/in/akshay-medidi-934a81202/))
- Zeyu (Edward) Qi ([LinkedIn](https://www.linkedin.com/in/qi-zeyu/))
- Zachary Thomason ([LinkedIn](https://www.linkedin.com/in/zachary-thomason/))

## Mentors

- Rajeev Chhajer
- Ryan Lingo

## License

This project was developed for the UC San Diego DSC180 Capstone (2025–2026 academic year).

**Evaluation Strategies for Next-Generation AI Systems**  
*Industry Partners - Honda Research Labs and 99P Labs*

<p>
    <img src="assets/99p-logo.png" alt="99P Labs Logo" width="130" />
    <img src="assets/hdsi-white.png" alt="HDSI Logo" width="240" style="margin-left: 24px; margin-top: -14px;" />
</p>