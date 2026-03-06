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

- Python 3.10
- openai (>=1.3.0)
- python-dotenv
- pymupdf
- nltk
- streamlit
- plotly
- numpy / scipy / scikit-learn / matplotlib

> **NLTK resources:** On first run, the pipeline auto-downloads `punkt`, `punkt_tab`, `omw-1.4`, and `wordnet`. Ensure internet access for the initial execution.

## Project Structure

```
HondaResearchLabs_DSC180A-Eval-Systems-Of-NextGen-LLMs/
├── data/
│   ├── references/                    # Human-written reference summaries
│   ├── slides/                        # Lecture PDFs
│   └── summaries/
│       ├── model_s0/                  # Initial LLM summaries
│       └── refined_iterations/
│           └── lectureX/
│               ├── iter_0.txt ... iter_n.txt
│               ├── final.txt
│               ├── result.json
│               └── pairwise_s0_vs_refined.json
├── example_run/
│   └── lecture1/                      # Sample run artifacts for demo use
├── src/
│   ├── evaluation/
│   │   ├── NormalSchema.py
│   │   ├── pairwise.py
│   │   ├── pipeline.py
│   │   └── scoring.py
│   ├── experiments/
│   │   └── run_eval.py
│   ├── models/
│   │   ├── judge.py
│   │   ├── refinement.py
│   │   ├── lever_based_refinement.py
│   │   └── summarizer.py
│   ├── utils/
│   └── visualization/
├── papers/
│   ├── Q1/
│   └── Q2/
├── environment.yml
├── requirements.txt
├── startup.sh
├── startup.ps1
└── README.md
```

## Paper

Our Quarter 1 Report/Paper and Quarter 2 Project Proposal are listed under the `papers/Q1/` and `papers/Q2/` folders.

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
- `min_agreement`: optional float, default `0.7`

Outputs are written to:

```
data/summaries/refined_iterations/lecture1/
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
- **Agreement analysis** against reference summary
- **METEOR** semantic similarity
- **Deterministic signals** (`length_error`, `section_coverage_pct`, `glossary_recall`, `suspected_hallucination_rate`)

### 4) Final Scoring

Comprehensive layered score:

$$
C = 0.6 \cdot \text{domain\_rubric} + 0.2 \cdot \text{nlp\_agreement} + 0.2 \cdot \text{meteor}
$$

Manual weighted score (explicit baseline):

$$
M = (0.6\cdot\text{base} + 0.2\cdot\text{meteor} + 0.2\cdot\text{coverage}) - 0.1\cdot2^{\text{hallucination}}
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
C = 0.6 \cdot \text{domain rubric} + 0.2 \cdot \text{nlp agreement} + 0.2 \cdot \text{meteor}
refinement_metadata
final_score_0to1
lecture_title
```

M = (0.6 \cdot \text{base} + 0.2 \cdot \text{meteor} + 0.2 \cdot \text{coverage}) - 0.1 \cdot 2^{\text{hallucination}}

## Dashboards

Run after generating evaluation outputs.

### Static dashboard

```bash
S_{\text{final}} = S(1 - 0.15h)
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

- `data/summaries/refined_iterations/lecture1/` - UCSD MGT 45 (Financial & Managerial Accounting) [Dr. Andreya Pérez Silva] - Week 1 Slides
- `data/summaries/refined_iterations/lecture2/` - UCSD MGT 45 (Financial & Managerial Accounting) [Dr. Andreya Pérez Silva] - Week 2 Slides
- `data/summaries/refined_iterations/lecture3/` - UCSD LATI 10 (Reading North by South: Latin American Studies and the US Liberation Movements) [Dr. Amy Kennemore] - Week 3 Slides
- `data/summaries/refined_iterations/lecture4/` - UCSD ANTH 2 (Human Origins) [Maria Carolina Marchetto, PhD] - Week 2 Slides
- `data/summaries/refined_iterations/lecture5/` - UCSD EDS/SOCI 117 (Language, Culture, and Education) [Gabrielle Jones, Ph.D.] - Week 2 Wednesday Slides
- `data/summaries/refined_iterations/lecture6/` - UCSD DSC 100 (Introduction to Data Management) [Babak Salimi] - Week 3 Slides
- `data/summaries/refined_iterations/lecture7/` - UCSD COGS 14A (Intro to Research Methods) [Sarah C. Creel] - Week 5 Slides

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

Note on pipeline evolution: the project originally emphasized human-reference-based comparison more heavily. The current hybrid pipeline has evolved to rely on multi-signal and model-judge evaluation, so references are no longer the sole requirement for assessing quality. Reference files are still included for compatibility, benchmarking continuity, and additional diagnostics (for example, agreement and METEOR when enabled).

## Adding Your Own Lecture

To evaluate a new lecture end-to-end with the current CLI pipeline:

1. Add your slide deck as `data/slides/lectureN.pdf` (for the next available `N`).
2. Add a reference summary as `data/references/lectureN_reference.txt`.
    - The hybrid pipeline does not rely only on reference matching, but the current `run_eval` workflow still uses references for agreement/METEOR and richer diagnostics.
    - A concise, high-quality reference (roughly 250–300 words) is sufficient.
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

for presentation or grading demos.

## Future Directions

- Improve domain routing precision and confidence calibration
- Expand multimodal handling for charts/figures/equations
- Add stronger uncertainty-aware evaluation reports
- Scale to larger lecture corpora and cross-institution studies
- Support human-in-the-loop editing workflows

## Authors

- Rahul Sengupta
- Akshay Medidi
- Zeyu (Edward) Qi
- Zachary Thomason

## Mentors

- Rajeev Chhajer
- Ryan Lingo

## License

This project was developed for the UC San Diego DSC180 Capstone (2025–2026 academic year).

**Evaluation Strategies for Next-Generation AI Systems**  
*Industry Partner - Honda Research Labs*