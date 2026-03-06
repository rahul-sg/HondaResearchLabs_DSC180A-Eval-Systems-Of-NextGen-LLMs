# DSC180 – Evaluation Strategies for Next-Generation AI Systems

**Capstone Project | UC San Diego | Spring 2026**

This repository implements an end-to-end pipeline for generating, refining, and evaluating large-language-model (LLM) summaries of university lecture slides.

Given a lecture PDF, the system:
1. Generates an initial summary (`S0`)
2. Iteratively refines it with lever-based guidance
3. Evaluates quality using deterministic signals + LLM judges
4. Produces reproducible artifacts (`iter_*.txt`, `final.txt`, `result.json`, pairwise outputs)

---

## Key Innovations

- **Lever-Based Iterative Refinement**: criterion-driven iteration instead of fixed loops
- **Domain-Aware Rubric Judging**: automatic domain detection with discipline-aware criteria
- **Hybrid Final Scoring**: combines comprehensive layered score with explicit weighted score
- **Trend-Aware Stopping**: decision-table stopping (`pass`, `borderline`, `stalled`) with plateau detection
- **Research-Grade Outputs**: full metadata and intermediate traces for reproducibility

---

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

---

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

---

## Environment Setup

### Option A: Start-Up Script (recommended)

#### Mac/Linux (bash)
```bash
chmod +x startup.sh
source startup.sh
```

#### Windows (PowerShell)
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

---

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

---

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
	ext{final\_score} = S\cdot(1 - 0.15\cdot\text{hallucination})
$$

`result.json` also logs both component scores and disagreement diagnostics.

---

## Example Sample Run (TA Requirement)

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

This directory is intended for quick demonstration and grading verification.

---

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

Notable metadata fields include stopping reason, iteration history, lever history, and quality trajectory.

---

## Dashboards

Run after generating evaluation outputs.

### Static dashboard

```bash
python -m src.visualization.dashboard lecture1
```

### Interactive Streamlit dashboard

```bash
streamlit run src/visualization/interactive_dashboard.py
```

Includes summary trends, rubric visuals, signal diagnostics, agreement metrics, and full-text iteration views.

---

## Provided Dataset Coverage

The repository includes multiple lecture/reference pairs across domains (business, humanities, social sciences, computer science, and psychology) under:

- `data/slides/lectureN.pdf`
- `data/references/lectureN_reference.txt`
- `data/summaries/model_s0/lectureN.txt`
- `data/summaries/refined_iterations/lectureN/`

---

## Adding Your Own Lecture

1. Add `data/slides/lectureN.pdf`
2. Add `data/references/lectureN_reference.txt` (recommended ~250–300 words)
3. Run:

```bash
python -m src.experiments.run_eval lectureN
```

---

## Future Directions

- Improve domain routing precision and confidence calibration
- Expand multimodal handling for charts/figures/equations
- Add stronger uncertainty-aware evaluation reports
- Scale to larger lecture corpora and cross-institution studies
- Support human-in-the-loop editing workflows

---

## 👥 Authors

#### Rahul Sengupta
#### Akshay Medidi
#### Zeyu (Edward) Qi
#### Zachary Thomason

## 👥 Mentors

#### Rajeev Chhajer
#### Ryan Lingo

## 📜 License

This project was developed for the UC San Diego DSC180 Capstone (2025–2026 academic year).

**Evaluation Strategies for Next-Generation AI Systems**  
*Industry Partner - Honda Research Labs*