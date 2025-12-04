# 📘 **DSC180A – Lecture Summarization Evaluation System**
### *A reproducible LLM-as-Judge evaluation framework with iterative refinement*

This repository implements a rigorous and extensible evaluation pipeline for lecture summarization. The system combines **LLM-based rubric scoring**, **reference-aware evaluation**, and **deterministic similarity metrics** to assess the quality and faithfulness of generated summaries. It also includes an **iterative refinement loop**, allowing summaries to be improved automatically using LLM feedback.

This project supports research on summarization quality, model comparison, and reliability analysis in academic and industrial settings.

---

## 🚀 **Core Capabilities**

### **1. Reference-Free Evaluation (Rubric-Based)**
A structured 5-dimension rubric evaluates:
- Coverage  
- Faithfulness to slides  
- Organization  
- Pedagogical clarity  
- Writing quality  

### **2. Reference-Aware Evaluation**
Compares model output with a human-written reference summary using:
- Agreement scoring  
- Detection of missing key points  
- Detection of added inaccuracies  

### **3. Deterministic Faithfulness Signals**
Non-LLM, fully reproducible metrics:
- Length deviation  
- Section keyword coverage  
- Glossary recall  
- Hallucination rate via TF-IDF retrieval similarity  

### **4. Iterative Summary Refinement**
An automated refinement procedure:

```
S₀ → Judge → S₁ → Judge → S₂ → Judge → S₃
```

Each iteration rewrites the summary using judge feedback to increase faithfulness and clarity.

### **5. Ensemble Judging**
Multiple LLM calls with different seeds reduce variance and produce more stable scores.

### **6. Pairwise Comparison**
Allows head-to-head evaluation of alternative summaries using an LLM judge.

### **7. Pluggable LLM Backend**
All API access is centralized, enabling easy replacement with:
- OpenAI models  
- Anthropic Claude  
- Azure OpenAI  
- Local inference backends  

---

## 📂 **Repository Structure**

```
.
├── eval_baseline.py        # Main evaluation, refinement, and scoring system
├── environment.yml         # Conda environment file
├── .env.example            # Template for environment variables
├── .gitignore              # Excludes secrets, venvs, and caches
└── README.md               # Project documentation
```

---

## ⚙️ **Setup and Installation**

### **1. Create the Conda Environment**

```bash
conda env create -f environment.yml
conda activate dsc180a-eval
```

### **2. Configure Environment Variables**

Copy the example:

```bash
cp .env.example .env
```

Insert your key in `.env`:

```
OPENAI_API_KEY=sk-...
```

---

## ▶️ **Running the Evaluation Pipeline**

```bash
python eval_baseline.py
```

This will:

1. Load example slides and summary  
2. Compute deterministic signals  
3. Run rubric-based LLM scoring  
4. Run reference-aware agreement scoring  
5. Perform iterative refinement (if enabled)  
6. Output the final summary and scalar evaluation score  

---

## 🧪 **Evaluating Your Own Summaries**

Replace the example definitions in `eval_baseline.py`:

```python
slides = [...]
human_ref = "..."
model_sum = "..."
```

or load from files:

```python
import json
slides = json.load(open("slides.json"))
model_sum = open("summary.txt").read()
```

---

## 🔄 **Controlling Iterative Refinement**

```python
res = evaluate_one_summary(
    slides,
    model_sum,
    human_ref,
    cfg,
    refine_iters=3
)
```

Set `refine_iters=0` to disable refinement.

---

## 📊 **Output Format**

```json
{
  "refined_summary": "...",
  "signals": {...},
  "rubric": {...},
  "agreement": {...},
  "final_score_0to1": 0.87
}
```

---

## 🔒 **Security Considerations**

- `.env` must not be committed  
- `.gitignore` excludes secrets, cache directories, and virtual environments  

---

## 📜 **License**

This project was developed for the UC San Diego DSC180A Capstone.  
It may be adapted for educational or research purposes with appropriate attribution.
