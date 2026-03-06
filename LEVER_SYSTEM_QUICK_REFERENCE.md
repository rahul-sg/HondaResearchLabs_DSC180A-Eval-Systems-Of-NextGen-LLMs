# Lever-Based Iterative Refinement - Quick Reference

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_eval.py (Entry Point)                           │
│  - Parses arguments: lecture_id, force_regen, use_lever_based               │
│  - Default: use_lever_based=True                                            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    evaluate_summary() [pipeline.py]                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ if use_lever_based=True:                                             │  │
│  │   Call iterative_refinement_lever_based()                            │  │
│  │ else:                                                                │  │
│  │   Call iterative_refinement() [legacy, 3 iterations]                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Later stages (same for both):                                              │
│  - Compute signals, METEOR, rubric, agreement                               │
│  - Final pairwise S0 vs Refined                                             │
│  - Save result.json with refinement_metadata                                │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  iterative_refinement_lever_based() [refinement.py]          │
    │                                                              │
    │  Main Loop (up to max_iterations):                          │
    │    1. Judge summary → extract rubric scores                 │
    │    2. Create RefinementState                                │
    │    3. Check stopping criteria (via Controller)              │
    │    4. If continue:                                          │
    │       a. Get lever-based guidance                           │
    │       b. Call refine_once_with_guidance()                   │
    │       c. Pairwise comparison (current vs refined)           │
    │       d. Keep winner as new current                         │
    │       e. Save iteration                                     │
    │    5. If stop: Break loop                                   │
    │                                                              │
    │  Return: (final_summary, metadata)                          │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
    ┌──────────────────────────┐  ┌──────────────────────────┐
    │ LeverBasedRefinement     │  │ refine_once_with        │
    │ Controller               │  │ _guidance()              │
    │                          │  │                          │
    │ - should_stop()          │  │ - Uses lever guidance    │
    │ - get_stopping_reason()  │  │ - Calls LLM with        │
    │ - get_refinement_        │  │   prioritized prompt     │
    │   guidance()             │  │ - Returns refined text   │
    └──────────────────────────┘  └──────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │ RefinementState Dataclass                │
    │                                          │
    │ Fields:                                  │
    │ - iteration: int                         │
    │ - summary: str                           │
    │ - rubric: Dict[lever→score]              │
    │ - lever_history: List[rubric]            │
    │ - word_count: int                        │
    │ - change_magnitude: float                │
    │ - avg_rubric_score: float                │
    │ - target_words: int                      │
    │                                          │
    │ Methods:                                 │
    │ - get_weak_levers()                      │
    │ - get_improving_levers()                 │
    │ - log_state()                            │
    └──────────────────────────────────────────┘
```

## Stopping Criteria Hierarchy

```
At each iteration, check in order:

┌─ STOP if avg_rubric_score >= min_avg_score (default 4.5)
│
├─ STOP if word_count ∈ [target±15%] AND avg_score >= 3.5
│
├─ STOP if iteration > 2 AND change_magnitude <= 0.03
│
└─ STOP if iteration >= max_iterations

Otherwise: Continue to next iteration
```

## Prompt Evolution

### Initial Summary (S0)
Generated from slides using LLM summarizer

### Iteration 1 Refinement Prompt
```
[Slides]
...slides text...

[Current Summary]
...S0...

[Judge Feedback]
{judge's rubric scores and issues}

[LEVER-BASED GUIDANCE]
- PRIORITY (levers ≤2): Focus on [coverage, organization]
- Weak levers identified from S0 judgment
- EXPAND: Current 250 words, target 350 (need +100 words)
- Current avg rubric: 2.6/5, target: min_avg_score (default 4.5/5)

[Current Rubric Scores]
- Coverage: 2
- Faithfulness: 3
- Organization: 2
- Clarity: 3
- Style: 3

TASK: Improve [coverage, organization], expand to 350 words...
```

### Iteration 2 Refinement Prompt
```
[Similar structure but updated with:]

[LEVER-BASED GUIDANCE]
- PRIORITY (levers ≤2): Focus on [coverage] (organization improved)
- MAINTAIN: Keep strengthening [organization, faithfulness, clarity]
- Word count: 310 (still need +40 for target 350)
- Current avg rubric: 3.4/5, target: min_avg_score (default 4.5/5)

[Current Rubric Scores]
- Coverage: 2
- Faithfulness: 4 ← IMPROVED
- Organization: 3 ← IMPROVED
- Clarity: 4 ← IMPROVED
- Style: 2

TASK: Improve [coverage], maintain improvements, expand...
```

## Output Files (Per Lecture)

```
data/summaries/refined_iterations/lectureX/
├── iter_0.txt          # Initial S0
├── iter_1.txt          # After iteration 1
├── iter_2.txt          # After iteration 2
├── iter_3.txt          # After iteration 3
├── final.txt           # Final refined summary
├── result.json         # Complete evaluation result
│                       #   - signals, rubric, agreement
│                       #   - refinement_metadata ← NEW!
│                       #     {
│                       #       "iterations_completed": 3,
│                       #       "final_avg_score": 4.1,
│                       #       "final_word_count": 365,
│                       #       "final_agreement": 0.72,
│                       #       "stopping_reason": "...",
│                       #       "lever_history": [...],
│                       #       "final_rubric": {...}
│                       #     }
└── pairwise_s0_vs_refined.json  # Final S0 vs refined comparison
```

## Usage Examples

```bash
# Default: lever-based refinement
python src/experiments/run_eval.py lecture1

# Force regenerate S0
python src/experiments/run_eval.py lecture1 yes

# Use legacy 3-iteration mode
python src/experiments/run_eval.py lecture1 no no

# Different lecture, lever-based
python src/experiments/run_eval.py lecture3

# Full options with all stopping parameters
python src/experiments/run_eval.py lecture2 yes yes 4.2 0.02 12 3 0.75
                                   ^^^^^^^^ ^^^ ^^^ ^^^ ^^^ ^^ ^^ ^^
                                   lecture  regen lever  minScore changeThresh maxIters minIters minAgree
```

## Configuration in run_eval.py

```python
result = evaluate_summary(
    ...
    target_words=350,              # ← Adjust for lecture length
    use_lever_based=True,          # ← Set False for legacy mode
    min_avg_score=4.5,             # ← Stopping quality threshold (tighter default)
    min_change_threshold=0.03,     # ← Convergence threshold
    max_iterations=10,             # ← Safety limit
)
```

## Key Differences from Original System

| Feature | Original | New |
|---------|----------|-----|
| **Fixed iterations** | Yes (always 3) | No (1-10, criteria-driven) |
| **Guidance to refiner** | Judge feedback only | Judge + lever priorities |
| **Word count targeting** | No | Yes (±15% of target) |
| **Quality threshold** | None | Yes (avg >= min_avg_score, default 4.5) |
| **Convergence detection** | None | Yes (change <= 0.03) |
| **Metadata** | None | Rich refinement_metadata |
| **Python version** | Any | 3.7+ (dataclass + type hints) |

## Lever System Design

**Levers** = Rubric dimensions that can be "adjusted" via refinement

1. **Coverage** (1-5)
   - Are key ideas included?
   - Fix: Add missing concepts, expand scope

2. **Faithfulness** (1-5)
   - Are claims accurate per slides?
   - Fix: Remove unsupported claims, correct errors

3. **Organization** (1-5)
   - Is reasoning flow clear?
   - Fix: Restructure for better logic flow

4. **Clarity** (1-5)
   - Are explanations usable?
   - Fix: Simplify, add examples, improve transitions

5. **Style** (1-5)
   - Is presentation professional?
   - Fix: Improve notation, fix grammar, add nuance

**Weak lever** = Score ≤ 2 (priority for improvement)
**Improving lever** = Score increased from previous iteration (maintain momentum)

## When to Use Lever-Based vs. Legacy Mode

### Use Lever-Based (default, recommended)
- Consistent quality outcomes desired
- Variable iteration count acceptable
- Want to optimize time/cost per lecture
- Need transparency in stopping reasons

### Use Legacy (3-iteration)
- Fixed iteration count required
- Consistent LLM call budgets important
- Prefer simpler, predictable behavior
- Testing/validation against old results

---

For detailed documentation, see **LEVER_BASED_REFINEMENT_GUIDE.md**
