# Lever-Based Iterative Refinement System

## Overview

I've implemented a sophisticated lever-based iterative refinement system that replaces the fixed 3-iteration approach with intelligent, criterion-based refinement. The system dynamically continues refining summaries until quality thresholds are reached, using rubric dimensions ("levers") to guide improvement priorities.

## Key Components

### 1. **New Module: `src/models/lever_based_refinement.py`**

This module contains the core lever-based refinement logic:

#### `RefinementState` Dataclass
Tracks the state of refinement at each iteration:
- `iteration`: Current iteration number
- `summary`: Current summary text
- `rubric`: Dict of current rubric scores {coverage, faithfulness, organization, clarity, style}
- `lever_history`: Historical rubric scores from all previous iterations
- `word_count`: Word count of current summary
- `change_magnitude`: Measure of summary change (0-1 Jaccard distance)
- `avg_rubric_score`: Average of all rubric dimensions
- `target_words`: Target word count (default 350)

Key methods:
- `get_weak_levers()`: Returns dimensions with scores ≤ 2 (priorities for improvement)
- `get_improving_levers()`: Returns dimensions that improved in the last iteration
- `log_state()`: Generates human-readable log of current state

#### `LeverBasedRefinementController`
Manages the refinement loop and stopping criteria:

**Constructor parameters:**
```python
LeverBasedRefinementController(
    target_words: int = 350,
    min_avg_score: float = 4.0,           # Stopping criterion 1 (default tightened to 4.5 in controller)
    min_change_threshold: float = 0.03,   # Stopping criterion 3
    target_word_tolerance: float = 0.15,  # ±15% of target
    max_iterations: int = 10,              # Stopping criterion 4
    min_iterations: int = 2,              # New guard: always run at least this many iterations
)
```

Stopping Criteria (any one triggers stopping, but checks are deferred until
at least `min_iterations` have run):
1. **High Quality**: Average rubric score ≥ min_avg_score (default 4.0/5.0)
2. **Target Range + Quality**: Word count within ±15% of target (300-400 words)
   **and** avg score ≥ 3.5
3. **Convergence**: Change magnitude ≤ min_change_threshold (default 0.03)
4. **Safety Limit**: Max iterations (10) reached
5. **Minimum Iterations**: Always run at least `min_iterations` (default 2)

**Methods:**
- `should_stop(state)`: Returns (bool, reason) indicating if refinement should stop
- `get_refinement_guidance(state)`: Generates prioritized guidance based on:
  - Weak levers (focus areas)
  - Improving levers (maintain momentum)
  - Word count feedback (expand/trim/maintain)
  - Current rubric scores vs. target

#### Helper Functions
- `compute_change_magnitude(prev, curr)`: Jaccard-based text similarity (0=identical, 1=completely different)
- `update_lever_history(prev, curr)`: Tracks lever directions (up/down/stable)
- `create_lever_guided_refine_prompt(...)`: Generates refinement prompt incorporating lever guidance

### 2. **Modified: `src/models/refinement.py`**

Added new function `iterative_refinement_lever_based()`:

```python
def iterative_refinement_lever_based(
    slides: List[Dict],
    initial_summary: str,
    cfg_judge: LLMConfig,
    cfg_refine: LLMConfig,
    save_callback: Callable | None = None,
    target_words: int = 350,
    min_avg_score: float = 4.0,
    min_change_threshold: float = 0.03,
    max_iterations: int = 10,
) -> Tuple[str, Dict]:
```

**Returns**: (refined_summary, metadata)

where metadata includes:
- `iterations_completed`: How many iterations were run
- `final_avg_score`: Final average rubric score
- `final_word_count`: Final summary length
- `stopping_reason`: Why refinement stopped
- `lever_history`: All rubric scores from each iteration
- `final_rubric`: Final rubric scores

**How it works:**
1. **Iteration Loop**:
   - Judge current summary (get rubric scores)
   - Extract rubric dimensions
   - Compute state metrics (word count, avg score, text change)
   - Check stopping criteria
   - If continuing: refine with lever-based guidance
   - Pairwise comparison to select best version
   - Save intermediate results

2. **Lever-Guided Refinement**:
   - Identifies weak levers (low-scoring dimensions)
   - Highlights improving levers (maintain progress)
   - Provides word count feedback
   - LLM refinement prompt includes specific guidance on what to prioritize

3. **Pairwise Selection**:
   - Compares current vs. refined candidate
   - Selects winner via ensemble judge (3 runs)
   - Ensures only improvements are accepted

### 3. **Modified: `src/evaluation/pipeline.py`**

Updated `evaluate_summary()` function with new parameters (including min_iterations guard):

```python
def evaluate_summary(
    slide_path: str,
    initial_summary: str,
    human_reference: str,
    cfg_judge,
    cfg_refine,
    out_dir: str,
    target_words: int = 350,
    use_lever_based: bool = True,          # NEW: Toggle lever-based vs. legacy
    min_avg_score: float = 4.0,           # NEW: Rubric score threshold
    min_change_threshold: float = 0.03,   # NEW: Convergence threshold
    max_iterations: int = 10,              # NEW: Safety limit
) -> Dict[str, Any]:
```

**Behavior:**
- If `use_lever_based=True`: Uses intelligent lever-based refinement
- If `use_lever_based=False`: Falls back to fixed 3-iteration refinement (legacy mode)
- Includes `refinement_metadata` in result for transparency

### 4. **Modified: `src/experiments/run_eval.py`**

Updated to use new parameters:

```bash
python src/experiments/run_eval.py [lecture_id] [force_regen] [use_lever_based]
```

**Arguments:**
- `lecture_id`: e.g., "lecture1" (default)
- `force_regen`: "yes"/"no" for regenerating S0 (default "no")
- `use_lever_based`: "yes"/"no" for lever-based refinement (default "yes")

**Example usages:**
```bash
# Use lever-based refinement (default)
python src/experiments/run_eval.py lecture1

# Force regenerate S0 with lever-based refinement
python src/experiments/run_eval.py lecture1 yes

# Use legacy fixed-iteration refinement
python src/experiments/run_eval.py lecture1 no no

# Use lever-based with regeneration
python src/experiments/run_eval.py lecture2 yes yes
```

## How Lever-Based Refinement Works

### Example Flow

```
Iteration 0: S0 generated
  - Saved as iter_0.txt

Iteration 1:
  - Judge S0 → coverage:2, faithfulness:3, organization:2, clarity:3, style:3
  - Avg score: 2.6/5 (below 4.0 target)
  - Weak levers: coverage, organization (scores ≤ 2)
  - Change: 0.0 (first iteration)
  - Guidance: "FOCUS on improving coverage and organization"
  - Refine S0 with guidance → S1
  - Pairwise(S0 vs S1) → S1 wins
  - Saved as iter_1.txt

Iteration 2:
  - Judge S1 → coverage:3, faithfulness:4, organization:3, clarity:4, style:3
  - Avg score: 3.4/5 (still below 4.0)
  - Improving levers: faithfulness, clarity (improved)
  - Weak levers: coverage (still ≤ 2)
  - Change: 0.25 (moderate change from S0)
  - Guidance: "MAINTAIN faithfulness/clarity improvements; continue improving coverage"
  - Refine with updated guidance → S2
  - Pairwise → S2 wins
  - Saved as iter_2.txt

Iteration 3:
  - Judge S2 → coverage:4, faithfulness:4, organization:4, clarity:4, style:3
  - Avg score: 4.0/5 ✓ (reached target!)
  - STOP: "High avg score: 4.0 >= 4.0"
  - Saved as final.txt

Result:
- iterations_completed: 3
- final_avg_score: 4.0
- stopping_reason: "High avg score: 4.0 >= 4.0"
- lever_history: [
    {coverage:2, faithfulness:3, organization:2, clarity:3, style:3},
    {coverage:3, faithfulness:4, organization:3, clarity:4, style:3},
    {coverage:4, faithfulness:4, organization:4, clarity:4, style:3}
  ]
```

## Stopping Criteria in Detail

### Criterion 1: High Quality
```
Condition: avg_rubric_score >= min_avg_score (default 4.0)
Reason: Summary reaches high-quality threshold
```

### Criterion 2: Target Range + Adequate Quality
```
Condition: word_count in [target * 0.85, target * 1.15] AND avg_score >= 3.5
Example: target=350, range=[297-402], current=370, score=3.6
Reason: Summary is right length AND meeting minimum quality
```

### Criterion 3: Convergence
```
Condition: change_magnitude <= min_change_threshold (default 0.03)
Duration: Only triggers after iteration 2+ (avoid premature stopping)
Reason: Summary has stabilized, further refinement unlikely to help
```

### Criterion 4: Safety Limit
```
Condition: iteration >= max_iterations (default 10)
Reason: Prevent infinite loops, ensure reasonable resource usage
```

## Integration with Existing Pipeline

The lever-based system integrates seamlessly:

1. **Backward Compatible**: `use_lever_based=False` falls back to 3-iteration refinement
2. **Transparent**: Metadata included in results showing how many iterations ran and why it stopped
3. **Output Format**: Same per-iteration and final output files as before
4. **Pairwise Evaluation**: Final pairwise comparison (S0 vs. refined) is unchanged

## Default Configuration (Tunable)

```python
# In run_eval.py
result = evaluate_summary(
    ...
    target_words=350,              # Target length (can be 300-400)
    use_lever_based=True,          # Enable lever-based refinement
    min_avg_score=4.0,             # Stop at 4.0/5.0 quality
    min_change_threshold=0.03,     # 3% change triggers convergence stop
    max_iterations=10,             # Safety limit
)
```

You can easily tune these parameters:
- Lower `min_avg_score` (e.g., 3.5) for faster refinement
- Higher `max_iterations` (e.g., 15) for more thorough refinement
- Adjust `target_words` based on lecture length needs

## Comparison: Old vs. New

| Aspect | Old (Fixed 3 iterations) | New (Lever-based) |
|--------|--------------------------|-------------------|
| **Iterations** | Always exactly 3 | Dynamic: 1-10 (adjustable) |
| **Stopping logic** | None (fixed count) | 4 intelligent criteria |
| **Quality guidance** | Judge feedback only | Lever-based + judge feedback |
| **Word count** | No specific targeting | Targets ±15% of 350 words |
| **Transparency** | No metadata | Rich metadata on why it stopped |
| **Resource usage** | Fixed (3x judge + 3x refine) | Variable (1-10x) |
| **Result quality** | Inconsistent | More consistent (targets 4.0/5.0) |

## Technical Implementation Notes

- **Jaccard Similarity**: Used for text change detection (token-based)
- **Lever Tracking**: Rubric dimensions tracked across all iterations
- **Guided Refinement**: Prompt explicitly lists priorities based on current state
- **Ensemble Pairwise**: Uses 3-run pairwise judge for stability
- **Graceful Fallback**: If refinement produces blank text, keeps previous version

## Files Modified

1. ✅ `src/models/lever_based_refinement.py` - **NEW** (240 lines)
2. ✅ `src/models/refinement.py` - Added `iterative_refinement_lever_based()` and helper function
3. ✅ `src/evaluation/pipeline.py` - Updated `evaluate_summary()` with new parameters
4. ✅ `src/experiments/run_eval.py` - Updated to use lever-based by default

## Testing

To test the implementation:

```bash
# Activate environment
conda activate dsc180a-eval

# Run with lever-based refinement
python src/experiments/run_eval.py lecture1

# Run with legacy mode for comparison
python src/experiments/run_eval.py lecture1 no no

# Force regenerate and use lever-based
python src/experiments/run_eval.py lecture1 yes yes
```

Check output in `data/summaries/refined_iterations/lecture1/result.json` for the `refinement_metadata` field showing iteration count and stopping reason.
