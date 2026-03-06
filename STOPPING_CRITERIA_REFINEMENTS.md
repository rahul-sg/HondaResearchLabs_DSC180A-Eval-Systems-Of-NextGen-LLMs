# Stopping Criteria & Hybrid Scoring Refinements

## Overview

This update implements tighter stopping rules, minimum iteration guardrails, and semantic agreement-based quality checks for the lever-based refinement system.

## Key Changes

### 1. **Tightened Quality Default**
- **Old**: `min_avg_score = 4.0`
- **New**: `min_avg_score = 4.5` (adjustable via CLI)
- Rationale: Ensures higher-quality summaries before stopping

### 2. **Minimum Iterations Guard**
- **New parameter**: `min_iterations = 2` (default)
- Behavior: Controller always enforces at least this many iterations before evaluating stopping conditions
- Purpose: Prevents premature stopping even if quality target is met early

### 3. **Hybrid Quality Check**
- **Old criterion 1**: High avg score AND length in target range
- **New criterion 1**: High avg score AND (length in target range **OR** semantic agreement ≥ threshold)
- **New parameter**: `min_agreement = 0.7` (meteor-based)
- Benefit: Allows stopping on semantic quality alone when length is suboptimal but agreement is strong

### 4. **Criterion Combination**
The controller now combines multiple levers:
- Quality (rubric average)
- Length proximity (±15% of target)
- Semantic similarity (meteor score vs. reference)
- Change convergence (stability check)
- Safety limits (max iterations)

## Implementation Details

### Files Modified

#### `src/models/lever_based_refinement.py`
- Added `min_iterations` and `min_agreement` parameters to `LeverBasedRefinementController`
- Updated `should_stop()` logic:
  - Checks `iteration < min_iterations` first (guard clause)
  - Criterion 1 now allows score + (length OR agreement)
  - Added `getattr(state, "agreement_score", 0.0)` for safe access
- Updated docstring to explain new stopping criteria

#### `src/models/refinement.py`
- Added NLTK imports and downloads for meteor computation
- Extended `RefinementState` dataclass with `agreement_score` field
- Updated `iterative_refinement_lever_based()` signature:
  - Added `min_iterations`, `human_reference`, and `min_agreement` parameters
  - Computes meteor agreement at each iteration if reference is provided
  - Passes all parameters to controller
- Updated metadata to include `final_agreement`

#### `src/evaluation/pipeline.py`
- Updated `evaluate_summary()` signature with new parameters
- Updated docstrings to document `min_iterations` and `min_agreement`
- Tightened default `min_avg_score` to 4.5
- Passes reference summary and all thresholds to lever-based refinement

#### `src/experiments/run_eval.py`
- Extended CLI argument parsing (args 3-8 for stopping parameters)
- Arg 3: `min_avg_score` (default 4.5)
- Arg 4: `min_change_threshold` (default 0.03)
- Arg 5: `max_iterations` (default 10)
- Arg 6: `min_iterations` (default 2)
- Arg 7: `min_agreement` (default 0.7)
- Updated print statements and metadata output

### Documentation Updates

#### `README.md`
- Extended parameter list with 7 CLI args
- Updated note on stopping behavior
- Clarified hybrid quality check (length OR agreement)

#### `LEVER_SYSTEM_QUICK_REFERENCE.md`
- Updated CLI example with all parameters
- Updated metadata example to include `final_agreement`
- Updated comparison table

#### `IMPLEMENTATION_SUMMARY.md`
- Updated configuration tuning section
- Added `min_agreement` to default parameters
- Noted tightened quality threshold (4.5 instead of 4.0)

#### `LEVER_BASED_REFINEMENT_GUIDE.md`
- Updated stopping criteria section
- Added `min_iterations` and `min_agreement` documentation
- Clarified hybrid criterion logic

## Example Usage

```bash
# Defaults: min_avg=4.5, min_change=0.03, max_iters=10, min_iters=2, min_agree=0.7
python src/experiments/run_eval.py lecture1

# Force regenerate with custom strict quality
python src/experiments/run_eval.py lecture1 yes yes 4.8 0.02 15 2 0.75

# Use lower quality for faster refinement
python src/experiments/run_eval.py lecture2 no yes 4.0 0.05 8 2 0.65
```

## Stopping Condition Flow

```
┌─ Check: iteration < min_iterations?
│  Yes → Continue (don't evaluate other criteria)
│
├─ Criterion 1: Quality + (Length OR Agreement)?
│  avg_score ≥ min_avg_score AND (in_range OR agreement ≥ min_agreement)
│  → STOP "High avg + length/agreement achieved"
│
├─ Criterion 2: Length in range + Adequate quality?
│  in_range AND avg_score ≥ 3.5
│  → STOP "Target range + minimum quality achieved"
│
├─ Criterion 3: Convergence?
│  change_magnitude ≤ min_change_threshold
│  → STOP "Summary stabilized"
│
└─ Criterion 4: Safety limit?
   iteration ≥ max_iterations
   → STOP "Max iterations reached"
```

## Metadata Fields

The refinement process now tracks:
- `iterations_completed` - How many iterations ran
- `final_avg_score` - Final rubric average
- `final_word_count` - Final summary length
- `final_agreement` - Final meteor similarity (NEW)
- `stopping_reason` - Explanation of why it stopped
- `lever_history` - All rubric scores per iteration
- `final_rubric` - Final breakdown by dimension

## Benefits

| Aspect | Impact |
|--------|--------|
| **Quality Control** | Tighter defaults ensure higher baseline quality |
| **Flexibility** | Multiple stopping criteria allow adaptive behavior |
| **Minimum Work** | `min_iterations` prevents under-refinement |
| **Semantic Awareness** | Agreement threshold provides quality signal without length constraint |
| **Transparency** | All parameters exposed via CLI for tuning |
| **Backward Compatible** | Can still adjust all thresholds dynamically |

## Testing Recommendations

1. **Verify minimum iterations**:
   ```bash
   python src/experiments/run_eval.py lecture1 no yes 5.0 0.01 10 2 0.9
   # Should run at least 2 iterations even if first iteration achieves 5.0
   ```

2. **Test semantic agreement fallback**:
   ```bash
   python src/experiments/run_eval.py lecture2 no yes 4.8 0.03 10 2 0.8
   # Should stop if avg ≥ 4.8 AND agreement ≥ 0.8 (even if length off)
   ```

3. **Confirm metadata output**:
   - Check `data/summaries/refined_iterations/lectureX/result.json`
   - Verify `final_agreement` and `stopping_reason` fields present

## Future Enhancements

- Add per-dimension agreement scores (coverage agreement, clarity agreement, etc.)
- Implement dynamic threshold adjustment based on convergence patterns
- Add early stopping detection (no improvement for N iterations)
- Support weighted criteria combinations (not just AND/OR logic)
