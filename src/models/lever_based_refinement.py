"""
lever_based_refinement.py

Implements lever-based iterative refinement with dynamic stopping criteria.
Tracks rubric dimensions (levers) and adjusts refinement priorities based on:
- Rubric score improvements
- Word count proximity to target
- Overall convergence/stability

This replaces fixed iteration counts with intelligent, criterion-based refinement.
"""

from typing import Dict, List, Callable, Tuple, Any
import re
from dataclasses import dataclass


@dataclass
class RefinementState:
    """Tracks refinement progress across iterations."""
    iteration: int
    summary: str
    rubric: Dict[str, float]  # {coverage, faithfulness, organization, clarity, style}
    lever_history: List[Dict[str, float]]  # Historical rubric scores
    word_count: int
    change_magnitude: float  # 0-1, how much summary changed
    avg_rubric_score: float  # Average of rubric dimensions
    target_words: int = 350  # Target word count (default)
    agreement_score: float = 0.0  # Semantic similarity to reference (meteor)
    signals: Dict[str, float] | None = None  # Current signals dict {length_error, section_coverage_pct, ...}
    signals_history: List[Dict[str, float]] | None = None  # Historical signals tracking
    quality_score: float = 0.0  # Composite 0..1 quality score for trend-aware stopping
    quality_history: List[float] | None = None  # Historical composite quality scores

    def get_weak_levers(self) -> List[str]:
        """Return levers (rubric dimensions) with scores <= 2."""
        return [k for k, v in self.rubric.items() if v <= 2]

    def get_improving_levers(self) -> List[str]:
        """Return levers that improved in the last iteration."""
        if len(self.lever_history) < 2:
            return []
        
        prev_rubric = self.lever_history[-2]
        curr_rubric = self.rubric
        
        improving = []
        for lever in prev_rubric:
            if curr_rubric.get(lever, 0) > prev_rubric.get(lever, 0):
                improving.append(lever)
        return improving

    def log_state(self) -> str:
        """Generate a log string of current state."""
        base = (
            f"Iter {self.iteration} | "
            f"Score: {self.avg_rubric_score:.2f}/5 | "
            f"Quality: {self.quality_score:.3f} | "
            f"Words: {self.word_count} | "
            f"Change: {self.change_magnitude:.3f} | "
            f"Rubric: {self.rubric}"
        )
        
        # Add signals if available
        if self.signals:
            signals_str = " | ".join([f"{k}: {v:.3f}" for k, v in self.signals.items()])
            base += f" | Signals: {{{signals_str}}}"
        
        return base


class LeverBasedRefinementController:
    """
    Manages lever-based iterative refinement with intelligent stopping.
    
    Stopping criteria (any one can trigger stopping), but the controller
    always enforces a minimum number of iterations before evaluating
    these conditions. The default configuration also tightens the quality
    requirement and combines word-count range or semantic agreement with score.
    
    1. Average rubric score >= min_avg_score **and** (length within tolerance
       OR agreement_score >= min_agreement)
    2. Word count within ±target_word_tolerance *and* avg score >=3.5
    3. Change magnitude <= min_change_threshold (convergence)
    4. Max iterations reached
    5. Always run at least `min_iterations` iterations before stopping
    6. Agreement threshold `min_agreement` provides an alternate stop condition
    """

    def __init__(
        self,
        target_words: int = 350,
        min_avg_score: float = 4.5,
        min_change_threshold: float = 0.03,
        target_word_tolerance: float = 0.15,  # ±15%
        max_iterations: int = 12,
        min_iterations: int = 4,
        min_agreement: float = 0.7,              # meteor threshold for quality+agreement
        # Signal-based stopping thresholds (domain-agnostic)
        signal_max_length_error: float = 0.15,
        signal_min_coverage: float = 0.85,
        signal_min_recall: float = 0.55,
        signal_max_hallucination: float = 0.55,
        # Progressive strict phase thresholds
        signal_early_phase_iterations: int = 3,
        signal_max_length_error_strict: float = 0.12,
        signal_min_coverage_strict: float = 0.88,
        signal_recall_strict: float = 0.62,
        signal_hallucination_strict: float = 0.48,
        # Trend/plateau controls
        plateau_window: int = 3,
        plateau_eps_metric: float = 0.03,
        plateau_eps_score: float = 0.02,
        trend_window: int = 3,
        stall_min_improvement_halluc: float = 0.02,
        stall_min_improvement_recall: float = 0.02,
        stall_min_score_gain: float = 0.015,
        # Decision thresholds
        pass_quality_threshold: float = 0.74,
        borderline_quality_threshold: float = 0.68,
    ):
        self.target_words = target_words
        self.min_avg_score = min_avg_score
        self.min_change_threshold = min_change_threshold
        self.target_word_tolerance = target_word_tolerance
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.min_agreement = min_agreement
        # Signal thresholds
        self.signal_max_length_error = signal_max_length_error
        self.signal_min_coverage = signal_min_coverage
        self.signal_min_recall = signal_min_recall
        self.signal_max_hallucination = signal_max_hallucination
        # Progressive thresholds
        self.signal_early_phase_iterations = signal_early_phase_iterations
        self.signal_max_length_error_strict = signal_max_length_error_strict
        self.signal_min_coverage_strict = signal_min_coverage_strict
        self.signal_recall_strict = signal_recall_strict
        self.signal_hallucination_strict = signal_hallucination_strict
        # Trend/plateau controls
        self.plateau_window = plateau_window
        self.plateau_eps_metric = plateau_eps_metric
        self.plateau_eps_score = plateau_eps_score
        self.trend_window = trend_window
        self.stall_min_improvement_halluc = stall_min_improvement_halluc
        self.stall_min_improvement_recall = stall_min_improvement_recall
        self.stall_min_score_gain = stall_min_score_gain
        # Decision thresholds
        self.pass_quality_threshold = pass_quality_threshold
        self.borderline_quality_threshold = borderline_quality_threshold

    def _check_signal_thresholds(self, signals: Dict[str, float], iteration: int = 1) -> Tuple[bool, str]:
        """
        Check if all signal-based quality metrics meet thresholds.
        Uses progressive thresholds: relaxed for early iterations, stricter for later ones.
        
        Returns:
            (all_met: bool, reason: str)
        """
        if not signals:
            return False, ""
        
        # Determine which thresholds to use based on iteration
        is_early_phase = iteration <= self.signal_early_phase_iterations
        max_length_error = self.signal_max_length_error if is_early_phase else self.signal_max_length_error_strict
        min_coverage = self.signal_min_coverage if is_early_phase else self.signal_min_coverage_strict
        min_recall = self.signal_min_recall if is_early_phase else self.signal_recall_strict
        max_hallucination = self.signal_max_hallucination if is_early_phase else self.signal_hallucination_strict
        
        checks = []
        all_met = True
        
        length_error = signals.get("length_error", float("inf"))
        if length_error <= max_length_error:
            checks.append(f"length_error {length_error:.3f} <= {max_length_error}")
        else:
            checks.append(f"length_error {length_error:.3f} > {max_length_error}")
            all_met = False
        
        coverage = signals.get("section_coverage_pct", 0.0)
        if coverage >= min_coverage:
            checks.append(f"coverage {coverage:.3f} >= {min_coverage}")
        else:
            checks.append(f"coverage {coverage:.3f} < {min_coverage}")
            all_met = False
        
        recall = signals.get("glossary_recall", 0.0)
        if recall >= min_recall:
            checks.append(f"recall {recall:.3f} >= {min_recall}")
        else:
            checks.append(f"recall {recall:.3f} < {min_recall}")
            all_met = False
        
        hallucination = signals.get("suspected_hallucination_rate", float("inf"))
        if hallucination < max_hallucination:
            checks.append(f"hallucination {hallucination:.3f} < {max_hallucination}")
        else:
            checks.append(f"hallucination {hallucination:.3f} >= {max_hallucination}")
            all_met = False
        
        # Add phase info to reason
        phase_info = f"(early phase, iter {iteration})" if is_early_phase else f"(strict phase, iter {iteration})"
        reason = " AND ".join(checks) + " " + phase_info
        return all_met, reason

    def _detect_signal_plateau(self, state: RefinementState) -> Tuple[bool, str]:
        """
        Detect if signals have plateaued (low variance for N consecutive iterations).
        
        Returns:
            (has_plateau: bool, reason: str)
        """
        if not state.signals_history or len(state.signals_history) < self.plateau_window:
            return False, ""
        
        # Get last N signal dicts
        recent_signals = state.signals_history[-self.plateau_window:]
        
        # Check variance for each metric
        signal_keys = ["length_error", "section_coverage_pct", "glossary_recall", "suspected_hallucination_rate"]
        all_plateaued = True
        plateau_reasons = []
        
        for key in signal_keys:
            values = [s.get(key, 0.0) for s in recent_signals]
            if not values:
                continue
            
            min_val = min(values)
            max_val = max(values)
            variance = max_val - min_val
            
            if variance <= self.plateau_eps_metric:
                plateau_reasons.append(f"{key}: {variance:.4f} variance")
            else:
                all_plateaued = False
                plateau_reasons.append(f"{key}: {variance:.4f} variance (still changing)")
        
        reason = " | ".join(plateau_reasons)
        return all_plateaued, reason

    def _detect_quality_plateau(self, state: RefinementState) -> Tuple[bool, str]:
        """Detect if composite quality score has plateaued."""
        if not state.quality_history or len(state.quality_history) < self.plateau_window:
            return False, ""

        recent_scores = state.quality_history[-self.plateau_window:]
        score_span = max(recent_scores) - min(recent_scores)
        if score_span <= self.plateau_eps_score:
            return True, f"quality span {score_span:.4f} <= {self.plateau_eps_score:.4f}"
        return False, f"quality span {score_span:.4f} > {self.plateau_eps_score:.4f}"

    def _detect_stalled_trend(self, state: RefinementState) -> Tuple[bool, str]:
        """Detect stalled progress in hallucination/recall and score gain."""
        if (
            not state.signals_history
            or len(state.signals_history) < self.trend_window
            or not state.quality_history
            or len(state.quality_history) < self.trend_window
        ):
            return False, ""

        start_signals = state.signals_history[-self.trend_window]
        end_signals = state.signals_history[-1]

        halluc_start = start_signals.get("suspected_hallucination_rate", 1.0)
        halluc_end = end_signals.get("suspected_hallucination_rate", 1.0)
        recall_start = start_signals.get("glossary_recall", 0.0)
        recall_end = end_signals.get("glossary_recall", 0.0)

        halluc_improvement = halluc_start - halluc_end
        recall_improvement = recall_end - recall_start
        score_gain = state.quality_history[-1] - state.quality_history[-self.trend_window]

        stalled = (
            halluc_improvement < self.stall_min_improvement_halluc
            and recall_improvement < self.stall_min_improvement_recall
            and score_gain < self.stall_min_score_gain
        )

        reason = (
            f"halluc_improvement {halluc_improvement:.4f} < {self.stall_min_improvement_halluc:.4f}, "
            f"recall_improvement {recall_improvement:.4f} < {self.stall_min_improvement_recall:.4f}, "
            f"score_gain {score_gain:.4f} < {self.stall_min_score_gain:.4f}"
        )
        return stalled, reason

    def should_stop(self, state: RefinementState) -> Tuple[bool, str]:
        """
        Determine if refinement should stop.
        
        Domain-agnostic decision table:
        - pass: strict criteria pass, or quality high and plateaued
        - borderline: quality moderate and stable
        - stalled: trend not improving (safety stop)
        - hard stop at max_iterations
        
        Returns:
            (should_stop: bool, reason: str)
        """
        
        # Guarantee minimum iterations
        if state.iteration < self.min_iterations:
            return False, f"Iteration {state.iteration} < min_iterations {self.min_iterations}"

        # Precompute core conditions
        lower_bound = self.target_words * (1 - self.target_word_tolerance)
        upper_bound = self.target_words * (1 + self.target_word_tolerance)
        in_range = lower_bound <= state.word_count <= upper_bound
        quality_score = getattr(state, "quality_score", 0.0)

        quality_plateau, quality_plateau_reason = self._detect_quality_plateau(state)

        signal_plateau = False
        signal_plateau_reason = ""
        if state.signals:
            signal_plateau, signal_plateau_reason = self._detect_signal_plateau(state)

        # Check signal criteria
        signal_criteria_met = False
        signal_reason = ""
        
        if state.signals:
            signals_met, signals_reason = self._check_signal_thresholds(state.signals, iteration=state.iteration)
            if signals_met:
                signal_criteria_met = True
                signal_reason = signals_reason
            else:
                # Debug: show signal evaluation when not met
                print(f"[SIGNALS] Thresholds not met: {signals_reason}")
                if state.signals_history and len(state.signals_history) >= self.plateau_window:
                    plateau_detected, plateau_reason = self._detect_signal_plateau(state)
                    print(f"[SIGNALS] Plateau check: {plateau_reason}")

        # PASS condition A: strict signal thresholds + quality/length sanity
        if signal_criteria_met and in_range and quality_score >= self.pass_quality_threshold:
            return True, (
                "pass: strict thresholds met and quality high - "
                f"signals: {signal_reason} | quality {quality_score:.3f} >= {self.pass_quality_threshold:.3f}"
            )

        # PASS condition B: quality high + stable plateau
        if quality_score >= self.pass_quality_threshold and quality_plateau:
            return True, (
                "pass: high quality plateau - "
                f"quality {quality_score:.3f} >= {self.pass_quality_threshold:.3f}; {quality_plateau_reason}"
            )

        # BORDERLINE condition: moderate quality + stable and no strong degradation
        if (
            quality_score >= self.borderline_quality_threshold
            and quality_plateau
            and (not state.signals or signal_plateau)
        ):
            signal_part = signal_plateau_reason if signal_plateau_reason else "signal plateau unavailable"
            return True, (
                "borderline: moderate quality with stability - "
                f"quality {quality_score:.3f} >= {self.borderline_quality_threshold:.3f}; "
                f"{quality_plateau_reason}; {signal_part}"
            )

        # STALLED condition: trend not improving enough
        stalled, stalled_reason = self._detect_stalled_trend(state)
        if stalled:
            return True, f"stalled: trend plateaued - {stalled_reason}"

        # Additional convergence safety valve
        if state.iteration > 2 and state.change_magnitude <= self.min_change_threshold:
            return True, (
                f"converged: change magnitude {state.change_magnitude:.4f} "
                f"<= threshold {self.min_change_threshold:.4f}"
            )

        # Hard stop at max iterations
        if state.iteration >= self.max_iterations:
            return True, f"max_iters: reached {state.iteration} without pass/borderline/stalled condition"

        return False, ""

    def get_refinement_guidance(self, state: RefinementState) -> str:
        """
        Generate guidance for the refiner based on current lever state.
        
        Prioritizes weak levers and suggests focus areas.
        """
        weak_levers = state.get_weak_levers()
        improving_levers = state.get_improving_levers()
        
        guidance_parts = []
        
        # Priority 1: Address weak levers
        if weak_levers:
            guidance_parts.append(
                f"PRIORITY (levers ≤2): Focus on improving {', '.join(weak_levers)}."
            )
        
        # Priority 2: Maintain improving levers
        if improving_levers:
            guidance_parts.append(
                f"MAINTAIN: Keep strengthening {', '.join(improving_levers)} "
                f"(already improving)."
            )
        
        # Word count guidance
        lower_bound = self.target_words * (1 - self.target_word_tolerance)
        upper_bound = self.target_words * (1 + self.target_word_tolerance)
        
        if state.word_count < lower_bound:
            guidance_parts.append(
                f"EXPAND: Current {state.word_count} words is below target "
                f"{self.target_words} (min {int(lower_bound)}). "
                f"Add more detail while maintaining quality."
            )
        elif state.word_count > upper_bound:
            guidance_parts.append(
                f"TRIM: Current {state.word_count} words exceeds target "
                f"{self.target_words} (max {int(upper_bound)}). "
                f"Remove redundancy and secondary details."
            )
        else:
            guidance_parts.append(f"LENGTH OK: {state.word_count} words (target {self.target_words}).")
        
        # Rubric score guidance
        guidance_parts.append(
            f"Current avg rubric: {state.avg_rubric_score:.2f}/5. "
            f"Target: {self.min_avg_score}/5."
        )
        
        return " ".join(guidance_parts)


def compute_change_magnitude(prev_text: str, curr_text: str) -> float:
    """
    Compute change magnitude between two texts using Jaccard similarity.
    Returns 0 for identical, 1 for completely different.
    """
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if t]

    ta = set(tokenize(prev_text))
    tb = set(tokenize(curr_text))
    
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    
    jaccard = len(ta & tb) / len(ta | tb)
    return 1.0 - jaccard  # 1 - similarity = change


def update_lever_history(
    prev_rubric: Dict[str, int],
    curr_rubric: Dict[str, int],
) -> Dict[str, str]:
    """
    Compare rubric dimensions and return lever directions.
    
    Returns:
        {lever_name: 'up' | 'down' | 'stable'}
    """
    directions = {}
    for lever in curr_rubric:
        prev_score = prev_rubric.get(lever, 0)
        curr_score = curr_rubric.get(lever, 0)
        
        if curr_score > prev_score:
            directions[lever] = 'up'
        elif curr_score < prev_score:
            directions[lever] = 'down'
        else:
            directions[lever] = 'stable'
    
    return directions


def create_lever_guided_refine_prompt(
    slides_text: str,
    summary: str,
    judge_feedback: Dict[str, Any],
    controller: LeverBasedRefinementController,
    state: RefinementState,
) -> str:
    """
    Create a refinement prompt that incorporates lever-based guidance.
    """
    
    guidance = controller.get_refinement_guidance(state)
    
    # Convert judge feedback to readable format
    feedback_text = ""
    if isinstance(judge_feedback, dict):
        strengths = judge_feedback.get("two_strengths", [])
        issues = judge_feedback.get("two_issues", [])
        evidence = judge_feedback.get("faithfulness_evidence", [])
        
        if strengths:
            feedback_text += f"\nSTRENGTHS: {', '.join(strengths)}"
        if issues:
            feedback_text += f"\nISSUES: {', '.join(issues)}"
        if evidence:
            feedback_text += f"\nFAITHFULNESS EVIDENCE: {', '.join(evidence)}"
    
    prompt = f"""
You are revising a student lecture summary using:
1) A **subset of the lecture slides** (content slides only)
2) **Structured judge feedback** and rubric scores
3) **Lever-based guidance** on what to prioritize

[Slides Provided]
{slides_text}

[Current Summary]
{summary}

[Judge Feedback]
{feedback_text}

[LEVER-BASED GUIDANCE (from previous iterations)]
{guidance}

[Current Rubric Scores (1-5)]
- Coverage: {state.rubric.get('coverage', '?')}
- Faithfulness: {state.rubric.get('faithfulness', '?')}
- Organization: {state.rubric.get('organization', '?')}
- Clarity: {state.rubric.get('clarity', '?')}
- Style: {state.rubric.get('style', '?')}

TASK:
Rewrite the summary to address the guidance above while:
- Incorporating missing key ideas
- Fixing inaccuracies
- Improving structure and flow
- Removing redundancy
- Staying within ±15% of {int(state.target_words * 1.15)} words (currently {state.word_count})
- Remaining strictly grounded in the slides

CRITICAL RULES:
- Use ONLY information from these slides
- Do NOT include syllabus, instructor info, grading, or logistics
- Do NOT reference topics not shown above
- NO hallucinations

Return ONLY the improved summary.
"""
    return prompt.strip()
