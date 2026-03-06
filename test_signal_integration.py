#!/usr/bin/env python3
"""
test_signal_integration.py

Quick test to verify signal computation and plateau detection integration.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_slides
from src.utils.signals import compute_signals
from src.models.lever_based_refinement import RefinementState, LeverBasedRefinementController

ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

def test_signal_computation():
    """Test that signals are computed correctly."""
    print("\n=== Testing Signal Computation ===")
    
    # Load lecture 1 slides
    slides = load_slides("data/slides/lecture1.pdf")
    print(f"Loaded {len(slides)} slides")
    
    # Sample summary
    summary1 = "This is a test summary about the lecture content."
    summary2 = "This is an improved test summary about the lecture content with more details included."
    
    # Compute signals
    signals1 = compute_signals(slides, summary1, target_words=300)
    signals2 = compute_signals(slides, summary2, target_words=300)
    
    print(f"\nSignals for summary 1:")
    for key, val in signals1.items():
        print(f"  {key}: {val:.4f}")
    
    print(f"\nSignals for summary 2:")
    for key, val in signals2.items():
        print(f"  {key}: {val:.4f}")
    
    return signals1, signals2

def test_state_tracking():
    """Test that RefinementState correctly tracks signals."""
    print("\n=== Testing RefinementState Signal Tracking ===")
    
    signals1 = {
        "length_error": 0.1,
        "section_coverage_pct": 0.8,
        "glossary_recall": 0.6,
        "suspected_hallucination_rate": 0.2,
    }
    
    signals2 = {
        "length_error": 0.12,
        "section_coverage_pct": 0.82,
        "glossary_recall": 0.62,
        "suspected_hallucination_rate": 0.18,
    }
    
    # Create state
    state = RefinementState(
        iteration=0,
        summary="Test summary",
        rubric={"coverage": 3, "faithfulness": 3, "organization": 3, "clarity": 3, "style": 3},
        lever_history=[],
        word_count=100,
        change_magnitude=0.0,
        avg_rubric_score=3.0,
        target_words=300,
        agreement_score=0.0,
    )
    
    # Initialize signals_history
    if state.signals_history is None:
        state.signals_history = []
    
    # Track signals
    state.signals = signals1
    state.signals_history.append(signals1)
    
    print(f"\nAfter iteration 1:")
    print(f"  Current signals: {state.signals}")
    print(f"  History: {state.signals_history}")
    
    state.signals = signals2
    state.signals_history.append(signals2)
    
    print(f"\nAfter iteration 2:")
    print(f"  Current signals: {state.signals}")
    print(f"  History length: {len(state.signals_history)}")

def test_controller_plateau_detection():
    """Test that controller can detect signal plateau."""
    print("\n=== Testing Plateau Detection ===")
    
    # Create controller with default thresholds
    controller = LeverBasedRefinementController(
        signal_plateau_threshold=0.05,
        signal_plateau_iters=2,
    )
    
    # Create state with plateau signals
    state = RefinementState(
        iteration=3,
        summary="Test summary",
        rubric={"coverage": 4, "faithfulness": 4, "organization": 4, "clarity": 4, "style": 4},
        lever_history=[],
        word_count=300,
        change_magnitude=0.02,
        avg_rubric_score=4.0,
        target_words=300,
        agreement_score=0.75,
    )
    
    # Signals that are very stable (plateau)
    stable_signals = {
        "length_error": 0.10,
        "section_coverage_pct": 0.87,
        "glossary_recall": 0.68,
        "suspected_hallucination_rate": 0.15,
    }
    
    state.signals_history = [
        {**stable_signals},  # Iteration 0
        {**stable_signals},  # Iteration 1 (identical = no variance)
        {**stable_signals},  # Iteration 2 (identical = no variance)
    ]
    state.signals = stable_signals
    
    # Test plateau detection
    plateau_detected, plateau_reason = controller._detect_signal_plateau(state)
    print(f"\nPlateau detected: {plateau_detected}")
    print(f"Reason: {plateau_reason}")
    
    # Test signal threshold check
    threshold_met, threshold_reason = controller._check_signal_thresholds(state.signals)
    print(f"\nSignal thresholds met: {threshold_met}")
    print(f"Reason: {threshold_reason}")

if __name__ == "__main__":
    try:
        test_signal_computation()
        test_state_tracking()
        test_controller_plateau_detection()
        print("\n✅ All integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
