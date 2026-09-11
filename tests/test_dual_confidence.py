"""Unit tests for Dual Confidence Engine."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision.dual_confidence import compute_resolution_confidence
from src.resolution.store import ResolutionCase


def make_case(case_id: str):
    return ResolutionCase(
        case_id=case_id,
        customer_problem="Prime video error code 5004",
        intent="prime_video_issue",
        brand_actions=["provide_help_link"],
        customer_reactions=[],
        final_state="RESOLVED",
        observed_resolution_signal=True,
        trajectory_length=2,
        created_at="2017-10-10",
        resolution_pattern="provide_help_link",
        evidence_score=1.0
    )


def test_confidence_separation_empty_cases():
    res = compute_resolution_confidence(
        intent_confidence=0.95,
        retrieved_cases=[],
        consistency={},
        outcome={}
    )
    assert res["resolution_confidence"] == 0.0


def test_high_resolution_confidence_with_strong_support():
    cases = [(make_case(str(i)), 0.85) for i in range(5)]
    consistency = {"consistency_score": 0.9}
    outcome = {"positive_resolution_rate": 0.9}
    
    res = compute_resolution_confidence(
        intent_confidence=0.90,
        retrieved_cases=cases,
        consistency=consistency,
        outcome=outcome
    )
    assert res["resolution_confidence"] > 0.6
