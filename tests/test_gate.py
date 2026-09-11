"""Unit tests for Evidence Sufficiency Gate."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision.gate import decide, DEFAULT_THRESHOLDS
from src.resolution.store import ResolutionCase


def make_case(case_id: str, brand_action: str = "provide_help_link", resolved: bool = True):
    return ResolutionCase(
        case_id=case_id,
        customer_problem="test customer problem",
        intent="prime_video_issue",
        brand_actions=[brand_action],
        customer_reactions=[],
        final_state="RESOLVED" if resolved else "UNRESOLVED",
        observed_resolution_signal=resolved,
        trajectory_length=2,
        created_at="2017-10-10",
        resolution_pattern=brand_action,
        evidence_score=1.0
    )


def test_gate_escalates_on_low_intent_confidence():
    consistency = {"consistency_score": 0.9, "dominant_action": "request_order_id"}
    outcome = {"outcome_consistency": 0.9, "positive_resolution_rate": 0.8}
    cases = [(make_case("1", "request_order_id"), 0.8)]
    
    result = decide(
        intent="delivery_delay",
        intent_confidence=0.3, # < 0.50
        resolution_confidence=0.8,
        consistency=consistency,
        outcome=outcome,
        retrieved_cases=cases
    )
    assert result["decision"] == "ESCALATE"
    assert any("Intent confidence too low" in f for f in result["escalation_factors"])


def test_gate_escalates_on_low_resolution_confidence():
    consistency = {"consistency_score": 0.9, "dominant_action": "request_order_id"}
    outcome = {"outcome_consistency": 0.9, "positive_resolution_rate": 0.8}
    cases = [(make_case("1", "request_order_id"), 0.8)]
    
    result = decide(
        intent="delivery_delay",
        intent_confidence=0.8,
        resolution_confidence=0.2, # < 0.35
        consistency=consistency,
        outcome=outcome,
        retrieved_cases=cases
    )
    assert result["decision"] == "ESCALATE"
    assert any("Resolution confidence too low" in f for f in result["escalation_factors"])


def test_gate_auto_handles_when_all_thresholds_met():
    consistency = {"consistency_score": 0.85, "dominant_action": "provide_help_link"}
    outcome = {"outcome_consistency": 0.9, "positive_resolution_rate": 0.8}
    cases = [(make_case(str(i)), 0.8) for i in range(3)]
    
    result = decide(
        intent="prime_video_issue",
        intent_confidence=0.90,
        resolution_confidence=0.75,
        consistency=consistency,
        outcome=outcome,
        retrieved_cases=cases
    )
    assert result["decision"] == "AUTO_HANDLE"
