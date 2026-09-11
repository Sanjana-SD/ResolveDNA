"""Unit tests for Resolution Consistency Scorer."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision.consistency import compute_resolution_consistency, compute_outcome_consistency
from src.resolution.store import ResolutionCase


def make_case(case_id: str, brand_actions: list, resolved: bool = True):
    return ResolutionCase(
        case_id=case_id,
        customer_problem="test customer problem",
        intent="delivery_delay",
        brand_actions=brand_actions,
        customer_reactions=[],
        final_state="RESOLVED" if resolved else "UNRESOLVED",
        observed_resolution_signal=resolved,
        trajectory_length=2,
        created_at="2017-10-10",
        resolution_pattern=brand_actions[0] if brand_actions else "general_response",
        evidence_score=1.0
    )


def test_consistency_unanimous_actions():
    cases = [
        make_case("1", ["request_order_id"], True),
        make_case("2", ["request_order_id"], True),
        make_case("3", ["request_order_id"], True),
    ]
    res = compute_resolution_consistency(cases)
    assert res["dominant_action"] == "request_order_id"
    assert res["dominant_action_ratio"] == 1.0
    assert res["conflicting"] is False


def test_consistency_split_actions():
    cases = [
        make_case("1", ["request_order_id"], True),
        make_case("2", ["ask_for_dm"], False),
        make_case("3", ["provide_help_link"], False),
    ]
    res = compute_resolution_consistency(cases)
    assert res["dominant_action_ratio"] <= 0.4
    assert res["conflicting"] is True
