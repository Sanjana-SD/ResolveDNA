"""Resolution Consistency Scorer.

Measures how consistently a brand historically handled similar problems.
If retrieved cases all took the same action, consistency is high.
If retrieved cases took conflicting actions, consistency drops.
"""
from collections import Counter
from typing import List, Dict, Any, Tuple

from src.resolution.store import ResolutionCase


def compute_resolution_consistency(cases: List[ResolutionCase]) -> Dict[str, Any]:
    """Compute consistency of brand actions across retrieved historical cases.

    Returns a dict with:
        - consistency_score (float 0..1): 1.0 = all cases used the same primary action
        - dominant_action (str): the most common brand action
        - dominant_action_ratio (float): fraction of cases using the dominant action
        - action_distribution (dict): action -> count
        - conflicting (bool): True when no single action exceeds 50%
    """
    if not cases:
        return {
            "consistency_score": 0.0,
            "dominant_action": "none",
            "dominant_action_ratio": 0.0,
            "action_distribution": {},
            "conflicting": True,
            "n_cases": 0
        }

    # Use the first brand action of each case as its "primary action"
    primary_actions = []
    for c in cases:
        if c.brand_actions:
            primary_actions.append(c.brand_actions[0])
        else:
            primary_actions.append("general_response")

    action_counts = Counter(primary_actions)
    total = len(primary_actions)
    dominant_action, dominant_count = action_counts.most_common(1)[0]
    dominant_ratio = dominant_count / total

    # Consistency = dominant_ratio weighted by number of unique actions
    # Perfect consistency: one action only -> 1.0
    # Two equally split actions -> ~0.50
    n_unique = len(action_counts)
    consistency_score = dominant_ratio * (1.0 / max(1, n_unique ** 0.3))
    consistency_score = round(min(1.0, consistency_score), 3)

    return {
        "consistency_score": consistency_score,
        "dominant_action": dominant_action,
        "dominant_action_ratio": round(dominant_ratio, 3),
        "action_distribution": dict(action_counts),
        "conflicting": dominant_ratio < 0.50,
        "n_cases": total
    }


def compute_outcome_consistency(cases: List[ResolutionCase]) -> Dict[str, Any]:
    """Compute consistency of outcomes (resolved vs not) across cases."""
    if not cases:
        return {"outcome_consistency": 0.0, "resolution_rate": 0.0, "n_cases": 0}

    resolved = sum(1 for c in cases if c.observed_resolution_signal)
    total = len(cases)
    resolution_rate = resolved / total

    # Outcome consistency = how one-sided the outcomes are
    # All resolved or all unresolved -> 1.0; 50/50 split -> 0.0
    outcome_consistency = abs(resolution_rate - 0.5) * 2.0

    return {
        "outcome_consistency": round(outcome_consistency, 3),
        "resolution_rate": round(resolution_rate, 3),
        "resolved_count": resolved,
        "n_cases": total
    }
