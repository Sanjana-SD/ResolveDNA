"""Next Best Action derivation.

Determines the evidence-backed recommended action BEFORE the LLM
generates any text. The LLM is only responsible for converting this
structured action into natural language — it does NOT invent the action.
"""
from typing import List, Dict, Any, Tuple
from collections import Counter

from src.resolution.store import ResolutionCase


# Canonical action labels and their customer-facing descriptions
ACTION_DESCRIPTIONS = {
    "provide_link": "Provide a self-service help link or resource",
    "troubleshoot": "Provide step-by-step troubleshooting instructions",
    "clarify_info": "Request additional details from the customer (order number, device, etc.)",
    "explain_policy": "Explain the relevant Amazon policy or timeline",
    "request_dm": "Ask the customer to continue via direct message for account-specific help",
    "apologize": "Acknowledge the issue and apologize",
    "general_response": "Provide a general acknowledgment and offer to help",
}


def derive_next_best_action(
    intent: str,
    retrieved_cases: List[Tuple[ResolutionCase, float]],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive the next-best action from historical evidence.

    The action is the one most commonly used in resolved historical cases.
    If no resolved cases are available, falls back to the most common action
    across all retrieved cases.
    """
    if not retrieved_cases:
        return {
            "action": "general_response",
            "action_description": ACTION_DESCRIPTIONS["general_response"],
            "evidence_basis": "no_historical_cases",
            "action_source": "fallback",
        }

    # Prefer actions from resolved cases
    resolved_actions = []
    all_actions = []

    for case, score in retrieved_cases:
        primary = case.brand_actions[0] if case.brand_actions else "general_response"
        all_actions.append(primary)
        if case.observed_resolution_signal:
            resolved_actions.append(primary)

    if resolved_actions:
        action_counts = Counter(resolved_actions)
        best_action = action_counts.most_common(1)[0][0]
        n_resolved = len(resolved_actions)
        evidence_basis = f"action used in {action_counts[best_action]}/{n_resolved} resolved cases"
        action_source = "resolved_cases"
    else:
        action_counts = Counter(all_actions)
        best_action = action_counts.most_common(1)[0][0]
        evidence_basis = f"most common action across {len(all_actions)} cases (none confirmed resolved)"
        action_source = "all_cases_fallback"

    # If consistency analysis provides a dominant action, cross-check
    dominant = consistency.get("dominant_action", "")
    if dominant and dominant != best_action and consistency.get("dominant_action_ratio", 0) > 0.7:
        best_action = dominant
        evidence_basis = f"dominant action ({consistency['dominant_action_ratio']*100:.0f}% of cases)"
        action_source = "consistency_dominant"

    description = ACTION_DESCRIPTIONS.get(best_action, f"Perform action: {best_action}")

    return {
        "action": best_action,
        "action_description": description,
        "evidence_basis": evidence_basis,
        "action_source": action_source,
        "action_distribution": dict(Counter(all_actions)),
    }
