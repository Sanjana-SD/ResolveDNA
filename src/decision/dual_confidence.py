"""Dual Confidence Engine.

Separates two fundamentally different confidence dimensions:
  1. Intent Confidence  – how sure are we about what the customer wants?
  2. Resolution Confidence – how sure are we that we can safely auto-respond?

These are NOT the same. A customer can clearly state "I want a refund"
(intent_confidence=0.96) but historical evidence may not support a safe
automated response (resolution_confidence=0.45).
"""
from typing import List, Dict, Any, Tuple

from src.resolution.store import ResolutionCase
from src.decision.consistency import compute_resolution_consistency, compute_outcome_consistency


def compute_resolution_confidence(
    intent_confidence: float,
    retrieved_cases: List[Tuple[ResolutionCase, float]],
    consistency: Dict[str, Any],
    outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute resolution confidence from multiple evidence signals.

    Signals considered:
        - intent_confidence: from the intent classifier
        - similarity_strength: average similarity of top retrieved cases
        - case_volume: how many relevant historical cases were found
        - action_consistency: do historical cases agree on the action?
        - outcome_consistency: do historical cases agree on the outcome?
        - resolution_rate: fraction of retrieved cases with positive resolution
        - requires_account_lookup: some intents inherently need escalation
    """
    if not retrieved_cases:
        return {
            "resolution_confidence": 0.0,
            "components": {
                "similarity_strength": 0.0,
                "case_volume_signal": 0.0,
                "action_consistency": 0.0,
                "outcome_strength": 0.0,
            },
            "explanation": "No historical cases retrieved."
        }

    # --- Component signals ---
    similarities = [score for _, score in retrieved_cases]
    avg_similarity = sum(similarities) / len(similarities)
    max_similarity = max(similarities)
    n_cases = len(retrieved_cases)

    # Similarity strength: higher is better, capped at 1.0
    similarity_strength = min(1.0, avg_similarity * 2.0)

    # Case volume signal: more cases = more confidence, diminishing returns
    case_volume_signal = min(1.0, n_cases / 5.0)

    # Action consistency from consistency module
    action_consistency = consistency.get("consistency_score", 0.0)

    # Outcome strength: prefer when most cases were resolved
    outcome_strength = outcome.get("resolution_rate", 0.0)

    # --- Weighted combination ---
    # Weights chosen to reflect: even if intent is clear, resolution confidence
    # should be LOW when evidence is thin, conflicting, or historically unsuccessful
    weights = {
        "similarity_strength": 0.30,
        "case_volume_signal": 0.15,
        "action_consistency": 0.30,
        "outcome_strength": 0.25,
    }

    raw_score = (
        weights["similarity_strength"] * similarity_strength
        + weights["case_volume_signal"] * case_volume_signal
        + weights["action_consistency"] * action_consistency
        + weights["outcome_strength"] * outcome_strength
    )

    # Conflicting evidence penalty
    if consistency.get("conflicting", False):
        raw_score *= 0.70

    resolution_confidence = round(min(1.0, max(0.0, raw_score)), 3)

    components = {
        "similarity_strength": round(similarity_strength, 3),
        "case_volume_signal": round(case_volume_signal, 3),
        "action_consistency": round(action_consistency, 3),
        "outcome_strength": round(outcome_strength, 3),
        "avg_similarity": round(avg_similarity, 4),
        "max_similarity": round(max_similarity, 4),
        "n_cases": n_cases,
    }

    # Build human-readable explanation
    explanations = []
    if similarity_strength < 0.3:
        explanations.append("retrieved cases have low similarity to query")
    if case_volume_signal < 0.4:
        explanations.append("insufficient historical case volume")
    if action_consistency < 0.4:
        explanations.append("historical brand actions are inconsistent/conflicting")
    if outcome_strength < 0.3:
        explanations.append("low historical resolution success rate")
    if not explanations:
        explanations.append("strong historical evidence supports automated response")

    return {
        "resolution_confidence": resolution_confidence,
        "components": components,
        "explanation": "; ".join(explanations),
    }
