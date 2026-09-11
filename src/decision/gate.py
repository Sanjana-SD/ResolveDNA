"""Evidence Sufficiency Gate.

Central decision module: determines AUTO_HANDLE vs ESCALATE.

The gate says: "Do we have enough reliable historical evidence to safely
respond automatically?"

This is a feature, not a failure. The system is designed to be conservative.
"""
from typing import List, Dict, Any, Tuple

from src.resolution.store import ResolutionCase
from src.intent.taxonomy import INTENT_TAXONOMY

# Thresholds tuned on validation data (NOT golden set)
# See decision_log.md Decision #6 for justification
DEFAULT_THRESHOLDS = {
    "min_intent_confidence": 0.50,
    "min_resolution_confidence": 0.35,
    "min_consistency_score": 0.30,
    "min_cases_retrieved": 2,
    "min_avg_similarity": 0.08,
    "max_false_autohandle_tolerance": 0.10,
}


def decide(
    intent: str,
    intent_confidence: float,
    resolution_confidence: float,
    consistency: Dict[str, Any],
    outcome: Dict[str, Any],
    retrieved_cases: List[Tuple[ResolutionCase, float]],
    thresholds: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Run the Evidence Sufficiency Gate.

    Returns:
        dict with keys: decision, reason, escalation_factors, confidence_summary
    """
    t = thresholds or DEFAULT_THRESHOLDS
    escalation_factors = []

    # --- Check intent confidence ---
    if intent_confidence < t["min_intent_confidence"]:
        escalation_factors.append(
            f"Intent confidence too low ({intent_confidence:.2f} < {t['min_intent_confidence']})"
        )

    # --- Check resolution confidence ---
    if resolution_confidence < t["min_resolution_confidence"]:
        escalation_factors.append(
            f"Resolution confidence too low ({resolution_confidence:.3f} < {t['min_resolution_confidence']})"
        )

    # --- Check consistency ---
    cons_score = consistency.get("consistency_score", 0.0)
    if cons_score < t["min_consistency_score"]:
        escalation_factors.append(
            f"Historical actions inconsistent (consistency={cons_score:.3f} < {t['min_consistency_score']})"
        )

    # --- Check conflicting evidence ---
    if consistency.get("conflicting", False):
        escalation_factors.append(
            "Retrieved cases show conflicting brand actions"
        )

    # --- Check case volume ---
    n_cases = len(retrieved_cases)
    if n_cases < t["min_cases_retrieved"]:
        escalation_factors.append(
            f"Insufficient historical cases ({n_cases} < {t['min_cases_retrieved']})"
        )

    # --- Check similarity quality ---
    if retrieved_cases:
        avg_sim = sum(s for _, s in retrieved_cases) / len(retrieved_cases)
        if avg_sim < t["min_avg_similarity"]:
            escalation_factors.append(
                f"Low similarity to historical cases (avg={avg_sim:.4f} < {t['min_avg_similarity']})"
            )

    # --- Check if intent inherently requires account lookup ---
    intent_info = INTENT_TAXONOMY.get(intent, {})
    if intent_info.get("requires_account_lookup", False):
        # Account-lookup intents can still be auto-handled if the brand
        # historically provided self-service links, but we lower the bar
        if resolution_confidence < 0.50:
            escalation_factors.append(
                f"Intent '{intent}' requires account-specific lookup with insufficient evidence"
            )

    # --- Final decision ---
    if escalation_factors:
        decision = "ESCALATE"
        reason = "; ".join(escalation_factors)
    else:
        decision = "AUTO_HANDLE"
        reason = "Strong historical evidence supports automated response"

    return {
        "decision": decision,
        "reason": reason,
        "escalation_factors": escalation_factors,
        "confidence_summary": {
            "intent_confidence": round(intent_confidence, 3),
            "resolution_confidence": round(resolution_confidence, 3),
            "consistency_score": round(cons_score, 3),
            "n_cases_retrieved": n_cases,
        }
    }
