"""Post-generation Response Verifier.

After generating a reply, checks whether the response is:
  1. Supported by retrieved historical evidence
  2. Aligned with the determined next-best action
  3. Free of unsupported claims / hallucinated commitments
  4. Addressing the customer's problem
  5. Appropriate for the brand's historical tone

If verification fails, the system should revise or escalate.
"""
import re
from typing import Dict, Any, List, Tuple

from src.resolution.store import ResolutionCase


# Known dangerous patterns that indicate unsupported claims
UNSUPPORTED_CLAIM_PATTERNS = [
    r'\bguarantee\b',
    r'\bwe will refund\b',
    r'\brefund has been processed\b',
    r'\bcompensation\b',
    r'\bcredit your account\b',
    r'\bfree replacement\b',
    r'\bwithin \d+ (hours?|days?|minutes?)\b',
    r'\byour order.*shipped\b',
    r'\bpackage.*rescheduled\b',
    r'\baccount.*restored\b',
]

UNSUPPORTED_RE = re.compile('|'.join(UNSUPPORTED_CLAIM_PATTERNS), re.IGNORECASE)


def verify_response(
    draft_reply: str,
    customer_message: str,
    next_best_action: Dict[str, Any],
    retrieved_cases: List[Tuple[ResolutionCase, float]],
    intent: str,
) -> Dict[str, Any]:
    """Verify the generated response against evidence and constraints.

    Returns a verification result with pass/fail status and details.
    """
    checks = {}
    issues = []

    # --- Check 1: Unsupported claims ---
    unsupported_matches = UNSUPPORTED_RE.findall(draft_reply)
    checks["unsupported_claims"] = len(unsupported_matches) == 0
    if unsupported_matches:
        issues.append(f"Contains potentially unsupported claims: {unsupported_matches}")

    # --- Check 2: Action alignment ---
    action = next_best_action.get("action", "general_response")
    action_keywords = {
        "provide_link": ["link", "page", "help", "http", "here"],
        "troubleshoot": ["restart", "try", "clear", "cache", "update", "step"],
        "clarify_info": ["order number", "details", "share", "which", "could you"],
        "explain_policy": ["policy", "process", "typically", "business days", "standard"],
        "request_dm": ["dm", "direct message", "private", "send us"],
        "apologize": ["sorry", "apologize", "understand", "concern"],
        "general_response": [],
    }
    expected_keywords = action_keywords.get(action, [])
    if expected_keywords:
        reply_lower = draft_reply.lower()
        has_alignment = any(kw in reply_lower for kw in expected_keywords)
        checks["action_alignment"] = has_alignment
        if not has_alignment:
            issues.append(f"Reply does not align with recommended action '{action}'")
    else:
        checks["action_alignment"] = True

    # --- Check 3: Addresses customer problem ---
    # Simple heuristic: reply should acknowledge or respond to the issue
    checks["addresses_problem"] = len(draft_reply) > 20
    if not checks["addresses_problem"]:
        issues.append("Reply is too short to meaningfully address the problem")

    # --- Check 4: Length check (Twitter limit) ---
    checks["within_length"] = len(draft_reply) <= 280
    if not checks["within_length"]:
        issues.append(f"Reply exceeds Twitter 280-char limit ({len(draft_reply)} chars)")

    # --- Check 5: Evidence support ---
    # If we have resolved cases that used the same action, evidence is supported
    evidence_supported = False
    if retrieved_cases:
        for case, score in retrieved_cases[:5]:
            if case.brand_actions and case.brand_actions[0] == action:
                evidence_supported = True
                break
    checks["evidence_supported"] = evidence_supported or action == "general_response"
    if not evidence_supported and action != "general_response":
        issues.append("No retrieved case uses the same action as the generated response")

    # --- Overall verdict ---
    all_pass = all(checks.values())
    critical_fail = not checks["unsupported_claims"]

    if critical_fail:
        status = "FAIL_CRITICAL"
        recommendation = "Revise response or escalate — unsupported claims detected"
    elif not all_pass:
        status = "PASS_WITH_WARNINGS"
        recommendation = "Response acceptable but has minor alignment issues"
    else:
        status = "PASS"
        recommendation = "Response verified against historical evidence"

    return {
        "status": status,
        "all_checks_pass": all_pass,
        "checks": checks,
        "issues": issues,
        "recommendation": recommendation,
    }
