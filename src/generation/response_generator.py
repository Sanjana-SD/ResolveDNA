"""Response Generator.

Uses an LLM to convert the structured decision (intent + next-best action +
historical evidence) into a natural-language customer reply.

The LLM is constrained:
  - Use only provided historical evidence
  - Do not invent policies, timelines, refunds, or compensation
  - Do not claim actions the brand cannot perform
  - Remain concise and customer-support appropriate
  - Match the brand's observed support style

When no LLM API key is available, falls back to template-based generation.
"""
import os
import json
from typing import List, Dict, Any, Tuple, Optional

from src.resolution.store import ResolutionCase

# Template-based fallback responses when no LLM API is available
TEMPLATE_RESPONSES = {
    "provide_link": "Hi there! We'd like to help with this. You can find relevant information and self-service options here. Could you try the steps on our help page and let us know if the issue persists?",
    "troubleshoot": "Sorry to hear about the trouble! Let's try some quick troubleshooting: please restart the device, clear the app cache, and check for any pending updates. Let us know how it goes!",
    "clarify_info": "We'd love to look into this for you! Could you share a few more details — such as your order number or the device you're using — so we can investigate further?",
    "explain_policy": "Thanks for reaching out! Regarding this, our standard process typically takes a few business days. You can check the latest status through your account. Let us know if you have any other questions!",
    "request_dm": "We understand your concern and want to help! For account-specific assistance, could you please send us a DM with your details? We'll take it from there.",
    "apologize": "We're really sorry to hear about this experience. That's definitely not what we want for our customers. Let us know more details so we can try to make this right.",
    "general_response": "Thanks for reaching out! We're here to help. Could you share a few more details about the issue you're experiencing so we can assist you better?",
}


def generate_response(
    customer_message: str,
    intent: str,
    next_best_action: Dict[str, Any],
    retrieved_cases: List[Tuple[ResolutionCase, float]],
    decision: str,
    llm_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a customer-facing reply.

    When an LLM API key is available, uses the LLM with a constrained prompt.
    Otherwise, uses template-based generation.
    """
    action = next_best_action.get("action", "general_response")

    if decision == "ESCALATE":
        draft_reply = (
            "Thank you for reaching out to us. We understand your concern and want "
            "to make sure you get the right help. This looks like it needs our "
            "specialized team to investigate properly. Please DM us your details "
            "and we'll connect you with the right support."
        )
        generation_method = "escalation_template"
    elif llm_api_key:
        draft_reply, generation_method = _generate_with_llm(
            customer_message, intent, action, retrieved_cases, llm_api_key
        )
    else:
        draft_reply = TEMPLATE_RESPONSES.get(action, TEMPLATE_RESPONSES["general_response"])
        generation_method = "template_fallback"

    # Build historical evidence summary for transparency
    evidence_summary = []
    for case, score in retrieved_cases[:3]:
        evidence_summary.append({
            "case_id": case.case_id,
            "problem": case.customer_problem[:120],
            "action": case.brand_actions[0] if case.brand_actions else "unknown",
            "outcome": case.final_state,
            "resolved": case.observed_resolution_signal,
            "similarity": score,
        })

    return {
        "draft_reply": draft_reply,
        "generation_method": generation_method,
        "action_used": action,
        "evidence_summary": evidence_summary,
    }


def _generate_with_llm(
    customer_message: str,
    intent: str,
    action: str,
    retrieved_cases: List[Tuple[ResolutionCase, float]],
    api_key: str,
) -> Tuple[str, str]:
    """Generate reply using LLM API with evidence-constrained prompt."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        evidence_text = ""
        for i, (case, score) in enumerate(retrieved_cases[:3], 1):
            outcome = "resolved" if case.observed_resolution_signal else case.final_state
            evidence_text += (
                f"\nCase {i} (similarity={score:.3f}):\n"
                f"  Customer: {case.customer_problem[:150]}\n"
                f"  Brand action: {case.brand_actions[0] if case.brand_actions else 'unknown'}\n"
                f"  Outcome: {outcome}\n"
            )

        prompt = f"""You are a customer support agent for Amazon (AmazonHelp on Twitter).
Generate a brief, helpful reply to the customer message below.

CONSTRAINTS:
- Use ONLY information from the historical evidence provided
- Do NOT invent policies, refund timelines, compensation, or account-specific actions
- Do NOT make promises you cannot keep
- Keep the response under 280 characters (Twitter limit)
- Match AmazonHelp's friendly, professional tone
- The recommended action is: {action}

CUSTOMER MESSAGE:
{customer_message}

DETECTED INTENT: {intent}

HISTORICAL EVIDENCE:
{evidence_text}

Generate ONLY the reply text, nothing else."""

        model = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.0-flash"))
        response = model.generate_content(prompt)
        return response.text.strip(), "llm_constrained"

    except Exception as e:
        # Fallback to template on any LLM failure
        fallback = TEMPLATE_RESPONSES.get(action, TEMPLATE_RESPONSES["general_response"])
        return fallback, f"template_fallback_llm_error:{str(e)[:50]}"
