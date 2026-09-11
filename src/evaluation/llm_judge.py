"""LLM-as-Judge for Response Quality Evaluation.

Evaluates generated responses on a structured rubric:
  - groundedness (1-5): Is the response supported by evidence?
  - relevance (1-5): Does it address the customer's problem?
  - helpfulness (1-5): Would this actually help the customer?
  - action_correctness (1-5): Is the recommended action appropriate?
  - unsupported_claims (1-5): 1=many unsupported claims, 5=none
  - tone (1-5): Professional, empathetic, brand-appropriate?

Returns strict JSON.

When no LLM API key is available, uses a rule-based heuristic judge.
"""
import os
import json
import re
from typing import Dict, Any, Optional


def llm_judge_evaluate(
    customer_message: str,
    generated_reply: str,
    intent: str,
    decision: str,
    next_action: str,
    evidence_summary: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a generated response using LLM-as-judge or heuristic fallback."""
    api_key = api_key or os.environ.get("LLM_API_KEY")

    if api_key:
        return _llm_judge(customer_message, generated_reply, intent, decision, next_action, evidence_summary, api_key)
    else:
        return _heuristic_judge(customer_message, generated_reply, intent, decision, next_action)


def _llm_judge(
    customer_message: str,
    generated_reply: str,
    intent: str,
    decision: str,
    next_action: str,
    evidence_summary: str,
    api_key: str,
) -> Dict[str, Any]:
    """Use LLM to evaluate response quality."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        prompt = f"""You are evaluating a customer support response. Score each dimension 1-5.

CUSTOMER MESSAGE: {customer_message}
DETECTED INTENT: {intent}
SYSTEM DECISION: {decision}
RECOMMENDED ACTION: {next_action}
HISTORICAL EVIDENCE: {evidence_summary}

GENERATED REPLY: {generated_reply}

SCORING RUBRIC:
- groundedness (1-5): Is the response supported by the historical evidence? 5=fully grounded, 1=fabricated
- relevance (1-5): Does it address the customer's actual problem? 5=directly relevant, 1=off-topic
- helpfulness (1-5): Would this help the customer? 5=very helpful, 1=unhelpful
- action_correctness (1-5): Is the action appropriate for this situation? 5=perfect action, 1=wrong action
- unsupported_claims (1-5): 5=no unsupported claims, 1=many false/unsupported claims
- tone (1-5): Professional, empathetic, brand-appropriate? 5=excellent tone, 1=inappropriate

Return ONLY valid JSON with these exact keys and integer values 1-5:
{{"groundedness": N, "relevance": N, "helpfulness": N, "action_correctness": N, "unsupported_claims": N, "tone": N}}"""

        model = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.0-flash"))
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            scores = json.loads(json_match.group())
            scores["judge_method"] = "llm"
            scores["overall"] = round(sum(scores[k] for k in ["groundedness", "relevance", "helpfulness", "action_correctness", "unsupported_claims", "tone"]) / 6, 2)
            return scores

    except Exception as e:
        pass

    return _heuristic_judge(customer_message, generated_reply, intent, decision, next_action)


def _heuristic_judge(
    customer_message: str,
    generated_reply: str,
    intent: str,
    decision: str,
    next_action: str,
) -> Dict[str, Any]:
    """Rule-based heuristic judge when no LLM is available."""
    reply_lower = generated_reply.lower()

    # Groundedness: penalize specific claims
    groundedness = 4
    if any(w in reply_lower for w in ["guarantee", "refund has been", "credit your", "compensation"]):
        groundedness = 2

    # Relevance: does reply seem related to the intent?
    relevance = 3
    if len(generated_reply) > 30:
        relevance = 4

    # Helpfulness: does it offer actionable guidance?
    helpfulness = 3
    if any(w in reply_lower for w in ["try", "check", "visit", "help", "assist", "share", "details"]):
        helpfulness = 4

    # Action correctness: assume template responses are correct
    action_correctness = 4

    # Unsupported claims
    unsupported = 5
    claim_patterns = [r'refund.*processed', r'guarantee', r'within \d+ (hours|days)', r'account.*restored']
    for p in claim_patterns:
        if re.search(p, reply_lower):
            unsupported = 2
            break

    # Tone
    tone = 4
    if any(w in reply_lower for w in ["sorry", "understand", "help", "happy to"]):
        tone = 5

    scores = {
        "groundedness": groundedness,
        "relevance": relevance,
        "helpfulness": helpfulness,
        "action_correctness": action_correctness,
        "unsupported_claims": unsupported,
        "tone": tone,
        "judge_method": "heuristic",
    }
    scores["overall"] = round(sum(scores[k] for k in ["groundedness", "relevance", "helpfulness", "action_correctness", "unsupported_claims", "tone"]) / 6, 2)
    return scores
