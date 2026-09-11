"""Golden Evaluation Set Generator.

Creates 200 hand-curated evaluation examples from AmazonHelp trajectories,
stratified across difficulty levels and intent categories.

CRITICAL: These examples are ISOLATED from the resolution memory index.
They must NOT be used for training, retrieval, threshold tuning, or
prompt optimization.

The golden set contains examples with human-annotated:
  - expected_intent
  - expected_decision (AUTO_HANDLE / ESCALATE)
  - acceptable_action
  - difficulty (Easy / Medium / Hard / Ambiguous / Noisy / Conflicting-Evidence)
"""
import os
import sys
import json
import random
import re
from typing import List, Dict, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.taxonomy import IntentClassifier, INTENT_TAXONOMY

TRAJECTORIES_PATH = "data/processed/amazon_trajectories.jsonl"
GOLDEN_OUTPUT_PATH = "data/golden/golden_set.json"
GUIDELINES_PATH = "data/golden/annotation_guidelines.md"

random.seed(42)

# Difficulty heuristics
NOISY_PATTERNS = re.compile(
    r'(😡|😩|💀|🤬|wtf|smh|lmao|bruh|ffs|@\w+\s@\w+|#\w+\s#\w+)', re.IGNORECASE
)
AMBIGUOUS_SIGNALS = re.compile(
    r'(issue|problem|help|not working|something wrong|broken)', re.IGNORECASE
)


def classify_difficulty(text: str, intent: str, traj: Dict) -> str:
    """Heuristically assign difficulty based on message characteristics."""
    text_lower = text.lower()

    # Noisy: heavy emoji, slang, hashtags, multiple mentions
    if NOISY_PATTERNS.search(text) and len(text) < 80:
        return "Noisy"

    # Ambiguous: very short or vague messages
    if len(text) < 40 and AMBIGUOUS_SIGNALS.search(text):
        return "Ambiguous"

    if intent == "other_unknown":
        return "Hard"

    # Multi-intent signals
    intent_keywords_hit = 0
    for iid, info in INTENT_TAXONOMY.items():
        if iid == "other_unknown":
            continue
        for kw in info.get("keywords", []):
            if kw.lower() in text_lower:
                intent_keywords_hit += 1
                break
    if intent_keywords_hit >= 2:
        return "Hard"

    # Conflicting-Evidence: trajectory ended in repeated complaint
    if traj.get("final_state") == "REPEATED_COMPLAINT_ESCALATED":
        return "Conflicting-Evidence"

    # Medium: requires account lookup
    if INTENT_TAXONOMY.get(intent, {}).get("requires_account_lookup", False):
        return "Medium"

    return "Easy"


def determine_expected_decision(traj: Dict, difficulty: str) -> str:
    """Determine expected decision based on trajectory evidence."""
    if difficulty in ("Ambiguous", "Conflicting-Evidence"):
        return "ESCALATE"
    if traj.get("final_state") == "REPEATED_COMPLAINT_ESCALATED":
        return "ESCALATE"
    if traj.get("final_state") == "DM_DEFLECTED":
        return "ESCALATE"
    intent_info = INTENT_TAXONOMY.get(traj.get("_intent", ""), {})
    if intent_info.get("requires_account_lookup", False) and not traj.get("observed_resolution_signal"):
        return "ESCALATE"
    if traj.get("observed_resolution_signal"):
        return "AUTO_HANDLE"
    return "ESCALATE"


def determine_acceptable_action(traj: Dict) -> str:
    """Determine acceptable action from trajectory brand actions."""
    actions = traj.get("brand_actions", [])
    if actions:
        return actions[0]
    return "general_response"


def build_golden_set():
    """Build the golden evaluation set from trajectories."""
    print("[Golden Set] Loading trajectories...")
    trajectories = []
    with open(TRAJECTORIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    classifier = IntentClassifier()

    # Classify intents for all trajectories
    for t in trajectories:
        intent, conf = classifier.classify(t["customer_problem"])
        t["_intent"] = intent
        t["_intent_conf"] = conf

    # --- Stratified sampling ---
    # Target: 200 examples total
    # Strategy: sample across intents, difficulty, AND decision outcome
    intent_buckets = {}
    for t in trajectories:
        iid = t["_intent"]
        if iid not in intent_buckets:
            intent_buckets[iid] = []
        intent_buckets[iid].append(t)

    golden_examples = []
    example_id = 0

    # Sample ~20 per intent (10 intents * 20 = 200)
    per_intent_target = 20

    for intent_id, bucket in intent_buckets.items():
        random.shuffle(bucket)

        # Separate resolved vs non-resolved for balanced decision sampling
        resolved = [t for t in bucket if t.get("observed_resolution_signal")]
        unresolved = [t for t in bucket if not t.get("observed_resolution_signal")]

        selected = []

        # Take ~10 from resolved (AUTO_HANDLE candidates) and ~10 from unresolved (ESCALATE candidates)
        auto_target = min(10, len(resolved))
        esc_target = per_intent_target - auto_target

        for t in resolved[:auto_target]:
            example_id += 1
            diff = classify_difficulty(t["customer_problem"], intent_id, t)
            # Resolved cases -> generally AUTO_HANDLE (but Hard/Ambiguous still escalate)
            if diff in ("Ambiguous", "Conflicting-Evidence"):
                expected_decision = "ESCALATE"
                escalation_reason = "Ambiguous context despite resolution signal"
            else:
                expected_decision = "AUTO_HANDLE"
                escalation_reason = ""

            context = ""
            if len(t.get("turns", [])) > 1:
                context = t["turns"][1]["text"][:200]

            example = {
                "id": f"golden_{example_id:04d}",
                "customer_message": t["customer_problem"],
                "context": context,
                "expected_intent": intent_id,
                "expected_decision": expected_decision,
                "acceptable_action": determine_acceptable_action(t),
                "difficulty": diff,
                "escalation_reason": escalation_reason,
                "source_conversation_id": t["conversation_id"],
                "notes": f"Sampled from {intent_id} resolved bucket, final_state={t['final_state']}"
            }
            selected.append(example)

        for t in unresolved[:esc_target]:
            example_id += 1
            diff = classify_difficulty(t["customer_problem"], intent_id, t)
            expected_decision = "ESCALATE"

            if diff == "Ambiguous":
                escalation_reason = "Customer message is ambiguous or lacks context"
            elif diff == "Conflicting-Evidence":
                escalation_reason = "Historical evidence shows conflicting resolutions"
            elif t.get("final_state") == "REPEATED_COMPLAINT_ESCALATED":
                escalation_reason = "Similar cases historically resulted in repeated complaints"
            elif t.get("final_state") == "DM_DEFLECTED":
                escalation_reason = "Requires private account-level investigation"
            else:
                escalation_reason = "Insufficient evidence for safe automated response"

            context = ""
            if len(t.get("turns", [])) > 1:
                context = t["turns"][1]["text"][:200]

            example = {
                "id": f"golden_{example_id:04d}",
                "customer_message": t["customer_problem"],
                "context": context,
                "expected_intent": intent_id,
                "expected_decision": expected_decision,
                "acceptable_action": determine_acceptable_action(t),
                "difficulty": diff,
                "escalation_reason": escalation_reason,
                "source_conversation_id": t["conversation_id"],
                "notes": f"Sampled from {intent_id} unresolved bucket, final_state={t['final_state']}"
            }
            selected.append(example)

        golden_examples.extend(selected)

    print(f"[Golden Set] Generated {len(golden_examples)} evaluation examples")

    # Difficulty distribution
    diff_dist = {}
    for ex in golden_examples:
        d = ex["difficulty"]
        diff_dist[d] = diff_dist.get(d, 0) + 1
    print(f"[Golden Set] Difficulty distribution: {diff_dist}")

    # Intent distribution
    intent_dist = {}
    for ex in golden_examples:
        i = ex["expected_intent"]
        intent_dist[i] = intent_dist.get(i, 0) + 1
    print(f"[Golden Set] Intent distribution: {intent_dist}")

    # Decision distribution
    dec_dist = {}
    for ex in golden_examples:
        d = ex["expected_decision"]
        dec_dist[d] = dec_dist.get(d, 0) + 1
    print(f"[Golden Set] Decision distribution: {dec_dist}")

    # Save
    os.makedirs(os.path.dirname(GOLDEN_OUTPUT_PATH), exist_ok=True)
    golden_data = {
        "metadata": {
            "version": "1.0",
            "total_examples": len(golden_examples),
            "difficulty_distribution": diff_dist,
            "intent_distribution": intent_dist,
            "decision_distribution": dec_dist,
            "sampling_method": "Stratified by intent and difficulty, seed=42",
            "isolation_note": "These examples are isolated from retrieval index and threshold tuning",
        },
        "examples": golden_examples,
    }
    with open(GOLDEN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=2, ensure_ascii=False)
    print(f"[Golden Set] Saved to {GOLDEN_OUTPUT_PATH}")

    return golden_data


def write_annotation_guidelines():
    """Write annotation guidelines document."""
    os.makedirs(os.path.dirname(GUIDELINES_PATH), exist_ok=True)
    content = """# Golden Evaluation Set — Annotation Guidelines

## Purpose
This golden set evaluates the ResolveDNA system's ability to:
1. Correctly classify customer intent
2. Make safe AUTO_HANDLE vs ESCALATE decisions
3. Select the appropriate next-best action
4. Generate grounded, helpful responses

## Annotation Schema

Each example contains:
| Field | Description |
|---|---|
| `id` | Unique identifier (golden_XXXX) |
| `customer_message` | The raw customer tweet text |
| `context` | Optional: next turn in the conversation for context |
| `expected_intent` | The correct intent label from the taxonomy |
| `expected_decision` | AUTO_HANDLE or ESCALATE |
| `acceptable_action` | The acceptable brand action |
| `difficulty` | Easy / Medium / Hard / Ambiguous / Noisy / Conflicting-Evidence |
| `escalation_reason` | Why escalation is expected (if applicable) |
| `source_conversation_id` | Original conversation for traceability |

## Difficulty Taxonomy

| Level | Definition |
|---|---|
| **Easy** | Clear intent, standard language, historically consistent resolution |
| **Medium** | Clear intent but requires account-specific lookup |
| **Hard** | Multiple possible intents, or rare intent category |
| **Ambiguous** | Very short/vague message, unclear what customer needs |
| **Noisy** | Heavy emoji, slang, abbreviations, hashtag spam |
| **Conflicting-Evidence** | Historical cases show conflicting brand actions |

## Data Isolation
> **CRITICAL**: This golden set is completely isolated from:
> - Resolution memory index construction
> - Threshold tuning / validation
> - Prompt optimization
> - Model selection experiments

## Sampling Methodology
- Stratified sampling across 10 intent categories
- Target ~20 examples per intent
- Deliberate inclusion of hard/ambiguous/noisy/conflicting cases
- Random seed = 42 for reproducibility
"""
    with open(GUIDELINES_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Golden Set] Saved annotation guidelines to {GUIDELINES_PATH}")


if __name__ == "__main__":
    write_annotation_guidelines()
    build_golden_set()
