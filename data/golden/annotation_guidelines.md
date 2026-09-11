# Golden Evaluation Set — Annotation Guidelines

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
