"""Human vs LLM Judge Agreement Analysis.

Computes agreement metrics between human evaluations and LLM judge scores:
  - Percentage agreement
  - Cohen's Kappa (per dimension)
  - Disagreement examples

Requires a human_evaluations.json file with the same rubric dimensions.
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


def compute_agreement(
    human_scores: List[Dict[str, int]],
    llm_scores: List[Dict[str, int]],
    dimensions: List[str] = None,
) -> Dict[str, Any]:
    """Compute agreement between human and LLM judge evaluations."""
    if not dimensions:
        dimensions = ["groundedness", "relevance", "helpfulness",
                       "action_correctness", "unsupported_claims", "tone"]

    results = {}
    overall_agreements = []

    for dim in dimensions:
        human_vals = [h.get(dim, 3) for h in human_scores]
        llm_vals = [l.get(dim, 3) for l in llm_scores]

        # Exact agreement
        exact = sum(1 for h, l in zip(human_vals, llm_vals) if h == l)
        exact_pct = exact / max(1, len(human_vals))

        # Within-1 agreement (more lenient)
        within_1 = sum(1 for h, l in zip(human_vals, llm_vals) if abs(h - l) <= 1)
        within_1_pct = within_1 / max(1, len(human_vals))

        # Cohen's Kappa (linearized)
        kappa = _cohens_kappa(human_vals, llm_vals)

        # Mean absolute difference
        mad = np.mean([abs(h - l) for h, l in zip(human_vals, llm_vals)])

        results[dim] = {
            "exact_agreement": round(exact_pct, 4),
            "within_1_agreement": round(within_1_pct, 4),
            "cohens_kappa": round(kappa, 4),
            "mean_absolute_diff": round(float(mad), 4),
        }
        overall_agreements.append(within_1_pct)

    results["overall"] = {
        "mean_within_1_agreement": round(float(np.mean(overall_agreements)), 4),
        "n_examples": len(human_scores),
    }
    return results


def _cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """Compute Cohen's Kappa for ordinal ratings."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    # Treat as categorical for kappa
    categories = sorted(set(rater1 + rater2))
    n = len(rater1)

    if len(categories) <= 1:
        return 1.0

    # Observed agreement
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement
    pe = 0
    for c in categories:
        p1 = sum(1 for r in rater1 if r == c) / n
        p2 = sum(1 for r in rater2 if r == c) / n
        pe += p1 * p2

    if pe >= 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def find_disagreements(
    human_scores: List[Dict[str, int]],
    llm_scores: List[Dict[str, int]],
    example_ids: List[str],
    threshold: int = 2,
) -> List[Dict[str, Any]]:
    """Find examples where human and LLM judge strongly disagree."""
    dimensions = ["groundedness", "relevance", "helpfulness",
                   "action_correctness", "unsupported_claims", "tone"]

    disagreements = []
    for i in range(len(human_scores)):
        for dim in dimensions:
            h = human_scores[i].get(dim, 3)
            l = llm_scores[i].get(dim, 3)
            if abs(h - l) >= threshold:
                disagreements.append({
                    "example_id": example_ids[i] if i < len(example_ids) else f"example_{i}",
                    "dimension": dim,
                    "human_score": h,
                    "llm_score": l,
                    "difference": abs(h - l),
                })

    return sorted(disagreements, key=lambda x: x["difference"], reverse=True)


def generate_human_evaluation_template(
    golden_set_path: str,
    output_path: str,
    n_samples: int = 50,
) -> List[Dict[str, Any]]:
    """Generate a template JSON for human evaluation annotation.

    The human evaluator fills in scores 1-5 for each dimension.
    """
    import random
    random.seed(42)

    with open(golden_set_path, 'r', encoding='utf-8') as f:
        golden = json.load(f)

    examples = golden.get("examples", golden if isinstance(golden, list) else [])
    samples = random.sample(examples, min(n_samples, len(examples)))

    template = []
    for ex in samples:
        template.append({
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "expected_intent": ex["expected_intent"],
            "expected_decision": ex["expected_decision"],
            "difficulty": ex["difficulty"],
            # Human fills these in:
            "human_scores": {
                "groundedness": 0,
                "relevance": 0,
                "helpfulness": 0,
                "action_correctness": 0,
                "unsupported_claims": 0,
                "tone": 0,
            },
            "human_notes": ""
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
    print(f"[Human Agreement] Saved template for {len(template)} examples to {output_path}")
    return template
