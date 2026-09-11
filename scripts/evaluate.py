"""Main Evaluation Script.

Runs the full evaluation pipeline:
1. Load golden set
2. Run ResolveDNA pipeline on all examples
3. Run all three baselines on all examples
4. Compute intent, decision, calibration, and TAR metrics
5. Run LLM-as-judge on generated responses
6. Compute difficulty-aware breakdown
7. Save all results to results/

Usage:
    python scripts/evaluate.py
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

from src.pipeline import ResolveDNAPipeline
from src.evaluation.baselines import train_baselines
from src.evaluation.metrics import (
    compute_intent_metrics, compute_decision_metrics,
    compute_trustworthy_automation_rate, compute_calibration,
    compute_difficulty_breakdown
)
from src.evaluation.llm_judge import llm_judge_evaluate

GOLDEN_SET_PATH = "data/golden/golden_set.json"
TRAJECTORIES_PATH = "data/processed/amazon_trajectories.jsonl"
MEMORY_PATH = "data/processed/resolution_memory_sample.pkl"
RESULTS_DIR = "results"


def load_golden_set() -> list:
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("examples", data)


def evaluate_system(system_name: str, predict_fn, examples: list) -> dict:
    """Run a system on all examples and collect predictions."""
    predictions = []
    for ex in examples:
        try:
            result = predict_fn(ex["customer_message"])
            predictions.append({
                "id": ex["id"],
                "expected_intent": ex["expected_intent"],
                "predicted_intent": result.get("intent", "other_unknown"),
                "expected_decision": ex["expected_decision"],
                "predicted_decision": result.get("decision", "ESCALATE"),
                "intent_confidence": result.get("intent_confidence", 0.0),
                "resolution_confidence": result.get("resolution_confidence", 0.0),
                "draft_reply": result.get("draft_reply", ""),
                "next_action": result.get("next_best_action", {}).get("action", "general_response"),
                "verification_status": result.get("verification", {}).get("status", "PASS"),
                "difficulty": ex.get("difficulty", "Medium"),
            })
        except Exception as e:
            predictions.append({
                "id": ex["id"],
                "expected_intent": ex["expected_intent"],
                "predicted_intent": "other_unknown",
                "expected_decision": ex["expected_decision"],
                "predicted_decision": "ESCALATE",
                "intent_confidence": 0.0,
                "resolution_confidence": 0.0,
                "draft_reply": "",
                "next_action": "general_response",
                "verification_status": "ERROR",
                "difficulty": ex.get("difficulty", "Medium"),
                "error": str(e),
            })

    # Compute metrics
    y_true_intent = [p["expected_intent"] for p in predictions]
    y_pred_intent = [p["predicted_intent"] for p in predictions]
    y_true_decision = [p["expected_decision"] for p in predictions]
    y_pred_decision = [p["predicted_decision"] for p in predictions]

    intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent)
    decision_metrics = compute_decision_metrics(y_true_decision, y_pred_decision)

    intent_correct = [t == p for t, p in zip(y_true_intent, y_pred_intent)]
    decision_correct = [t == p for t, p in zip(y_true_decision, y_pred_decision)]
    verification_passed = [p["verification_status"] in ("PASS", "PASS_WITH_WARNINGS") for p in predictions]

    tar = compute_trustworthy_automation_rate(
        intent_correct, decision_correct, verification_passed, y_pred_decision
    )

    confidences = [p["intent_confidence"] for p in predictions]
    calibration = compute_calibration(confidences, intent_correct)

    difficulties = [p["difficulty"] for p in predictions]
    difficulty_breakdown = compute_difficulty_breakdown(difficulties, intent_correct, decision_correct)

    return {
        "system": system_name,
        "intent_metrics": intent_metrics,
        "decision_metrics": decision_metrics,
        "trustworthy_automation_rate": tar,
        "calibration": calibration,
        "difficulty_breakdown": difficulty_breakdown,
        "predictions": predictions,
    }


def run_llm_judge_evaluation(predictions: list) -> list:
    """Run LLM-as-judge on all predictions."""
    judge_results = []
    for pred in predictions:
        score = llm_judge_evaluate(
            customer_message=pred.get("id", ""),
            generated_reply=pred.get("draft_reply", ""),
            intent=pred.get("predicted_intent", ""),
            decision=pred.get("predicted_decision", ""),
            next_action=pred.get("next_action", ""),
            evidence_summary="",
        )
        judge_results.append({
            "id": pred["id"],
            "scores": score,
        })
    return judge_results


def main():
    print("=" * 60)
    print("RESOLVE DNA — FULL EVALUATION PIPELINE")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load golden set
    print("\n[1/5] Loading golden evaluation set...")
    examples = load_golden_set()
    print(f"Loaded {len(examples)} evaluation examples")

    # Initialize ResolveDNA pipeline
    print("\n[2/5] Initializing ResolveDNA pipeline...")
    start = time.time()
    pipeline = ResolveDNAPipeline(memory_path=MEMORY_PATH)
    print(f"Pipeline loaded in {time.time() - start:.1f}s")

    # Train baselines
    print("\n[3/5] Training baselines...")
    b1, b2, b3 = train_baselines(TRAJECTORIES_PATH)

    # Run evaluation
    print("\n[4/5] Running evaluation on golden set...")
    systems = {
        "ResolveDNA": pipeline.process,
        "Baseline_MajorityClass": b1.predict,
        "Baseline_TfidfLogreg": b2.predict,
        "Baseline_GenericRAG": b3.predict,
    }

    all_results = {}
    for name, predict_fn in systems.items():
        print(f"\n  Evaluating: {name}...")
        start = time.time()
        result = evaluate_system(name, predict_fn, examples)
        elapsed = time.time() - start
        print(f"    Intent Accuracy:  {result['intent_metrics']['accuracy']:.4f}")
        print(f"    Macro F1:         {result['intent_metrics']['macro_f1']:.4f}")
        print(f"    Decision Acc:     {result['decision_metrics']['decision_accuracy']:.4f}")
        print(f"    Auto-Handle Prec: {result['decision_metrics']['auto_handle_precision']:.4f}")
        print(f"    False Auto Rate:  {result['decision_metrics']['false_auto_handle_rate']:.4f}")
        print(f"    TAR:              {result['trustworthy_automation_rate']['trustworthy_automation_rate']:.4f}")
        print(f"    Time: {elapsed:.1f}s")
        all_results[name] = result

    # Run LLM judge on ResolveDNA predictions
    print("\n[5/5] Running LLM-as-judge on ResolveDNA responses...")
    rdna_preds = all_results["ResolveDNA"]["predictions"]
    judge_results = run_llm_judge_evaluation(rdna_preds)

    # Save results
    print("\nSaving results...")

    # Main metrics comparison
    metrics_summary = {}
    for name, result in all_results.items():
        metrics_summary[name] = {
            "intent_metrics": result["intent_metrics"],
            "decision_metrics": result["decision_metrics"],
            "trustworthy_automation_rate": result["trustworthy_automation_rate"],
            "calibration": result["calibration"],
            "difficulty_breakdown": result["difficulty_breakdown"],
        }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)

    # Baseline comparison
    baseline_comparison = {
        name: {
            "intent_accuracy": r["intent_metrics"]["accuracy"],
            "macro_f1": r["intent_metrics"]["macro_f1"],
            "decision_accuracy": r["decision_metrics"]["decision_accuracy"],
            "auto_handle_precision": r["decision_metrics"]["auto_handle_precision"],
            "false_auto_handle_rate": r["decision_metrics"]["false_auto_handle_rate"],
            "trustworthy_automation_rate": r["trustworthy_automation_rate"]["trustworthy_automation_rate"],
        }
        for name, r in all_results.items()
    }
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), 'w') as f:
        json.dump(baseline_comparison, f, indent=2, cls=NumpyEncoder)

    # LLM judge results
    with open(os.path.join(RESULTS_DIR, "llm_judge_results.json"), 'w') as f:
        json.dump(judge_results, f, indent=2, cls=NumpyEncoder)

    # Predictions
    for name, result in all_results.items():
        fname = f"predictions_{name.lower().replace(' ', '_')}.json"
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            json.dump(result["predictions"], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # Print headline comparison table
    print(f"\n{'System':<25} {'Intent Acc':>10} {'Macro F1':>10} {'Dec Acc':>10} {'AH Prec':>10} {'FAH Rate':>10} {'TAR':>10}")
    print("-" * 85)
    for name, r in baseline_comparison.items():
        print(f"{name:<25} {r['intent_accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['decision_accuracy']:>10.4f} {r['auto_handle_precision']:>10.4f} {r['false_auto_handle_rate']:>10.4f} {r['trustworthy_automation_rate']:>10.4f}")


if __name__ == "__main__":
    main()
