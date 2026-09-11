"""Evaluation Metrics for ResolveDNA.

Computes:
  - Intent: accuracy, macro precision/recall/F1, per-intent F1
  - Decision: auto-handle precision, escalation precision, false-auto-handle rate
  - Calibration: confidence vs correctness
  - Trustworthy Automation Rate (business metric)
"""
import numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute intent classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    labels = sorted(set(y_true + y_pred))
    per_intent_f1 = {}
    for label in labels:
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]
        per_intent_f1[label] = round(f1_score(y_true_binary, y_pred_binary, zero_division=0), 4)

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_intent_f1": per_intent_f1,
    }


def compute_decision_metrics(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, Any]:
    """Compute AUTO_HANDLE vs ESCALATE decision metrics.

    Critical metrics:
      - auto_handle_precision: of cases we auto-handled, what fraction was correct?
      - false_auto_handle_rate: how often did we auto-handle when we should have escalated?
      - escalation_precision: of cases we escalated, what fraction was correct?
      - false_escalation_rate: how often did we escalate when auto-handle was safe?
    """
    auto_correct = sum(1 for t, p in zip(y_true, y_pred) if p == "AUTO_HANDLE" and t == "AUTO_HANDLE")
    auto_total = sum(1 for p in y_pred if p == "AUTO_HANDLE")
    esc_correct = sum(1 for t, p in zip(y_true, y_pred) if p == "ESCALATE" and t == "ESCALATE")
    esc_total = sum(1 for p in y_pred if p == "ESCALATE")

    false_auto = sum(1 for t, p in zip(y_true, y_pred) if p == "AUTO_HANDLE" and t == "ESCALATE")
    false_esc = sum(1 for t, p in zip(y_true, y_pred) if p == "ESCALATE" and t == "AUTO_HANDLE")

    total = len(y_true)

    return {
        "decision_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "auto_handle_precision": round(auto_correct / max(1, auto_total), 4),
        "auto_handle_count": auto_total,
        "escalation_precision": round(esc_correct / max(1, esc_total), 4),
        "escalation_count": esc_total,
        "false_auto_handle_rate": round(false_auto / max(1, total), 4),
        "false_auto_handle_count": false_auto,
        "false_escalation_rate": round(false_esc / max(1, total), 4),
        "false_escalation_count": false_esc,
    }


def compute_trustworthy_automation_rate(
    intent_correct: List[bool],
    decision_correct: List[bool],
    verification_passed: List[bool],
    predicted_decisions: List[str],
) -> Dict[str, Any]:
    """Compute the Trustworthy Automation Rate.

    Definition: A case is trustworthy-automated if:
      1. Intent is correct
      2. Decision is correct
      3. Response verification passed
      4. The system decided AUTO_HANDLE

    TAR = (trustworthy auto-handled cases) / (total incoming cases)

    Also report:
      Auto-handled Case Precision = trustworthy / total auto-handled
    """
    total = len(intent_correct)
    trustworthy = 0
    total_auto = 0

    for i in range(total):
        if predicted_decisions[i] == "AUTO_HANDLE":
            total_auto += 1
            if intent_correct[i] and decision_correct[i] and verification_passed[i]:
                trustworthy += 1

    tar = trustworthy / max(1, total)
    auto_precision = trustworthy / max(1, total_auto)

    return {
        "trustworthy_automation_rate": round(tar, 4),
        "trustworthy_count": trustworthy,
        "total_auto_handled": total_auto,
        "auto_handled_case_precision": round(auto_precision, 4),
        "total_cases": total,
    }


def compute_calibration(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Compute confidence calibration: is confidence predictive of correctness?"""
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_data = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        n = sum(mask)
        if n == 0:
            continue
        avg_conf = np.mean([c for c, m in zip(confidences, mask) if m])
        avg_correct = np.mean([int(c) for c, m in zip(correct, mask) if m])
        calibration_data.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "count": n,
            "avg_confidence": round(float(avg_conf), 4),
            "avg_correctness": round(float(avg_correct), 4),
            "gap": round(abs(float(avg_conf) - float(avg_correct)), 4),
        })

    ece = np.mean([d["gap"] * d["count"] for d in calibration_data]) / max(1, len(confidences))

    return {
        "expected_calibration_error": round(float(ece), 4),
        "bins": calibration_data,
    }


def compute_difficulty_breakdown(
    difficulties: List[str],
    intent_correct: List[bool],
    decision_correct: List[bool],
) -> Dict[str, Any]:
    """Compute metrics broken down by difficulty level."""
    breakdown = {}
    for diff in set(difficulties):
        mask = [d == diff for d in difficulties]
        n = sum(mask)
        if n == 0:
            continue
        ic = [c for c, m in zip(intent_correct, mask) if m]
        dc = [c for c, m in zip(decision_correct, mask) if m]
        breakdown[diff] = {
            "count": n,
            "intent_accuracy": round(sum(ic) / n, 4),
            "decision_accuracy": round(sum(dc) / n, 4),
        }
    return breakdown
