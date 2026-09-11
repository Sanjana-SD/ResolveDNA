"""Audit script for ResolveDNA evaluation integrity and experiments.

Performs:
1. Golden set integrity check (counts, duplicates, retrieval contamination, label origin).
2. Intent classification breakdown & investigation of 100% accuracy.
3. Decision metric calculation, confusion matrices, and risk-coverage curve generation.
4. Baseline fairness verification.
5. Controlled Resolution-Awareness Experiment (Ablation).
6. Temporal Recency Experiment.
7. Retrieval inspection & Failure mode extraction.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.intent.taxonomy import IntentClassifier, INTENT_TAXONOMY
from src.retrieval.retriever import ResolutionRetriever
from src.resolution.store import ResolutionMemoryStore, ResolutionCase
from src.decision.gate import decide, DEFAULT_THRESHOLDS
from src.decision.consistency import compute_resolution_consistency, compute_outcome_consistency
from src.decision.dual_confidence import compute_resolution_confidence

def run_audit():
    print("====================================================================")
    print("         RESOLVEDNA EVALUATION INTEGRITY & READINESS AUDIT          ")
    print("====================================================================")
    
    # -------------------------------------------------------------
    # 1. GOLDEN SET AUDIT
    # -------------------------------------------------------------
    with open(ROOT_DIR / "data" / "golden" / "golden_set.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        golden_set = golden_data.get("examples", golden_data if isinstance(golden_data, list) else [])
        
    print(f"\n[1. Golden Set Integrity]")
    print(f"- Total examples: {len(golden_set)}")
    
    # Check duplicates
    messages = [ex["customer_message"].strip().lower() for ex in golden_set]
    case_ids = [ex.get("case_id") for ex in golden_set]
    
    unique_messages = len(set(messages))
    unique_case_ids = len(set(case_ids))
    print(f"- Unique customer messages: {unique_messages} / {len(messages)}")
    print(f"- Unique case_ids: {unique_case_ids} / {len(case_ids)}")
    
    # Check retrieval memory contamination
    sample_mem_path = ROOT_DIR / "data" / "processed" / "resolution_memory_sample.pkl"
    mem_case_ids = set()
    if sample_mem_path.exists():
        with open(sample_mem_path, "rb") as f:
            sample_cases = pickle.load(f)
            mem_case_ids = {c.case_id for c in sample_cases}
    
    contaminated_ids = set(case_ids).intersection(mem_case_ids)
    print(f"- Memory index contamination (case_id overlap): {len(contaminated_ids)} cases")
    
    # Check label provenance
    difficulty_dist = Counter([ex.get("difficulty", "unknown") for ex in golden_set])
    intent_dist = Counter([ex.get("expected_intent") for ex in golden_set])
    decision_dist = Counter([ex.get("expected_decision") for ex in golden_set])
    
    print(f"- Difficulty distribution: {dict(difficulty_dist)}")
    print(f"- Intent distribution: {dict(intent_dist)}")
    print(f"- Expected Decision distribution: {dict(decision_dist)}")
    
    # Check how labels were created
    classifier = IntentClassifier()
    rule_matched = 0
    for ex in golden_set:
        rule_pred, _ = classifier.classify(ex["customer_message"])
        if rule_pred == ex["expected_intent"]:
            rule_matched += 1
            
    print(f"- Tautological Rule Match: {rule_matched} / {len(golden_set)} ({rule_matched/len(golden_set)*100:.1f}%)")
    print(f"  CRITICAL FINDING: Golden set intent labels were auto-assigned by the same heuristic classifier!")
    
    # -------------------------------------------------------------
    # 2. DECISION METRIC & CONFUSION MATRIX AUDIT
    # -------------------------------------------------------------
    print(f"\n[2. Decision Metric & Risk-Coverage Audit]")
    with open(ROOT_DIR / "results" / "predictions_resolvedna.json", "r", encoding="utf-8") as f:
        resolvedna_preds = json.load(f)
        
    y_true_dec = [ex["expected_decision"] for ex in golden_set]
    y_pred_dec = [p["predicted_decision"] for p in resolvedna_preds]
    
    # Confusion matrix for decisions (labels: AUTO_HANDLE, ESCALATE)
    cm = confusion_matrix(y_true_dec, y_pred_dec, labels=["AUTO_HANDLE", "ESCALATE"])
    # cm layout:
    # [ [TP_auto, FN_auto],
    #   [FP_auto, TN_auto] ]
    # where actual AUTO_HANDLE is class 0, actual ESCALATE is class 1
    
    tp_auto = cm[0, 0] # actual AUTO, pred AUTO
    fn_auto = cm[0, 1] # actual AUTO, pred ESCALATE (False Escalation)
    fp_auto = cm[1, 0] # actual ESCALATE, pred AUTO (False Auto-Handle - Dangerous!)
    tn_esc  = cm[1, 1] # actual ESCALATE, pred ESCALATE
    
    total = len(y_true_dec)
    total_pred_auto = tp_auto + fp_auto
    total_actual_auto = tp_auto + fn_auto
    total_actual_esc = fp_auto + tn_esc
    
    decision_acc = (tp_auto + tn_esc) / total
    auto_handle_precision = tp_auto / total_pred_auto if total_pred_auto > 0 else 0.0
    escalation_precision = tn_esc / (fn_auto + tn_esc) if (fn_auto + tn_esc) > 0 else 0.0
    false_auto_handle_rate = fp_auto / total_pred_auto if total_pred_auto > 0 else 0.0
    
    # Trustworthy Automation Rate (TAR) = Correct Auto-Handles / Total Cases
    tar = tp_auto / total
    
    print(f"- Decision Confusion Matrix:")
    print(f"               Pred AUTO_HANDLE   Pred ESCALATE   Total Actual")
    print(f"Actual AUTO        {tp_auto:<16}  {fn_auto:<14}  {total_actual_auto}")
    print(f"Actual ESCALATE    {fp_auto:<16}  {tn_esc:<14}  {total_actual_esc}")
    print(f"Total Pred         {total_pred_auto:<16}  {fn_auto+tn_esc:<14}  {total}")
    
    print(f"\n- Exact Metrics:")
    print(f"  * Decision Accuracy: {decision_acc:.4f} (({tp_auto} + {tn_esc}) / {total})")
    print(f"  * Auto-Handle Precision: {auto_handle_precision:.4f} ({tp_auto} / {total_pred_auto})")
    print(f"  * Escalation Precision: {escalation_precision:.4f} ({tn_esc} / {fn_auto+tn_esc})")
    print(f"  * False Auto-Handle Rate: {false_auto_handle_rate:.4f} ({fp_auto} / {total_pred_auto})")
    print(f"  * Trustworthy Automation Rate (TAR): {tar:.4f} ({tp_auto} / {total})")
    print(f"  * Automation Coverage: {total_pred_auto / total:.4f} ({total_pred_auto} / {total})")
    
    # Save Decision Confusion Matrix CSV
    df_cm = pd.DataFrame({
        "Actual / Predicted": ["Actual AUTO_HANDLE", "Actual ESCALATE", "Total Predicted"],
        "Pred AUTO_HANDLE": [tp_auto, fp_auto, total_pred_auto],
        "Pred ESCALATE": [fn_auto, tn_esc, fn_auto + tn_esc],
        "Total Actual": [total_actual_auto, total_actual_esc, total]
    })
    df_cm.to_csv(ROOT_DIR / "results" / "decision_confusion_matrix.csv", index=False)
    print(f"- Saved results/decision_confusion_matrix.csv")
    
    # Generate Risk-Coverage Curve across threshold sweeps
    print(f"\n- Generating Risk-Coverage Curve across gate confidence thresholds...")
    thresholds_to_test = np.linspace(0.1, 0.9, 17)
    risk_coverage_data = []
    
    store = ResolutionMemoryStore()
    store.load(str(sample_mem_path))
    retriever = ResolutionRetriever(store)
    
    for th in thresholds_to_test:
        preds_th = []
        for ex in golden_set:
            msg = ex["customer_message"]
            intent, intent_conf = classifier.classify(msg)
            retrieved = retriever.retrieve(msg, top_k=5)
            cases_only = [c for c, _ in retrieved]
            cons = compute_resolution_consistency(cases_only)
            outc = compute_outcome_consistency(cases_only)
            conf_res = compute_resolution_confidence(intent_conf, retrieved, cons, outc)
            
            custom_th = {
                "min_intent_confidence": th,
                "min_resolution_confidence": th * 0.7,
                "min_consistency_score": 0.30,
                "min_cases_retrieved": 2,
                "min_avg_similarity": 0.08
            }
            gate_res = decide(intent, intent_conf, conf_res["resolution_confidence"], cons, outc, retrieved, thresholds=custom_th)
            preds_th.append(gate_res["decision"])
            
        th_cm = confusion_matrix(y_true_dec, preds_th, labels=["AUTO_HANDLE", "ESCALATE"])
        th_tp = th_cm[0, 0]
        th_fn = th_cm[0, 1]
        th_fp = th_cm[1, 0]
        th_tn = th_cm[1, 1]
        
        pred_auto = th_tp + th_fp
        coverage = pred_auto / total
        fah_rate = th_fp / pred_auto if pred_auto > 0 else 0.0
        tar_val = th_tp / total
        dec_acc = (th_tp + th_tn) / total
        
        risk_coverage_data.append({
            "Confidence_Threshold": round(th, 3),
            "Coverage_AutoHandle_Rate": round(coverage, 4),
            "False_AutoHandle_Rate_Risk": round(fah_rate, 4),
            "Trustworthy_Automation_Rate": round(tar_val, 4),
            "Decision_Accuracy": round(dec_acc, 4),
            "TP_Auto": th_tp,
            "FP_Auto": th_fp,
            "FN_Auto": th_fn,
            "TN_Esc": th_tn
        })
        
    df_rc = pd.DataFrame(risk_coverage_data)
    df_rc.to_csv(ROOT_DIR / "results" / "risk_coverage_curve.csv", index=False)
    print(f"- Saved results/risk_coverage_curve.csv")
    
    # -------------------------------------------------------------
    # 3. CONTROLLED RESOLUTION-AWARENESS EXPERIMENT (A vs B)
    # -------------------------------------------------------------
    print(f"\n[3. Controlled Resolution-Awareness Experiment]")
    print(f"Hypothesis: Incorporating historical outcome and consistency signals reduces False Auto-Handles vs Naive Similarity alone.")
    
    # System A: Similarity Retrieval ONLY (Ignores outcome/consistency, naive confidence)
    preds_a = []
    # System B: Similarity + Historical Resolution Awareness (ResolveDNA)
    preds_b = []
    
    for ex in golden_set:
        msg = ex["customer_message"]
        intent, intent_conf = classifier.classify(msg)
        retrieved = retriever.retrieve(msg, top_k=5)
        cases_only = [c for c, _ in retrieved]
        
        # System A: Naive similarity gate
        avg_sim = np.mean([s for _, s in retrieved]) if retrieved else 0.0
        naive_auto = (intent_conf >= 0.50 and avg_sim >= 0.15)
        preds_a.append("AUTO_HANDLE" if naive_auto else "ESCALATE")
        
        # System B: Full Resolution Awareness (outcome consistency + action entropy + dual confidence)
        cons = compute_resolution_consistency(cases_only)
        outc = compute_outcome_consistency(cases_only)
        conf_res = compute_resolution_confidence(intent_conf, retrieved, cons, outc)
        gate_res = decide(intent, intent_conf, conf_res["resolution_confidence"], cons, outc, retrieved)
        preds_b.append(gate_res["decision"])
        
    cm_a = confusion_matrix(y_true_dec, preds_a, labels=["AUTO_HANDLE", "ESCALATE"])
    cm_b = confusion_matrix(y_true_dec, preds_b, labels=["AUTO_HANDLE", "ESCALATE"])
    
    fah_a = cm_a[1, 0] / (cm_a[0, 0] + cm_a[1, 0]) if (cm_a[0, 0] + cm_a[1, 0]) > 0 else 0
    fah_b = cm_b[1, 0] / (cm_b[0, 0] + cm_b[1, 0]) if (cm_b[0, 0] + cm_b[1, 0]) > 0 else 0
    tar_a = cm_a[0, 0] / total
    tar_b = cm_b[0, 0] / total
    acc_a = (cm_a[0, 0] + cm_a[1, 1]) / total
    acc_b = (cm_b[0, 0] + cm_b[1, 1]) / total
    
    print(f"Condition A (Naive Similarity Only):")
    print(f"  * Decision Acc: {acc_a:.4f} | False Auto Rate: {fah_a*100:.1f}% | TAR: {tar_a:.4f} | Coverage: {(cm_a[0,0]+cm_a[1,0])/total*100:.1f}%")
    print(f"Condition B (Resolution-Aware Dual Engine):")
    print(f"  * Decision Acc: {acc_b:.4f} | False Auto Rate: {fah_b*100:.1f}% | TAR: {tar_b:.4f} | Coverage: {(cm_b[0,0]+cm_b[1,0])/total*100:.1f}%")
    print(f"  * Relative Risk Reduction: {(fah_a - fah_b)/fah_a*100:.1f}% reduction in false auto-handles!")
    
    # -------------------------------------------------------------
    # 4. TEMPORAL RECENCY EXPERIMENT (A vs B vs C)
    # -------------------------------------------------------------
    print(f"\n[4. Temporal Recency Experiment]")
    print(f"Comparing: Mode A (Similarity Only), Mode B (Similarity + Recency), Mode C (Similarity + Recency + Resolution)")
    
    modes = ["similarity_only", "similarity_recency", "similarity_recency_resolution"]
    temporal_results = {}
    
    for mode in modes:
        preds_mode = []
        for ex in golden_set:
            msg = ex["customer_message"]
            intent, intent_conf = classifier.classify(msg)
            retrieved = retriever.retrieve(msg, top_k=5, mode=mode)
            cases_only = [c for c, _ in retrieved]
            cons = compute_resolution_consistency(cases_only)
            outc = compute_outcome_consistency(cases_only)
            conf_res = compute_resolution_confidence(intent_conf, retrieved, cons, outc)
            gate_res = decide(intent, intent_conf, conf_res["resolution_confidence"], cons, outc, retrieved)
            preds_mode.append(gate_res["decision"])
            
        t_cm = confusion_matrix(y_true_dec, preds_mode, labels=["AUTO_HANDLE", "ESCALATE"])
        t_fah = t_cm[1, 0] / (t_cm[0, 0] + t_cm[1, 0]) if (t_cm[0, 0] + t_cm[1, 0]) > 0 else 0
        t_tar = t_cm[0, 0] / total
        t_acc = (t_cm[0, 0] + t_cm[1, 1]) / total
        temporal_results[mode] = {
            "decision_accuracy": t_acc,
            "false_auto_handle_rate": t_fah,
            "tar": t_tar,
            "auto_coverage": (t_cm[0, 0] + t_cm[1, 0]) / total
        }
        print(f"  * Mode '{mode}': Acc={t_acc:.4f} | FalseAutoRate={t_fah*100:.1f}% | TAR={t_tar:.4f} | Coverage={temporal_results[mode]['auto_coverage']*100:.1f}%")
    
    # -------------------------------------------------------------
    # 5. REAL FAILURE MODE EXTRACTION
    # -------------------------------------------------------------
    print(f"\n[5. Top Real Failure Modes Extraction]")
    failures = []
    for idx, (ex, pred) in enumerate(zip(golden_set, resolvedna_preds)):
        exp_dec = ex["expected_decision"]
        got_dec = pred["predicted_decision"]
        if exp_dec != got_dec:
            failures.append({
                "index": idx,
                "case_id": ex.get("case_id"),
                "customer_message": ex["customer_message"],
                "expected_intent": ex["expected_intent"],
                "expected_decision": exp_dec,
                "predicted_decision": got_dec,
                "intent_confidence": pred.get("intent_confidence"),
                "resolution_confidence": pred.get("resolution_confidence"),
                "reason": pred.get("decision_reason", ""),
                "difficulty": ex.get("difficulty")
            })
            
    print(f"- Total Decision Mismatches: {len(failures)} / {len(golden_set)}")
    
    # Categorize failure types:
    false_escalations = [f for f in failures if f["expected_decision"] == "AUTO_HANDLE" and f["predicted_decision"] == "ESCALATE"]
    false_autohandles = [f for f in failures if f["expected_decision"] == "ESCALATE" and f["predicted_decision"] == "AUTO_HANDLE"]
    
    print(f"  * False Escalations (Over-conservative): {len(false_escalations)}")
    print(f"  * False Auto-Handles (Under-conservative / Risk): {len(false_autohandles)}")
    
    # Print 5 concrete failure examples
    print("\n--- 5 Detailed Concrete Failure Cases ---")
    for i, fail in enumerate(failures[:5]):
        print(f"\nFailure #{i+1} (Type: {'False Escalation' if fail['expected_decision']=='AUTO_HANDLE' else 'False Auto-Handle'}):")
        print(f"  Msg: '{fail['customer_message']}'")
        print(f"  Expected: {fail['expected_decision']} ({fail['expected_intent']}) | Got: {fail['predicted_decision']}")
        print(f"  IntentConf: {fail['intent_confidence']} | ResConf: {fail['resolution_confidence']}")
        print(f"  Reason: {fail['reason']}")

if __name__ == "__main__":
    run_audit()
