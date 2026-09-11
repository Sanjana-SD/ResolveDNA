"""Baseline Models for Comparison.

Implements three baselines to determine whether ResolveDNA's resolution-aware
approach provides genuine value:

  Baseline 1: Majority-class classifier (always predicts most common intent + auto-handle)
  Baseline 2: TF-IDF + Logistic Regression intent classifier + rule-based decision
  Baseline 3: Generic semantic retrieval + template response (standard RAG without resolution awareness)
"""
import os
import sys
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


class MajorityClassBaseline:
    """Baseline 1: Always predicts the most common intent and AUTO_HANDLE."""

    def __init__(self):
        self.majority_intent = "other_unknown"
        self.majority_action = "general_response"

    def fit(self, texts: List[str], intents: List[str]):
        intent_counts = Counter(intents)
        self.majority_intent = intent_counts.most_common(1)[0][0]

    def predict(self, customer_message: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 1.0,
            "resolution_confidence": 0.5,
            "decision": "AUTO_HANDLE",
            "decision_reason": "Majority class baseline always auto-handles",
            "next_best_action": {"action": self.majority_action},
            "draft_reply": "Thanks for reaching out! We're here to help. Could you share more details so we can assist you better?",
            "verification": {"status": "PASS"},
            "baseline": "majority_class",
        }


class TfidfLogregBaseline:
    """Baseline 2: TF-IDF + Logistic Regression for intent, rule-based decision."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        self.is_fitted = False
        self.action_map = {}

    def fit(self, texts: List[str], intents: List[str], actions: Optional[List[str]] = None):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, intents)
        self.is_fitted = True
        # Build intent -> most common action mapping
        if actions:
            intent_actions = {}
            for intent, action in zip(intents, actions):
                if intent not in intent_actions:
                    intent_actions[intent] = []
                intent_actions[intent].append(action)
            for intent, acts in intent_actions.items():
                self.action_map[intent] = Counter(acts).most_common(1)[0][0]

    def predict(self, customer_message: str) -> Dict[str, Any]:
        if not self.is_fitted:
            return {"intent": "other_unknown", "intent_confidence": 0.0, "decision": "ESCALATE",
                    "decision_reason": "Model not fitted", "baseline": "tfidf_logreg"}

        X = self.vectorizer.transform([customer_message])
        intent = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = float(max(proba))

        action = self.action_map.get(intent, "general_response")

        # Rule-based decision: auto-handle if confidence > 0.5
        if confidence > 0.5:
            decision = "AUTO_HANDLE"
            reason = f"TF-IDF+LR confidence {confidence:.2f} exceeds threshold"
        else:
            decision = "ESCALATE"
            reason = f"TF-IDF+LR confidence {confidence:.2f} below threshold"

        from src.generation.response_generator import TEMPLATE_RESPONSES
        draft_reply = TEMPLATE_RESPONSES.get(action, TEMPLATE_RESPONSES["general_response"])

        return {
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "resolution_confidence": round(confidence * 0.8, 3),  # proxy
            "decision": decision,
            "decision_reason": reason,
            "next_best_action": {"action": action},
            "draft_reply": draft_reply,
            "verification": {"status": "PASS"},
            "baseline": "tfidf_logreg",
        }


class GenericRAGBaseline:
    """Baseline 3: Standard semantic retrieval + template response (no resolution awareness).

    This baseline uses the same TF-IDF retrieval but ignores resolution signals,
    consistency, and outcome information. It directly retrieves similar tweets
    and generates a response based on the most similar case's action.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        self.doc_texts = []
        self.doc_intents = []
        self.doc_actions = []
        self.doc_matrix = None

    def fit(self, texts: List[str], intents: List[str], actions: List[str]):
        self.doc_texts = texts
        self.doc_intents = intents
        self.doc_actions = actions
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def predict(self, customer_message: str) -> Dict[str, Any]:
        if self.doc_matrix is None:
            return {"intent": "other_unknown", "decision": "ESCALATE", "baseline": "generic_rag"}

        query_vec = self.vectorizer.transform([customer_message])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()

        top_idx = np.argsort(similarities)[-5:][::-1]
        top_sims = similarities[top_idx]

        # Use majority intent from top-5
        top_intents = [self.doc_intents[i] for i in top_idx if similarities[i] > 0.05]
        if top_intents:
            intent = Counter(top_intents).most_common(1)[0][0]
        else:
            intent = "other_unknown"

        avg_sim = float(np.mean(top_sims))
        top_actions = [self.doc_actions[i] for i in top_idx if similarities[i] > 0.05]
        if top_actions:
            action = Counter(top_actions).most_common(1)[0][0]
        else:
            action = "general_response"

        # Simple threshold decision (no resolution awareness)
        if avg_sim > 0.10:
            decision = "AUTO_HANDLE"
            reason = f"Generic RAG similarity {avg_sim:.3f} exceeds threshold"
        else:
            decision = "ESCALATE"
            reason = f"Generic RAG similarity {avg_sim:.3f} below threshold"

        from src.generation.response_generator import TEMPLATE_RESPONSES
        draft_reply = TEMPLATE_RESPONSES.get(action, TEMPLATE_RESPONSES["general_response"])

        return {
            "intent": intent,
            "intent_confidence": round(min(1.0, avg_sim * 5), 3),
            "resolution_confidence": round(avg_sim * 3, 3),
            "decision": decision,
            "decision_reason": reason,
            "next_best_action": {"action": action},
            "draft_reply": draft_reply,
            "verification": {"status": "PASS"},
            "baseline": "generic_rag",
        }


def train_baselines(trajectories_path: str) -> Tuple[MajorityClassBaseline, TfidfLogregBaseline, GenericRAGBaseline]:
    """Train all three baselines on trajectory data."""
    print("[Baselines] Loading training data...")
    texts = []
    intents = []
    actions = []

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.intent.taxonomy import IntentClassifier

    classifier = IntentClassifier()

    with open(trajectories_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            text = t['customer_problem']
            intent, _ = classifier.classify(text)
            action = t['brand_actions'][0] if t.get('brand_actions') else 'general_response'
            texts.append(text)
            intents.append(intent)
            actions.append(action)

    print(f"[Baselines] Training on {len(texts):,} examples")

    b1 = MajorityClassBaseline()
    b1.fit(texts, intents)
    print(f"[Baselines] Majority class baseline: always predicts '{b1.majority_intent}'")

    b2 = TfidfLogregBaseline()
    b2.fit(texts, intents, actions)
    print("[Baselines] TF-IDF + Logistic Regression fitted")

    b3 = GenericRAGBaseline()
    b3.fit(texts, intents, actions)
    print("[Baselines] Generic RAG baseline fitted")

    return b1, b2, b3
