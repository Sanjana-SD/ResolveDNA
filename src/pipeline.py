"""ResolveDNA End-to-End Pipeline.

Programmatic entry point: takes a customer message and returns a complete
auditable decision payload.

    Customer message
        -> Intent classification
        -> Historical case retrieval
        -> Resolution consistency
        -> Dual confidence (intent vs resolution)
        -> Evidence Sufficiency Gate
        -> Next Best Action
        -> Response generation
        -> Post-generation verification
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.taxonomy import IntentClassifier
from src.resolution.store import ResolutionMemoryStore, ResolutionCase
from src.retrieval.retriever import ResolutionRetriever
from src.decision.consistency import compute_resolution_consistency, compute_outcome_consistency
from src.decision.dual_confidence import compute_resolution_confidence
from src.decision.gate import decide
from src.generation.next_best_action import derive_next_best_action
from src.generation.response_generator import generate_response
from src.generation.verifier import verify_response


class ResolveDNAPipeline:
    """End-to-end pipeline for evidence-first customer support decisions."""

    def __init__(
        self,
        memory_path: str = "data/processed/resolution_memory.pkl",
        retrieval_mode: str = "similarity_recency_resolution",
        llm_api_key: Optional[str] = None,
        top_k: int = 5,
    ):
        self.classifier = IntentClassifier()
        self.store = ResolutionMemoryStore()
        self.store.load(memory_path)
        self.retriever = ResolutionRetriever(self.store)
        self.retrieval_mode = retrieval_mode
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.top_k = top_k

    def process(self, customer_message: str) -> Dict[str, Any]:
        """Process a single customer message through the full pipeline.

        Returns a complete auditable decision payload.
        """
        # --- Step 1: Intent Classification ---
        intent, intent_confidence = self.classifier.classify(customer_message)

        # --- Step 2: Retrieve Historical Cases ---
        retrieved = self.retriever.retrieve(
            query=customer_message,
            top_k=self.top_k,
            mode=self.retrieval_mode,
            filter_intent=None,  # Don't filter — let similarity handle relevance
        )
        retrieved_cases_only = [c for c, s in retrieved]

        # --- Step 3: Resolution Consistency ---
        consistency = compute_resolution_consistency(retrieved_cases_only)
        outcome = compute_outcome_consistency(retrieved_cases_only)

        # --- Step 4: Dual Confidence ---
        confidence_result = compute_resolution_confidence(
            intent_confidence=intent_confidence,
            retrieved_cases=retrieved,
            consistency=consistency,
            outcome=outcome,
        )
        resolution_confidence = confidence_result["resolution_confidence"]

        # --- Step 5: Evidence Sufficiency Gate ---
        gate_result = decide(
            intent=intent,
            intent_confidence=intent_confidence,
            resolution_confidence=resolution_confidence,
            consistency=consistency,
            outcome=outcome,
            retrieved_cases=retrieved,
        )
        decision = gate_result["decision"]

        # --- Step 6: Next Best Action ---
        nba = derive_next_best_action(
            intent=intent,
            retrieved_cases=retrieved,
            consistency=consistency,
        )

        # --- Step 7: Response Generation ---
        gen_result = generate_response(
            customer_message=customer_message,
            intent=intent,
            next_best_action=nba,
            retrieved_cases=retrieved,
            decision=decision,
            llm_api_key=self.llm_api_key,
        )

        # --- Step 8: Post-generation Verification ---
        verification = verify_response(
            draft_reply=gen_result["draft_reply"],
            customer_message=customer_message,
            next_best_action=nba,
            retrieved_cases=retrieved,
            intent=intent,
        )

        # --- Build complete payload ---
        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieved_cases": [
                {
                    "case_id": c.case_id,
                    "customer_problem": c.customer_problem[:200],
                    "brand_actions": c.brand_actions,
                    "final_state": c.final_state,
                    "resolved": c.observed_resolution_signal,
                    "similarity_score": round(s, 4),
                }
                for c, s in retrieved
            ],
            "resolution_pattern": consistency.get("dominant_action", "unknown"),
            "resolution_consistency": consistency,
            "outcome_consistency": outcome,
            "resolution_confidence": resolution_confidence,
            "confidence_components": confidence_result["components"],
            "confidence_explanation": confidence_result["explanation"],
            "next_best_action": nba,
            "decision": decision,
            "decision_reason": gate_result["reason"],
            "escalation_factors": gate_result["escalation_factors"],
            "draft_reply": gen_result["draft_reply"],
            "generation_method": gen_result["generation_method"],
            "verification": verification,
        }


def main():
    """Quick CLI demo."""
    import json

    pipeline = ResolveDNAPipeline(
        memory_path="data/processed/resolution_memory_sample.pkl"
    )

    test_messages = [
        "My package says delivered but it's not at my door!",
        "I've been waiting two weeks for my refund.",
        "Prime video keeps buffering on my Fire TV, error code 7031",
        "Why was I charged $14.99 for Prime? I never signed up!",
        "asdf lol idk what happened to my stuff",
    ]

    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"CUSTOMER: {msg}")
        result = pipeline.process(msg)
        print(f"INTENT: {result['intent']} (conf={result['intent_confidence']})")
        print(f"RESOLUTION CONF: {result['resolution_confidence']}")
        print(f"DECISION: {result['decision']}")
        print(f"REASON: {result['decision_reason']}")
        print(f"NEXT ACTION: {result['next_best_action']['action']}")
        print(f"REPLY: {result['draft_reply'][:200]}")
        print(f"VERIFICATION: {result['verification']['status']}")


if __name__ == "__main__":
    main()
