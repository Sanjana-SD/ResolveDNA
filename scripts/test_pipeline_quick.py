import sys, os
sys.path.insert(0, ".")
from src.pipeline import ResolveDNAPipeline

p = ResolveDNAPipeline(memory_path="data/processed/resolution_memory_sample.pkl")

tests = [
    "My package says delivered but nothing is at my door!",
    "I've been waiting two weeks for my refund.",
    "Prime video keeps buffering on my Fire TV, error code 7031",
]

for msg in tests:
    r = p.process(msg)
    print("=" * 60)
    print(f"CUSTOMER: {msg}")
    print(f"INTENT: {r['intent']} (conf={r['intent_confidence']})")
    print(f"RES_CONF: {r['resolution_confidence']}")
    print(f"DECISION: {r['decision']}")
    print(f"REASON: {r['decision_reason'][:120]}")
    print(f"ACTION: {r['next_best_action']['action']}")
    print(f"VERIFY: {r['verification']['status']}")
    print()
