import os
import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parser import parse_amazon_trajectories
from src.intent.taxonomy import IntentClassifier
from src.resolution.store import ResolutionCase, ResolutionMemoryStore

TRAJECTORIES_PATH = "data/processed/amazon_trajectories.jsonl"
MEMORY_PICKLE_PATH = "data/processed/resolution_memory.pkl"
SAMPLE_MEMORY_PICKLE_PATH = "data/processed/resolution_memory_sample.pkl"

def main():
    if not os.path.exists(TRAJECTORIES_PATH):
        print(f"Error: Trajectories file not found at {TRAJECTORIES_PATH}")
        sys.exit(1)
        
    print("=== RESOLVE DNA — BUILDING RESOLUTION MEMORY STORE ===")
    classifier = IntentClassifier()
    store = ResolutionMemoryStore()
    
    with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            
            problem_text = t['customer_problem']
            intent_id, conf = classifier.classify(problem_text)
            
            actions = t.get('brand_actions', [])
            actions_str = " -> ".join(actions) if actions else "no_action"
            pattern_sig = f"[{actions_str}] => {t['final_state']}"
            
            case = ResolutionCase(
                case_id=t['conversation_id'],
                customer_problem=problem_text,
                intent=intent_id,
                brand_actions=actions,
                customer_reactions=t.get('customer_reactions', []),
                final_state=t['final_state'],
                observed_resolution_signal=t['observed_resolution_signal'],
                trajectory_length=t['trajectory_length'],
                created_at=t['created_at'],
                resolution_pattern=pattern_sig,
                evidence_score=1.0 if t['observed_resolution_signal'] else 0.6
            )
            store.add_case(case)
            
    print(f"Successfully processed {len(store):,} resolution cases.")
    store.save(MEMORY_PICKLE_PATH)
    
    # Also save a 5,000-case sample store for fast reproduction (<15 minutes)
    sample_store = ResolutionMemoryStore()
    sample_store.cases = store.cases[:5000]
    sample_store.case_map = {c.case_id: c for c in sample_store.cases}
    sample_store.save(SAMPLE_MEMORY_PICKLE_PATH)
    print(f"Saved 5,000 case evaluation sample store to {SAMPLE_MEMORY_PICKLE_PATH}")

if __name__ == "__main__":
    main()
