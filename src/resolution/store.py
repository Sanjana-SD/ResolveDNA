import os
import sys
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class ResolutionCase:
    case_id: str
    customer_problem: str
    intent: str
    brand_actions: List[str]
    customer_reactions: List[str]
    final_state: str
    observed_resolution_signal: bool
    trajectory_length: int
    created_at: str
    resolution_pattern: str
    evidence_score: float = 1.0

class ResolutionMemoryStore:
    def __init__(self):
        self.cases: List[ResolutionCase] = []
        self.case_map: Dict[str, ResolutionCase] = {}
        
    def add_case(self, case: ResolutionCase):
        self.cases.append(case)
        self.case_map[case.case_id] = case
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.cases, f)
        print(f"[Resolution Memory] Saved {len(self.cases):,} cases to {filepath}")
        
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.cases = pickle.load(f)
        self.case_map = {c.case_id: c for c in self.cases}
        print(f"[Resolution Memory] Loaded {len(self.cases):,} cases from {filepath}")
        
    def __len__(self):
        return len(self.cases)
