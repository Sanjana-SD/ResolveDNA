import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.resolution.store import ResolutionCase, ResolutionMemoryStore

class ResolutionRetriever:
    def __init__(self, store: ResolutionMemoryStore):
        self.store = store
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        self.doc_texts = [c.customer_problem for c in store.cases]
        if self.doc_texts:
            self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        else:
            self.doc_matrix = None
            
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "similarity_recency_resolution",
        filter_intent: Optional[str] = None
    ) -> List[Tuple[ResolutionCase, float]]:
        if not self.doc_texts or self.doc_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        
        results = []
        for idx, case in enumerate(self.store.cases):
            if filter_intent and case.intent != filter_intent and filter_intent != "other_unknown":
                continue
                
            base_sim = float(sim_scores[idx])
            if base_sim < 0.05:
                continue
                
            final_score = base_sim
            
            if mode == "similarity_recency":
                # Recency bonus heuristic (shorter index offset = slightly newer in dataset stream)
                recency_weight = 1.0 + (idx / len(self.store.cases)) * 0.10
                final_score *= recency_weight
            elif mode == "similarity_recency_resolution":
                recency_weight = 1.0 + (idx / len(self.store.cases)) * 0.10
                # Resolution signal multiplier: boost confirmed resolved cases, penalize repeated complaint escalations
                if case.observed_resolution_signal:
                    resolution_multiplier = 1.25
                elif case.final_state == "REPEATED_COMPLAINT_ESCALATED":
                    resolution_multiplier = 0.70
                else:
                    resolution_multiplier = 1.0
                final_score *= (recency_weight * resolution_multiplier)
                
            results.append((case, round(final_score, 4)))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
