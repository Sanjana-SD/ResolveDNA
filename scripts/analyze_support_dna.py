import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.discovery.pattern_miner import analyze_support_dna, generate_markdown_report

TRAJECTORIES_PATH = "data/processed/amazon_trajectories.jsonl"
REPORT_PATH = "results/support_dna_analysis.md"
JSON_PATH = "results/support_dna_summary.json"

def main():
    if not os.path.exists(TRAJECTORIES_PATH):
        print(f"Error: Trajectories file not found at {TRAJECTORIES_PATH}")
        sys.exit(1)
        
    print("=== RESOLVE DNA — MINING SUPPORT DNA PATTERNS ===")
    summary = analyze_support_dna(TRAJECTORIES_PATH)
    
    generate_markdown_report(summary, REPORT_PATH)
    
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to {JSON_PATH}")

if __name__ == "__main__":
    main()
