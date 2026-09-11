import os
import sys
import json
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parser import parse_amazon_trajectories

DATA_RAW_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/amazon_trajectories.jsonl"

def main():
    if not os.path.exists(DATA_RAW_PATH):
        print(f"Error: Raw dataset not found at {DATA_RAW_PATH}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print("=== RESOLVE DNA — PREPARING AMAZONHELP TRAJECTORIES ===")
    trajectories = parse_amazon_trajectories(DATA_RAW_PATH, target_brand="AmazonHelp")
    
    print(f"Saving {len(trajectories):,} trajectories to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')
            
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Successfully saved {OUTPUT_PATH} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
