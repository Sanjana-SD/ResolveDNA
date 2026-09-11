"""One-Click End-to-End Reproduction Script for ResolveDNA.

Runs the complete research and evaluation pipeline from raw/processed data to final metrics.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT_DIR = Path(__file__).resolve().parent

def log_step(step_name: str):
    print(f"\n{'='*70}")
    print(f"[*] [STEP] {step_name}")
    print(f"{'='*70}")

def run_cmd(cmd: str, desc: str):
    log_step(desc)
    start = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, shell=True, cwd=str(ROOT_DIR), env=env)
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"[ERROR] Failed: {desc} (Exit code: {res.returncode})")
        sys.exit(res.returncode)
    print(f"[SUCCESS] Finished {desc} in {elapsed:.2f}s")

def main():
    print("======================================================================")
    print("           ResolveDNA: Evidence-First Support AI Pipeline            ")
    print("         Hiver SDE Intern Take-Home Project Reproduction              ")
    print("======================================================================")
    
    # 1. Run Unit Tests
    run_cmd("pytest tests/ -v", "Running Test Suite (20 Unit & Integration Tests)")
    
    # 2. Build Golden Evaluation Set
    run_cmd("python scripts/create_eval_sample.py", "Generating Stratified Golden Evaluation Set")
    
    # 3. Run Benchmark Evaluation
    run_cmd("python scripts/evaluate.py", "Executing Full Benchmark Evaluation (ResolveDNA vs 3 Baselines)")
    
    # 4. Pipeline Smoke Demo
    run_cmd("python src/pipeline.py", "Running End-to-End Pipeline Smoke Test")
    
    print("\n" + "="*70)
    print("[COMPLETED] ALL STEPS COMPLETED SUCCESSFULLY!")
    print("   To launch the interactive UI, run:")
    print("   streamlit run app/app.py")
    print("="*70)

if __name__ == "__main__":
    main()
