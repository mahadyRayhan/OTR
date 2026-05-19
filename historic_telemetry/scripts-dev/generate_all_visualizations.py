import os
import subprocess
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
VIZ_DIR = ROOT_DIR / "visualizations"

# List of all the evaluation scripts in the order they should run
EVALUATORS = [
    "visualize_ablation_metrics.py",     # Base System Metrics (GTS, Pareto, Sweet Spot)
    "agent_nqs_evaluator.py",            # Agent-Level NQS (Boxplots & Mode Comparison)
    "first_pass_evaluator.py",           # First-Pass Success Rate (Baseline Reliability)
    "reflexion_evaluator.py",            # Self-Healing (Recovery Depth, Reflexion Quality)
    "communication_evaluator.py",        # Communication Stats (CRS, CUS, DSR)
    "advanced_comm_evaluator.py",        # Advanced Comm (ΔQ, CE, Compression Ratio)
    "knowledge_transfer_evaluator.py",   # Knowledge Transfer (Atom Preservation)
    "pairwise_evaluator.py",             # LLM-as-a-Judge A/B Testing (Win Rate & Margin)
    "offline_llm_judge.py"               # System Qualitative Evaluation (Helpfulness, Actionability)
]

def print_header(title):
    print("\n" + "═" * 70)
    print(f"🚀 RUNNING: {title}")
    print("═" * 70)

def main():
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║      MASTER VISUALIZATION & EVALUATION GENERATOR SUITE          ║")
    print("╚═════════════════════════════════════════════════════════════════╝")
    print(f"📂 Output Directory: {VIZ_DIR.resolve()}")
    
    start_time = time.time()
    success_count = 0
    failure_count = 0
    failed_scripts = []

    for script_name in EVALUATORS:
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            print(f"\n⚠️ WARNING: Script '{script_name}' not found in {SCRIPTS_DIR}. Skipping.")
            continue
            
        print_header(script_name)
        
        try:
            # Run the script as a subprocess
            result = subprocess.run(
                ["python", str(script_path)],
                check=True,
                capture_output=False # Set to False so the output prints directly to the terminal
            )
            success_count += 1
            print(f"\n✅ {script_name} completed successfully.")
            
        except subprocess.CalledProcessError as e:
            failure_count += 1
            failed_scripts.append(script_name)
            print(f"\n❌ {script_name} FAILED with exit code {e.returncode}.")
            print("Moving to next script...")

    # --- FINAL SUMMARY ---
    total_time = time.time() - start_time
    print("\n\n" + "╔" + "═" * 68 + "╗")
    print("║" + "EVALUATION GENERATION COMPLETE".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Total Time Elapsed : {total_time:.2f} seconds".ljust(69) + "║")
    print(f"║  Scripts Succeeded  : {success_count}/{len(EVALUATORS)}".ljust(69) + "║")
    print(f"║  Scripts Failed     : {failure_count}/{len(EVALUATORS)}".ljust(69) + "║")
    
    if failure_count > 0:
        print("╠" + "═" * 68 + "╣")
        print("║  Failed Scripts:".ljust(69) + "║")
        for fail in failed_scripts:
            print(f"║    - {fail}".ljust(69) + "║")
            
    print("╚" + "═" * 68 + "╝")
    print(f"\n🎉 All graphs have been saved to: {VIZ_DIR.resolve()}")

if __name__ == "__main__":
    main()