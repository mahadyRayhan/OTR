"""
run_paper_eval.py — Master runner for the 6-layer paper evaluation suite
=========================================================================
Runs all layers in sequence and prints a manifest of outputs.

Usage:
    conda activate agent
    python scripts/eval/run_paper_eval.py

Output directory:  figures/paper/
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import L1_headline
import L2_efficiency
import L3_agent
import L4_mechanism
import L5_failure
import L6_categories

from L0_loader import apply_style, load_all, FIG_DIR


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         OTR PAPER EVALUATION — 6-LAYER PIPELINE         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print(f"Output → {FIG_DIR.resolve()}\n")

    apply_style()
    runs = load_all()

    t0 = time.monotonic()

    L1_headline.apply_style = apply_style  # ensure shared style
    L1_headline.fig01_pass_rate(runs)
    L1_headline.fig02_score_dist(runs)

    L2_efficiency.fig03_latency(runs)
    L2_efficiency.fig04_tokens(runs)
    L2_efficiency.fig05_normalized(runs)
    L2_efficiency.table01_system(runs)

    L3_agent.fig06_agent_passrate(runs)
    L3_agent.fig07_agent_heatmap(runs)
    L3_agent.table02_retry_rate(runs)

    L4_mechanism.fig08_scatter(runs)
    L4_mechanism.fig09_token_vol(runs)
    L4_mechanism.table03_gateway(runs)

    L5_failure.fig10_retry_traj(runs)
    L5_failure.table04_failures(runs)

    L6_categories.table05_categories(runs)
    L6_categories.fig11_categories(runs)

    elapsed = time.monotonic() - t0

    # Manifest
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                     OUTPUT MANIFEST                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    outputs = sorted(FIG_DIR.glob("fig*.png")) + sorted(FIG_DIR.glob("table*.csv"))
    for p in sorted(FIG_DIR.iterdir()):
        print(f"║  {p.name:<54}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nAll done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
