"""
L5_failure.py — Layer 5: Failure Modes (honesty section)
=========================================================
Discrete defects found and what they tell you about the design.
Reviewers respect papers that show failure modes openly.

  fig10_retry_traj   — line per mode: judge score trajectory across retries
  table04_failures   — failure mode counts per mode (CSV + console)

Run:  python scripts/eval/L5_failure.py
Out:  figures/paper/fig10_retry_traj.{png,pdf}
      figures/paper/table04_failures.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from L0_loader import *

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.ticker as mticker


def fig10_retry_traj(runs: list[dict]):
    """
    Line per mode: mean judge score at each step.
    Step 0 = initial score (before any retry).
    Step k = mean score_after at retry_num=k.
    """
    bm = by_mode(runs)

    fig, ax = plt.subplots(figsize=(11, 7))

    for m in MODES:
        mr = bm[m]
        s  = STYLES[m]

        # Step 0: initial score
        # For runs WITH retries: score_before from retry_num=1 event
        # For runs WITHOUT retries: final_judge_score (they passed first try)
        step0 = []
        for r in mr:
            evs = sorted(r["retry_events"], key=lambda e: e.get("retry_num", 0))
            if evs:
                step0.append(evs[0].get("score_before", r["judge_score"]))
            else:
                step0.append(r["judge_score"])

        # Steps 1, 2, …: mean score_after at each retry_num
        max_retries = max((r["retries"] for r in mr), default=0)
        steps  = [np.mean(step0)]
        ns     = [len(step0)]
        x_vals = [0]

        for k in range(1, max_retries + 1):
            scores_k = [
                ev.get("score_after", 0)
                for r in mr for ev in r["retry_events"]
                if ev.get("retry_num") == k
            ]
            if scores_k:
                steps.append(np.mean(scores_k))
                ns.append(len(scores_k))
                x_vals.append(k)

        ax.plot(x_vals, steps,
                color=s["edge"], linewidth=3,
                marker="o", markersize=12,
                markerfacecolor=s["face"],
                markeredgecolor=s["edge"],
                markeredgewidth=2,
                label=s["label"], zorder=4)

        # Annotate each point
        for x, y, n in zip(x_vals, steps, ns):
            ax.text(x + 0.04, y + 0.25, f"{y:.1f}\n(n={n})",
                    fontsize=17, color=s["edge"], fontweight="bold",
                    va="bottom")

    ax.axhline(PASS_THRESHOLD, color="steelblue", linestyle="--",
               linewidth=2, alpha=0.7)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1,
            PASS_THRESHOLD + 0.2,
            f"pass ≥ {PASS_THRESHOLD}", va="bottom", ha="right",
            fontsize=18, color="steelblue", fontweight="bold")

    ax.set_xlabel("Reflexion Retry Step  (0 = initial attempt)")
    ax.set_ylabel("Mean Judge Score  (0 – 10)")
    ax.set_title("Score Trajectory Across Reflexion Retries")
    ax.set_ylim(0, 11.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig10_retry_traj")


def table04_failures(runs: list[dict]):
    bm = by_mode(runs)

    # Collect all non-null failure modes
    all_modes: set[str] = set()
    for m in MODES:
        for n in (n for r in bm[m] for n in r["nodes"]):
            if n["failure_mode"]:
                all_modes.add(n["failure_mode"])

    rows = []
    for fm in sorted(all_modes):
        row = {"Failure Mode": fm.replace("_", " ").title()}
        for m in MODES:
            nodes_m  = [n for r in bm[m] for n in r["nodes"]]
            count    = sum(1 for n in nodes_m if n["failure_mode"] == fm)
            total    = len(nodes_m)
            row[STYLES[m]["label"]] = f"{count}  ({count/total*100:.1f}%)" if total else "0"
        rows.append(row)

    # Also add totals row
    totals = {"Failure Mode": "Any Failure"}
    for m in MODES:
        nodes_m = [n for r in bm[m] for n in r["nodes"]]
        count   = sum(1 for n in nodes_m if n["failure_mode"])
        total   = len(nodes_m)
        totals[STYLES[m]["label"]] = f"{count}  ({count/total*100:.1f}%)" if total else "0"
    rows.append(totals)

    df = pd.DataFrame(rows)
    print("\n  ── Table 4: Failure Mode Counts per Mode ──")
    save_table(df, "table04_failures")


def main():
    print("\n── Layer 5: Failure Modes ──────────────────────────────────")
    apply_style()
    runs = load_all()
    fig10_retry_traj(runs)
    table04_failures(runs)
    print("Layer 5 done.\n")


if __name__ == "__main__":
    main()
