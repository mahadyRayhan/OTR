"""
L1_headline.py — Layer 1: Headline System Performance
======================================================
The single most important question: did the system work?
Two figures, no qualifications, no ablations yet.

  fig01_pass_rate  — bar chart: % of runs scoring ≥ 7 per mode
  fig02_score_dist — violin+box: full judge score distribution per mode

Run:  python scripts/eval/L1_headline.py
Out:  figures/paper/fig01_pass_rate.{png,pdf}
      figures/paper/fig02_score_dist.{png,pdf}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from L0_loader import *

import numpy as np
import matplotlib.ticker as mticker


def fig01_pass_rate(runs: list[dict]):
    bm = by_mode(runs)
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, m in enumerate(MODES):
        mr   = bm[m]
        rate = np.mean([r["passed"] for r in mr]) * 100 if mr else 0
        s    = STYLES[m]
        ax.bar(i, rate, width=0.55,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=2.5, zorder=3)
        ax.text(i, rate + 2.5, f"{rate:.0f}%",
                ha="center", va="bottom",
                fontsize=26, fontweight="bold", color=s["edge"])
        ax.text(i, -7, f"n={len(mr)}",
                ha="center", va="top",
                fontsize=20, color="#555555")

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylim(-10, 115)
    ax.set_ylabel(f"Pass Rate  (score ≥ {PASS_THRESHOLD},  %)")
    ax.set_title("System Success Rate by Communication Mode")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig01_pass_rate")


def fig02_score_dist(runs: list[dict]):
    bm   = by_mode(runs)
    vals = [[r["judge_score"] for r in bm[m]] for m in MODES]

    fig, ax = plt.subplots(figsize=(10, 7))

    parts = ax.violinplot(
        [v or [0] for v in vals],
        positions=range(len(MODES)),
        widths=0.60, showmedians=False, showextrema=False,
    )
    for pc, m in zip(parts["bodies"], MODES):
        s = STYLES[m]
        pc.set_facecolor(s["face"]); pc.set_edgecolor(s["edge"])
        pc.set_linewidth(2); pc.set_alpha(0.85)

    bp = ax.boxplot(
        [v or [0] for v in vals],
        positions=range(len(MODES)), widths=0.20,
        patch_artist=True, showfliers=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2),
        flierprops=dict(marker="o", markersize=5, alpha=0.55, linestyle="none"),
    )
    for patch, m in zip(bp["boxes"], MODES):
        s = STYLES[m]
        patch.set_facecolor(s["face"]); patch.set_hatch(s["hatch"])
        patch.set_edgecolor(s["edge"])

    ax.axhline(PASS_THRESHOLD, color="steelblue", linestyle="--",
               linewidth=2, alpha=0.7)
    ax.text(len(MODES) - 0.05, PASS_THRESHOLD + 0.2,
            f"pass ≥ {PASS_THRESHOLD}", va="bottom", ha="right",
            fontsize=18, color="steelblue", fontweight="bold")

    for i, (m, v) in enumerate(zip(MODES, vals)):
        if v:
            med = np.median(v)
            ax.text(i + 0.35, med, f"{med:.0f}",
                    va="center", fontsize=20, fontweight="bold",
                    color=STYLES[m]["edge"])

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylim(0, 12)
    ax.set_ylabel("Final Judge Score  (0 – 10)")
    ax.set_title("Judge Score Distribution by Communication Mode")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    save_fig(fig, "fig02_score_dist")


def main():
    print("\n── Layer 1: Headline Performance ──────────────────────────")
    apply_style()
    runs = load_all()
    fig01_pass_rate(runs)
    fig02_score_dist(runs)
    print("Layer 1 done.\n")


if __name__ == "__main__":
    main()
