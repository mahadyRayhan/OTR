"""
L6_categories.py — Layer 6: Per-Query-Category Breakdown
=========================================================
Stratify by query complexity. Reviewers always ask.
Better to volunteer it.

  table05_categories — pass rate per query category × mode (CSV + console)

Query groups:
  G1  Full-Stack Pipeline          (BQ01-05)
  G2  Coreference / Knowledge Xfer (BQ06-10)
  G3  Context Overload Stress Test (BQ11-15)
  G4  Missing Data / KB Gaps       (BQ16-20)
  G5  Adversarial / Ambiguous      (BQ21-25)
  G6  Single-Agent Baselines       (BQ26-30)

Run:  python scripts/eval/L6_categories.py
Out:  figures/paper/table05_categories.csv
      figures/paper/fig11_categories.{png,pdf}  (heatmap visual of Table 5)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from L0_loader import *

import numpy as np
import pandas as pd
import matplotlib.ticker as mticker

GROUP_DESCRIPTIONS = {
    "G1": "Full-Stack Pipeline",
    "G2": "Coreference / Knowledge Transfer",
    "G3": "Context Overload Stress Test",
    "G4": "Missing Data / KB Gaps",
    "G5": "Adversarial / Ambiguous Routing",
    "G6": "Single-Agent Baselines",
    "G7": "Context Volume Stress Test (>1500 tok)",
}


def table05_categories(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for gk in GROUPS:
        gr = [r for r in runs if r["group"] == gk]
        row = {
            "Group": gk,
            "Description": GROUP_DESCRIPTIONS.get(gk, ""),
            "n":           len(gr) // len(MODES),  # queries per mode
        }
        for m in MODES:
            mr = [r for r in gr if r["mode"] == m]
            if mr:
                pass_rate  = np.mean([r["passed"]      for r in mr]) * 100
                mean_score = np.mean([r["judge_score"] for r in mr])
                row[f"{STYLES[m]['label']} Pass%"] = f"{pass_rate:.0f}%"
                row[f"{STYLES[m]['label']} Score"] = f"{mean_score:.1f}"
            else:
                row[f"{STYLES[m]['label']} Pass%"] = "—"
                row[f"{STYLES[m]['label']} Score"] = "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n  ── Table 5: Pass Rate per Query Category × Mode ──")
    save_table(df, "table05_categories")
    return df


def fig11_categories(runs: list[dict]):
    """Heatmap visual of per-group pass rate — each cell = one (group, mode) pair."""
    group_keys = list(GROUPS.keys())

    matrix = np.zeros((len(MODES), len(group_keys)))
    for m_i, m in enumerate(MODES):
        for g_i, gk in enumerate(group_keys):
            gr = [r for r in runs if r["group"] == gk and r["mode"] == m]
            matrix[m_i, g_i] = np.mean([r["passed"] for r in gr]) * 100 if gr else 0

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(group_keys)))
    ax.set_xticklabels(
        [f"{gk}\n{GROUP_DESCRIPTIONS[gk]}" for gk in group_keys],
        fontsize=18,
    )
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([STYLES[m]["label"] for m in MODES], fontsize=22)
    ax.set_title("Pass Rate (%) per Query Category × Mode")

    for i in range(len(MODES)):
        for j in range(len(group_keys)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="black" if 20 < v < 80 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Pass Rate (%)", fontsize=20)

    fig.tight_layout()
    save_fig(fig, "fig11_categories")


def main():
    print("\n── Layer 6: Per-Query-Category Breakdown ────────────────────")
    apply_style()
    runs = load_all()
    table05_categories(runs)
    fig11_categories(runs)
    print("Layer 6 done.\n")


if __name__ == "__main__":
    main()
