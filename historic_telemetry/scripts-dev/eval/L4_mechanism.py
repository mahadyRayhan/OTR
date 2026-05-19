"""
L4_mechanism.py — Layer 4: Mechanism — How Context Flow Drives Differences
===========================================================================
Explains WHY ADAPTIVE wins (or loses). Shows the CUGCR gateway in action.

  fig08_scatter     — 3-panel (1×3): CRS vs CUS scatter per mode, coloured by agent
  fig09_token_vol   — violin+box: context token volume per communication edge, per mode
  table03_gateway   — gateway behavior counts: CUGCR fired, CRS-blocked, raw-passed (CSV)

Run:  python scripts/eval/L4_mechanism.py
Out:  figures/paper/fig08_scatter.{png,pdf}
      figures/paper/fig09_token_vol.{png,pdf}
      figures/paper/table03_gateway.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from L0_loader import *

import numpy as np
import pandas as pd
import matplotlib.ticker as mticker


def fig08_scatter(runs: list[dict]):
    """1×3 scatter panels: CRS (relevance) vs CUS (utilization) per mode.
    Each dot = one communication edge. Colour = receiving agent type."""

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True, sharex=True)
    fig.suptitle(
        "Context Utilization (CUS) vs. Relevance (CRS) per Communication Edge",
        fontsize=28, fontweight="bold", y=1.02,
    )

    for ax, m in zip(axes, MODES):
        edges_m = [e for r in runs if r["mode"] == m for e in r["edges"]]
        ax.set_title(STYLES[m]["label"], fontsize=28, pad=12,
                     color=STYLES[m]["edge"], fontweight="bold")

        plotted = False
        for a in AGENTS:
            xs = [e["relevance"]   * 100 for e in edges_m if e["to_agent"] == a]
            ys = [e["utilization"] * 100 for e in edges_m if e["to_agent"] == a]
            if not xs:
                continue
            ax.scatter(xs, ys,
                       c=AGENT_COLORS[a], label=AGENT_LABELS[a],
                       s=70, alpha=0.75, edgecolors="black",
                       linewidths=0.8, zorder=3)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No context\npassed",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=22, color="gray")

        ax.set_xlabel("Context Relevance — CRS (%)")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.set_axisbelow(True)

        # Mean crosshair
        all_xs = [e["relevance"]   * 100 for e in edges_m]
        all_ys = [e["utilization"] * 100 for e in edges_m]
        if all_xs:
            ax.axvline(np.mean(all_xs), color="black", linestyle="--",
                       linewidth=1.5, alpha=0.45)
            ax.axhline(np.mean(all_ys), color="black", linestyle="--",
                       linewidth=1.5, alpha=0.45)

    axes[0].set_ylabel("Context Utilization — CUS (%)")

    # Shared legend
    handles = [
        mpatches.Patch(facecolor=AGENT_COLORS[a], label=AGENT_LABELS[a])
        for a in AGENTS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.04), fontsize=22)

    fig.tight_layout()
    save_fig(fig, "fig08_scatter")


def fig09_token_vol(runs: list[dict]):
    """Violin + box: context token volume per edge, by mode."""
    bm = by_mode(runs)

    # Collect all edge token counts per mode
    vals = [
        [e["context_tokens"] for r in bm[m] for e in r["edges"]]
        for m in MODES
    ]

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
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2),
    )
    for patch, m in zip(bp["boxes"], MODES):
        s = STYLES[m]
        patch.set_facecolor(s["face"]); patch.set_hatch(s["hatch"])
        patch.set_edgecolor(s["edge"])

    for i, (m, v) in enumerate(zip(MODES, vals)):
        med = np.median(v) if v else 0
        mu  = np.mean(v)   if v else 0
        ax.text(i + 0.35, max(np.percentile(v, 75), 1) if v else 1,
                f"med={med:.0f}\nμ={mu:.0f}",
                va="bottom", fontsize=17, fontweight="bold",
                color=STYLES[m]["edge"])

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Context Tokens per Communication Edge")
    ax.set_title("Context Volume Injected per Agent-to-Agent Edge")
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    save_fig(fig, "fig09_token_vol")


def table03_gateway(runs: list[dict]):
    rows = []
    for m in MODES:
        mr = [r for r in runs if r["mode"] == m]
        all_edges = [e for r in mr for e in r["edges"]]
        all_nodes = [n for r in mr for n in r["nodes"]]

        total_edges   = len(all_edges)
        ctx_passed    = sum(1 for e in all_edges if e["context_tokens"] > 0)
        crs_blocked   = sum(1 for e in all_edges if e["low_confidence"])
        cugcr_fired   = sum(1 for n in all_nodes if n["cugcr_triggered"])
        raw_passed    = ctx_passed - cugcr_fired

        mean_ctx_tok  = (
            np.mean([e["context_tokens"] for e in all_edges if e["context_tokens"] > 0])
            if any(e["context_tokens"] > 0 for e in all_edges) else 0
        )

        rows.append({
            "Mode":                     STYLES[m]["label"],
            "Total Edges":              total_edges,
            "Context Passed":           ctx_passed,
            "CRS-Blocked (no inject)":  crs_blocked,
            "CUGCR Compressed":         cugcr_fired,
            "Passed Raw":               max(raw_passed, 0),
            "Mean Tokens (when passed)":f"{mean_ctx_tok:.0f}",
        })

    df = pd.DataFrame(rows)
    print("\n  ── Table 3: Gateway Behavior Counts ──")
    save_table(df, "table03_gateway")


def main():
    print("\n── Layer 4: Mechanism ──────────────────────────────────────")
    apply_style()
    runs = load_all()
    fig08_scatter(runs)
    fig09_token_vol(runs)
    table03_gateway(runs)
    print("Layer 4 done.\n")


if __name__ == "__main__":
    main()
