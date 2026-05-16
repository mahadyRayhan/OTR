"""
L3_agent.py — Layer 3: Per-Agent Behavior
==========================================
Where does success and failure originate in the pipeline?

  fig06_agent_passrate — grouped bar: per-agent node pass rate × mode
  fig07_agent_heatmap  — heatmap: mean quality score per agent × mode
  table02_retry_rate   — per-agent retry trigger counts per mode (CSV + console)

Run:  python scripts/eval/L3_agent.py
Out:  figures/paper/fig06_*.{png,pdf}
      figures/paper/fig07_*.{png,pdf}
      figures/paper/table02_retry_rate.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from L0_loader import *

import numpy as np
import pandas as pd
import matplotlib.ticker as mticker


def _agent_quality_matrix(runs: list[dict]) -> dict:
    """Returns {mode: {agent: [quality_scores]}}"""
    mat: dict = {m: {a: [] for a in AGENTS} for m in MODES}
    for r in runs:
        for n in r["nodes"]:
            a = n["agent"]
            if a in AGENTS and n["quality_score"] is not None:
                mat[r["mode"]][a].append(float(n["quality_score"]))
    return mat


def fig06_agent_passrate(runs: list[dict]):
    mat = _agent_quality_matrix(runs)

    n_agents = len(AGENTS)
    group_w  = 1.0
    offsets  = np.linspace(-0.28, 0.28, len(MODES))

    fig, ax = plt.subplots(figsize=(14, 8))
    x_base  = np.arange(n_agents) * 1.5
    handles = []

    for m_i, m in enumerate(MODES):
        s    = STYLES[m]
        rates = [
            np.mean([q >= NODE_PASS_THRESHOLD for q in mat[m][a]]) * 100
            if mat[m][a] else 0
            for a in AGENTS
        ]
        bars = ax.bar(x_base + offsets[m_i], rates, width=0.28,
                      color=s["face"], hatch=s["hatch"],
                      edgecolor=s["edge"], linewidth=1.8, zorder=3)
        for bar, v in zip(bars, rates):
            if v > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5, f"{v:.0f}%",
                        ha="center", va="bottom",
                        fontsize=16, fontweight="bold", color=s["edge"])
        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"],
            hatch=s["hatch"], label=s["label"]))

    ax.set_xticks(x_base)
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS])
    ax.set_ylim(0, 120)
    ax.set_ylabel(f"Node Pass Rate  (quality ≥ {NODE_PASS_THRESHOLD},  %)")
    ax.set_title("Per-Agent Node Pass Rate by Ablation Mode")
    ax.legend(handles=handles, loc="lower right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig06_agent_passrate")


def fig07_agent_heatmap(runs: list[dict]):
    mat = _agent_quality_matrix(runs)

    matrix = np.array([
        [np.mean(mat[m][a]) if mat[m][a] else np.nan for a in AGENTS]
        for m in MODES
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS], fontsize=26)
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([STYLES[m]["label"] for m in MODES], fontsize=24)
    ax.set_title("Mean GoT Node Quality Score  (0 – 1)")

    for i in range(len(MODES)):
        for j in range(len(AGENTS)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=22, fontweight="bold",
                        color="black" if 0.35 < v < 0.75 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean Quality Score", fontsize=20)

    fig.tight_layout()
    save_fig(fig, "fig07_agent_heatmap")


def table02_retry_rate(runs: list[dict]):
    from collections import Counter

    rows = []
    for m in MODES:
        mr = [r for r in runs if r["mode"] == m]
        agents_retried = [
            (ev.get("failing_agent") or "unknown").lower()
            for r in mr for ev in r["retry_events"]
        ]
        c = Counter(agents_retried)
        n = len(mr)
        row = {"Mode": STYLES[m]["label"], "Total Retries": sum(c.values())}
        for a in AGENTS:
            row[f"{AGENT_LABELS[a]} Retries"] = c.get(a, 0)
        row["Retry Trigger Rate (%)"] = f"{sum(1 for r in mr if r['retries']>0)/n*100:.1f}" if n else "0"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n  ── Table 2: Per-Agent Retry Trigger Counts ──")
    save_table(df, "table02_retry_rate")


def main():
    print("\n── Layer 3: Per-Agent Behavior ─────────────────────────────")
    apply_style()
    runs = load_all()
    fig06_agent_passrate(runs)
    fig07_agent_heatmap(runs)
    table02_retry_rate(runs)
    print("Layer 3 done.\n")


if __name__ == "__main__":
    main()
