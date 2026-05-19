"""
L2_efficiency.py — Layer 2: System-level Efficiency (the cost of success)
==========================================================================
Did it work, and at what price?

  fig03_latency     — bar: mean ± std wall latency per mode
  fig04_tokens      — bar: mean orchestrator token consumption (input + output)
  fig05_normalized  — bar: pass rate / (mean tokens / 1000) — success per 1k tokens
  table01_system    — full system-level summary table (CSV + console)

Run:  python scripts/eval/L2_efficiency.py
Out:  figures/paper/fig03_*.{png,pdf}
      figures/paper/table01_system.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from L0_loader import *

import numpy as np
import pandas as pd
import matplotlib.ticker as mticker


def fig03_latency(runs: list[dict]):
    bm = by_mode(runs)
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, m in enumerate(MODES):
        vals = [r["wall_s"] for r in bm[m]]
        mu, sd = np.mean(vals), np.std(vals)
        s = STYLES[m]
        ax.bar(i, mu, width=0.55, yerr=sd, capsize=8,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=2.5,
               error_kw={"elinewidth": 2, "ecolor": s["edge"]},
               zorder=3)
        ax.text(i, mu + sd + 1.5, f"{mu:.1f}s",
                ha="center", va="bottom",
                fontsize=22, fontweight="bold", color=s["edge"])

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Mean Wall Latency  (s)")
    ax.set_title("End-to-End Latency by Communication Mode")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig03_latency")


def fig04_tokens(runs: list[dict]):
    bm = by_mode(runs)
    fig, ax = plt.subplots(figsize=(10, 7))

    x   = np.arange(len(MODES))
    w   = 0.35
    bot = np.zeros(len(MODES))
    layer_info = [
        ("in_tokens",  "#6baed6", "//",  "Input tokens"),
        ("out_tokens", "#fd8d3c", "..",  "Output tokens"),
    ]

    for key, face, hatch, label in layer_info:
        vals = np.array([np.mean([r[key] for r in bm[m]]) if bm[m] else 0
                         for m in MODES])
        stds = np.array([np.std([r[key] for r in bm[m]]) if bm[m] else 0
                         for m in MODES])
        bars = ax.bar(x, vals, width=0.55, bottom=bot,
                      color=face, hatch=hatch, edgecolor="black",
                      linewidth=1.5, label=label, zorder=3)
        for bar, v, b in zip(bars, vals, bot):
            if v > 100:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        b + v / 2, f"{v:.0f}",
                        ha="center", va="center",
                        fontsize=18, fontweight="bold", color="black")
        bot += vals

    # total annotation
    for i, m in enumerate(MODES):
        total = np.mean([r["tokens"] for r in bm[m]]) if bm[m] else 0
        ax.text(i, bot[i] + 50, f"Σ{total:.0f}",
                ha="center", va="bottom",
                fontsize=20, fontweight="bold", color=STYLES[m]["edge"])

    ax.set_xticks(x)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Mean Orchestrator Tokens")
    ax.set_title("Token Consumption by Communication Mode")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig04_tokens")


def fig05_normalized(runs: list[dict]):
    """Pass rate normalised by token cost — efficiency metric."""
    bm = by_mode(runs)
    fig, ax = plt.subplots(figsize=(10, 7))

    scores = {}
    for m in MODES:
        mr = bm[m]
        if not mr:
            scores[m] = 0.0
            continue
        pass_rate  = np.mean([r["passed"] for r in mr]) * 100
        mean_tok   = np.mean([r["tokens"] for r in mr])
        scores[m]  = pass_rate / (mean_tok / 1000) if mean_tok > 0 else 0

    for i, m in enumerate(MODES):
        s = STYLES[m]
        ax.bar(i, scores[m], width=0.55,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=2.5, zorder=3)
        ax.text(i, scores[m] + 0.3, f"{scores[m]:.2f}",
                ha="center", va="bottom",
                fontsize=22, fontweight="bold", color=s["edge"])

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Pass Rate (%)  per 1 000 Tokens")
    ax.set_title("Cost-Normalised Success Rate")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_axisbelow(True)

    # Annotation explaining the metric
    ax.text(0.98, 0.05, "↑ Higher = more passes per token spent",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=18, color="gray", style="italic")

    fig.tight_layout()
    save_fig(fig, "fig05_normalized")


def table01_system(runs: list[dict]):
    bm = by_mode(runs)
    rows = []
    for m in MODES:
        mr = bm[m]
        if not mr:
            continue
        pass_rate   = np.mean([r["passed"] for r in mr]) * 100
        mean_score  = np.mean([r["judge_score"] for r in mr])
        mean_lat    = np.mean([r["wall_s"] for r in mr])
        mean_tok    = np.mean([r["tokens"] for r in mr])
        mean_calls  = np.mean([r["agent_calls"] for r in mr])
        rec_trigger = np.mean([r["retries"] > 0 for r in mr]) * 100
        rec_success = recovery_success_rate(mr)
        rows.append({
            "Mode":                   STYLES[m]["label"],
            "Pass Rate (%)":          f"{pass_rate:.1f}",
            "Mean Score":             f"{mean_score:.2f}",
            "Mean Latency (s)":       f"{mean_lat:.1f}",
            "Mean Tokens":            f"{mean_tok:.0f}",
            "Mean Agent Calls":       f"{mean_calls:.1f}",
            "Recovery Trigger (%)":   f"{rec_trigger:.1f}",
            "Recovery Success (%)":   f"{rec_success:.1f}",
        })

    df = pd.DataFrame(rows)
    print("\n  ── Table 1: System-Level Summary ──")
    save_table(df, "table01_system")


def main():
    print("\n── Layer 2: Efficiency ─────────────────────────────────────")
    apply_style()
    runs = load_all()
    fig03_latency(runs)
    fig04_tokens(runs)
    fig05_normalized(runs)
    table01_system(runs)
    print("Layer 2 done.\n")


if __name__ == "__main__":
    main()
