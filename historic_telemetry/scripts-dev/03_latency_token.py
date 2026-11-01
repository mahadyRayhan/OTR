"""
03_latency_token.py — Latency & Token Trade-off Analysis
=========================================================
Analyses end-to-end latency, component breakdowns, and token consumption
across the three ablation modes (NONE / RAW / ADAPTIVE).

Figures produced
----------------
  03_latency_total.{png,pdf}     — violin + box: total wall latency per mode
  03_latency_breakdown.{png,pdf} — stacked bar: decomposer / node-sum / overhead
  03_token_cost.{png,pdf}        — bar: orchestrator token consumption per mode
  03_efficiency.{png,pdf}        — scatter: judge score vs latency, coloured by mode
  03_panel.{png,pdf}             — 2×2 combined panel

Run:   python scripts/eval/03_latency_token.py
Input: raw_telemetry/run_*.json
Out:   figures/03_latency_*.{png,pdf}
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
TELEMETRY_DIR = ROOT / "raw_telemetry"
FIG_DIR       = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODES = ["none", "raw", "adaptive"]
STYLES = {
    "none":     {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx",  "label": "NONE"},
    "raw":      {"face": "#ffcc99", "edge": "#cc3300", "hatch": "//",  "label": "RAW"},
    "adaptive": {"face": "#99ffcc", "edge": "#006633", "hatch": "..",  "label": "ADAPTIVE (Ours)"},
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        20,
    "font.weight":      "bold",
    "axes.labelsize":   34,
    "axes.titlesize":   38,
    "xtick.labelsize":  28,
    "ytick.labelsize":  28,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.fontsize":  26,
    "hatch.linewidth":  2.0,
})
sns.set_style("whitegrid", {"grid.linestyle": "--"})


# ── Data loading ───────────────────────────────────────────────────────────

def load_records(directory: Path) -> list[dict]:
    records = []
    for f in sorted(directory.glob("run_*.json")):
        try:
            with open(f) as fh:
                records.append(json.load(fh))
        except Exception as exc:
            print(f"  [WARN] skipping {f.name}: {exc}")
    print(f"Loaded {len(records)} run records.")
    return records


# ── Extraction ─────────────────────────────────────────────────────────────

def extract_latency_data(records: list[dict]) -> dict:
    """
    Returns {mode: {
        "wall_s":        [float],   total wall latency in seconds
        "decomposer_s":  [float],
        "node_sum_s":    [float],   sum of per-node elapsed_s
        "overhead_s":    [float],   wall - decomposer - node_sum
        "input_tok":     [int],
        "output_tok":    [int],
        "total_tok":     [int],
        "judge_score":   [float],
    }}
    """
    data: dict = {m: defaultdict(list) for m in MODES}

    for rec in records:
        mode = rec.get("ablation_mode", "")
        if mode not in MODES:
            continue
        et = rec.get("engine_telemetry", {})
        if not et or rec.get("failed"):
            continue

        # Wall latency (prefer engine's own measurement)
        wall_s = et.get("latency_seconds") or rec.get("wall_latency_ms", 0) / 1000

        decomp_s = (et.get("decomposer_latency_ms") or 0) / 1000
        node_sum_s = sum(
            n.get("elapsed_s", 0) for n in et.get("got_dag_nodes", [])
        )
        overhead_s = max(wall_s - decomp_s - node_sum_s, 0.0)

        tok = et.get("orchestrator_tokens") or {}
        in_tok  = tok.get("input",  0) or 0
        out_tok = tok.get("output", 0) or 0

        judge = et.get("final_judge_score")
        if judge is None:
            continue

        bucket = data[mode]
        bucket["wall_s"].append(wall_s)
        bucket["decomposer_s"].append(decomp_s)
        bucket["node_sum_s"].append(node_sum_s)
        bucket["overhead_s"].append(overhead_s)
        bucket["input_tok"].append(in_tok)
        bucket["output_tok"].append(out_tok)
        bucket["total_tok"].append(in_tok + out_tok)
        bucket["judge_score"].append(float(judge))

    return data


# ── Figure 1: Total Latency Violin ─────────────────────────────────────────

def fig_latency_violin(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    positions = np.arange(len(MODES))
    all_vals = [data[m]["wall_s"] for m in MODES]

    # violin
    parts = ax.violinplot(
        [v if v else [0] for v in all_vals],
        positions=positions,
        widths=0.55,
        showmedians=False,
        showextrema=False,
    )
    for i, (pc, m) in enumerate(zip(parts["bodies"], MODES)):
        s = STYLES[m]
        pc.set_facecolor(s["face"])
        pc.set_edgecolor(s["edge"])
        pc.set_linewidth(2)
        pc.set_alpha(0.85)

    # box overlay
    bp = ax.boxplot(
        [v if v else [0] for v in all_vals],
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
    )
    for patch, m in zip(bp["boxes"], MODES):
        s = STYLES[m]
        patch.set_facecolor(s["face"])
        patch.set_hatch(s["hatch"])
        patch.set_edgecolor(s["edge"])

    # annotate medians
    for i, m in enumerate(MODES):
        vals = data[m]["wall_s"]
        if vals:
            med = np.median(vals)
            ax.text(i, med + 1.5, f"{med:.1f}s",
                    ha="center", va="bottom", fontsize=20,
                    fontweight="bold", color=STYLES[m]["edge"])

    ax.set_xticks(positions)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Wall Latency (s)")
    ax.set_title("End-to-End Latency per Ablation Mode")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 2: Latency Breakdown Stacked Bar ────────────────────────────────

def fig_latency_breakdown(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    x = np.arange(len(MODES))
    w = 0.55

    components = [
        ("decomposer_s", "#7fc97f", "//",  "Decomposer"),
        ("node_sum_s",   "#beaed4", "..",  "Agent Nodes"),
        ("overhead_s",   "#fdc086", "xx",  "Overhead"),
    ]

    bottoms = np.zeros(len(MODES))
    for key, face, hatch, label in components:
        vals = np.array([np.mean(data[m][key]) if data[m][key] else 0 for m in MODES])
        bars = ax.bar(x, vals, width=w, bottom=bottoms,
                      color=face, hatch=hatch, edgecolor="black",
                      linewidth=1.2, label=label)
        for bar, v, bot in zip(bars, vals, bottoms):
            if v > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + v / 2, f"{v:.1f}s",
                        ha="center", va="center",
                        fontsize=17, fontweight="bold", color="black")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Mean Latency (s)")
    ax.set_title("Latency Component Breakdown")
    ax.legend(loc="upper right", framealpha=0.85)

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 3: Orchestrator Token Consumption ───────────────────────────────

def fig_token_cost(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    x = np.arange(len(MODES))
    w = 0.35
    token_types = [
        ("input_tok",  "#6baed6", "//",  "Input Tokens"),
        ("output_tok", "#fd8d3c", "..",  "Output Tokens"),
    ]

    for shift, (key, face, hatch, label) in zip([-0.18, 0.18], token_types):
        vals = np.array([np.mean(data[m][key]) if data[m][key] else 0 for m in MODES])
        bars = ax.bar(x + shift, vals, width=w,
                      color=face, hatch=hatch, edgecolor="black",
                      linewidth=1.2, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30, f"{v:.0f}",
                    ha="center", va="bottom",
                    fontsize=18, fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Mean Tokens (Orchestrator LLM)")
    ax.set_title("Orchestrator Token Consumption")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 4: Quality-vs-Latency Scatter ──────────────────────────────────

def fig_efficiency_scatter(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    handles = []
    for m in MODES:
        s = STYLES[m]
        xs = data[m]["wall_s"]
        ys = data[m]["judge_score"]
        if not xs:
            continue
        sc = ax.scatter(xs, ys,
                        c=s["face"], edgecolors=s["edge"],
                        linewidths=1.8, s=90, alpha=0.82,
                        marker="o", zorder=3)
        # mean crosshairs
        mx, my = np.mean(xs), np.mean(ys)
        ax.scatter([mx], [my], c=s["edge"], s=220, marker="D",
                   zorder=5, edgecolors="black", linewidths=1.5)
        handles.append(mpatches.Patch(facecolor=s["face"],
                                      edgecolor=s["edge"],
                                      label=s["label"]))

    ax.set_xlabel("Wall Latency (s)")
    ax.set_ylabel("Final Judge Score (0–10)")
    ax.set_title("Quality vs. Latency Trade-off")
    ax.legend(handles=handles, loc="lower right", framealpha=0.85)
    ax.set_ylim(0, 10.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Combined 2×2 Panel ─────────────────────────────────────────────────────

def fig_panel(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(
        "Latency & Token Trade-off Analysis",
        fontsize=36, fontweight="bold", y=1.01,
    )

    fig_latency_violin(data,    ax=axes[0, 0])
    fig_latency_breakdown(data, ax=axes[0, 1])
    fig_token_cost(data,        ax=axes[1, 0])
    fig_efficiency_scatter(data,ax=axes[1, 1])

    labels = ["(A)", "(B)", "(C)", "(D)"]
    for ax, lbl in zip(axes.flat, labels):
        ax.text(-0.08, 1.04, lbl, transform=ax.transAxes,
                fontsize=28, fontweight="bold", va="top")

    fig.tight_layout()
    return fig


# ── Save helper ────────────────────────────────────────────────────────────

def save(fig: plt.Figure, stem: str):
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved → {path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=== 03_latency_token.py ===")
    records = load_records(TELEMETRY_DIR)
    data    = extract_latency_data(records)

    for m in MODES:
        n = len(data[m]["wall_s"])
        med = np.median(data[m]["wall_s"]) if data[m]["wall_s"] else 0
        print(f"  {m.upper():8s}: n={n:3d}, median_latency={med:.1f}s, "
              f"mean_tokens={np.mean(data[m]['total_tok']):.0f}" if data[m]["total_tok"] else f"  {m.upper():8s}: n={n:3d}")

    print("\nPlotting figures...")
    save(fig_latency_violin(data),    "03_latency_total")
    save(fig_latency_breakdown(data), "03_latency_breakdown")
    save(fig_token_cost(data),        "03_token_cost")
    save(fig_efficiency_scatter(data),"03_efficiency")
    save(fig_panel(data),             "03_panel")
    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
