"""
06_agent_level.py — Per-Agent Performance Analysis
====================================================
Drills into individual agent behaviour: quality scores from the GoT
per-node judge, execution latency, paper retrieval counts (RAG),
and strategy selection distribution.

Metrics extracted from got_dag_nodes per run
---------------------------------------------
  quality_score   — GoT per-node judge score (0–1)
  elapsed_s       — agent wall time
  failure_mode    — judge failure category (or null)
  paper_count     — RAG only: papers retrieved
  rag_strategy    — RAG only: cot | vanilla | hybrid
  agent           — rag | code | cloud | scholar | aggregate

Figures produced
----------------
  06_node_quality.{png,pdf}      — per-agent quality score violin per mode
  06_node_latency.{png,pdf}      — per-agent execution latency box per mode
  06_failure_heatmap.{png,pdf}   — failure mode heatmap (agent × mode)
  06_rag_detail.{png,pdf}        — RAG: paper count dist + strategy breakdown
  06_panel.{png,pdf}             — 2×2 combined panel

Run:   python scripts/eval/06_agent_level.py
Input: raw_telemetry/run_*.json
Out:   figures/06_*.{png,pdf}
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
AGENTS = ["rag", "scholar", "code", "cloud"]  # exclude aggregate

STYLES = {
    "none":     {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx",  "label": "NONE"},
    "raw":      {"face": "#ffcc99", "edge": "#cc3300", "hatch": "//",  "label": "RAW"},
    "adaptive": {"face": "#99ffcc", "edge": "#006633", "hatch": "..",  "label": "ADAPTIVE (Ours)"},
}

AGENT_COLORS = {
    "rag":     "#6baed6",
    "scholar": "#9e9ac8",
    "code":    "#fd8d3c",
    "cloud":   "#74c476",
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

def extract_node_data(records: list[dict]) -> dict:
    """
    Returns {mode: {agent: {
        "quality":       [float 0-1],
        "elapsed_s":     [float],
        "failure_mode":  [str|None],
        "paper_count":   [int],        RAG only
        "rag_strategy":  [str],        RAG only
    }}}
    """
    data: dict = {
        m: {a: defaultdict(list) for a in AGENTS}
        for m in MODES
    }

    for rec in records:
        mode = rec.get("ablation_mode", "")
        if mode not in MODES:
            continue
        et = rec.get("engine_telemetry", {})
        if not et or rec.get("failed"):
            continue

        for node in et.get("got_dag_nodes", []):
            agent = (node.get("agent") or "").lower()
            if agent not in AGENTS:
                continue
            b = data[mode][agent]
            q = node.get("quality_score")
            if q is not None:
                b["quality"].append(float(q))
            b["elapsed_s"].append(node.get("elapsed_s", 0))
            b["failure_mode"].append(node.get("failure_mode"))  # may be None

            if agent == "rag":
                pc = node.get("paper_count")
                if pc is not None:
                    b["paper_count"].append(int(pc))
                strat = node.get("rag_strategy") or node.get("strategy") or "unknown"
                b["rag_strategy"].append(str(strat))

    return data


# ── Figure 1: Per-Agent Quality Violin per Mode ───────────────────────────

def fig_node_quality(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = None

    n_agents = len(AGENTS)
    group_w  = 1.0
    mode_w   = group_w / len(MODES)
    offsets  = np.linspace(-group_w / 2 + mode_w / 2, group_w / 2 - mode_w / 2, len(MODES))
    x_base   = np.arange(n_agents) * (group_w + 0.5)

    handles = []
    for m_i, m in enumerate(MODES):
        s = STYLES[m]
        positions = x_base + offsets[m_i]
        vals = [data[m][a]["quality"] for a in AGENTS]

        parts = ax.violinplot(
            [v if v else [0] for v in vals],
            positions=positions,
            widths=mode_w * 0.8,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(s["face"])
            pc.set_edgecolor(s["edge"])
            pc.set_linewidth(1.5)
            pc.set_alpha(0.8)

        bp = ax.boxplot(
            [v if v else [0] for v in vals],
            positions=positions,
            widths=mode_w * 0.3,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(s["face"])
            patch.set_hatch(s["hatch"])
            patch.set_edgecolor(s["edge"])

        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"],
            hatch=s["hatch"], label=s["label"]))

    ax.set_xticks(x_base)
    ax.set_xticklabels([a.title() for a in AGENTS])
    ax.set_ylabel("GoT Node Quality Score (0–1)")
    ax.set_title("Per-Agent Quality by Ablation Mode")
    ax.set_ylim(0, 1.12)
    ax.legend(handles=handles, loc="lower right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 2: Per-Agent Latency Box per Mode ──────────────────────────────

def fig_node_latency(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = None

    n_agents = len(AGENTS)
    group_w  = 1.0
    mode_w   = group_w / len(MODES)
    offsets  = np.linspace(-group_w / 2 + mode_w / 2, group_w / 2 - mode_w / 2, len(MODES))
    x_base   = np.arange(n_agents) * (group_w + 0.5)

    handles = []
    for m_i, m in enumerate(MODES):
        s = STYLES[m]
        positions = x_base + offsets[m_i]
        vals = [data[m][a]["elapsed_s"] for a in AGENTS]

        bp = ax.boxplot(
            [v if v else [0] for v in vals],
            positions=positions,
            widths=mode_w * 0.75,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(s["face"])
            patch.set_hatch(s["hatch"])
            patch.set_edgecolor(s["edge"])
        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"],
            hatch=s["hatch"], label=s["label"]))

    ax.set_xticks(x_base)
    ax.set_xticklabels([a.title() for a in AGENTS])
    ax.set_ylabel("Node Execution Time (s)")
    ax.set_title("Per-Agent Latency by Ablation Mode")
    ax.legend(handles=handles, loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 3: Failure Mode Heatmap ────────────────────────────────────────

def fig_failure_heatmap(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = None

    # Collect all non-null failure modes
    all_modes_set: set[str] = set()
    for m in MODES:
        for a in AGENTS:
            all_modes_set.update(
                fm for fm in data[m][a]["failure_mode"] if fm
            )

    if not all_modes_set:
        ax.text(0.5, 0.5, "No failure modes recorded",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=24, color="gray")
        ax.set_title("Failure Mode Heatmap")
        if standalone:
            fig.tight_layout()
        return fig

    failure_modes = sorted(all_modes_set)
    rows  = [f"{a.title()}\n({m.upper()})" for m in MODES for a in AGENTS]
    matrix = np.zeros((len(rows), len(failure_modes)))

    for r_i, (m, a) in enumerate([(m, a) for m in MODES for a in AGENTS]):
        total = len(data[m][a]["failure_mode"])
        if total == 0:
            continue
        for c_i, fm in enumerate(failure_modes):
            count = sum(1 for x in data[m][a]["failure_mode"] if x == fm)
            matrix[r_i, c_i] = count / total * 100

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(failure_modes)))
    ax.set_xticklabels(
        [f.replace("_", " ").title() for f in failure_modes],
        rotation=35, ha="right", fontsize=18)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=16)
    ax.set_title("Failure Mode Frequency Heatmap (% of nodes)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% Nodes with Failure", fontsize=20)

    for i in range(len(rows)):
        for j in range(len(failure_modes)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="black" if v < 60 else "white")

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 4: RAG Detail ──────────────────────────────────────────────────

def fig_rag_detail(data: dict, ax=None) -> plt.Figure | None:
    """Two sub-panels: paper count bars + strategy pie."""
    standalone = ax is None
    if standalone:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax = ax1   # will draw both from here
    else:
        fig = None
        # In panel mode we only have one ax — draw paper count bar there.
        ax1 = ax
        ax2 = None

    # ── paper count bars ──
    x  = np.arange(len(MODES))
    w  = 0.55
    means = [
        np.mean(data[m]["rag"]["paper_count"]) if data[m]["rag"]["paper_count"] else 0
        for m in MODES
    ]
    stds = [
        np.std(data[m]["rag"]["paper_count"]) if data[m]["rag"]["paper_count"] else 0
        for m in MODES
    ]
    for i, (m, mean, std) in enumerate(zip(MODES, means, stds)):
        s = STYLES[m]
        ax1.bar(i, mean, width=w, yerr=std, capsize=6,
                color=s["face"], hatch=s["hatch"],
                edgecolor=s["edge"], linewidth=1.8)
        ax1.text(i, mean + std + 0.15, f"{mean:.1f}",
                 ha="center", va="bottom",
                 fontsize=20, fontweight="bold", color=s["edge"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax1.set_ylabel("Mean Papers Retrieved (RAG)")
    ax1.set_title("RAG Paper Count by Mode")
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # ── strategy pie (overall) ──
    if ax2 is not None:
        all_strats: list[str] = []
        for m in MODES:
            all_strats.extend(data[m]["rag"]["rag_strategy"])
        unique = sorted(set(all_strats))
        counts = [all_strats.count(s) for s in unique]
        colors = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#d9d9d9"]
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=[s.upper() for s in unique],
            autopct="%1.0f%%",
            colors=colors[: len(unique)],
            startangle=90,
            textprops={"fontsize": 20, "fontweight": "bold"},
        )
        for at in autotexts:
            at.set_fontsize(18)
        ax2.set_title("RAG Strategy Distribution (All Modes)")

    if standalone:
        fig.suptitle("RAG Agent Detail", fontsize=32, fontweight="bold")
        fig.tight_layout()
    return fig


# ── Combined 2×2 Panel ─────────────────────────────────────────────────────

def fig_panel(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(26, 16))
    fig.suptitle(
        "Per-Agent Performance Analysis",
        fontsize=36, fontweight="bold", y=1.01,
    )

    fig_node_quality(data,   ax=axes[0, 0])
    fig_node_latency(data,   ax=axes[0, 1])
    fig_failure_heatmap(data,ax=axes[1, 0])
    fig_rag_detail(data,     ax=axes[1, 1])

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
    print("=== 06_agent_level.py ===")
    records = load_records(TELEMETRY_DIR)
    data    = extract_node_data(records)

    for a in AGENTS:
        for m in MODES:
            quals = data[m][a]["quality"]
            lats  = data[m][a]["elapsed_s"]
            n     = len(quals)
            if n:
                print(f"  {m.upper():8s} | {a:7s}: n={n:3d}, "
                      f"med_quality={np.median(quals):.3f}, "
                      f"med_lat={np.median(lats):.2f}s")

    print("\nPlotting figures...")
    save(fig_node_quality(data),    "06_node_quality")
    save(fig_node_latency(data),    "06_node_latency")
    save(fig_failure_heatmap(data), "06_failure_heatmap")
    save(fig_rag_detail(data),      "06_rag_detail")
    save(fig_panel(data),           "06_panel")
    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
