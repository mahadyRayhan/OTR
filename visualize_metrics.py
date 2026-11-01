"""
visualize_metrics.py
====================
Phase 3 of the evaluation framework.
Reads all CSVs and JSON from metrics/ and produces publication-ready figures.

Output: visualizations/
  L1_system_overview.png
  L2_agent_performance.png
  L3a_cus_inverted_u_curve.png
  L3b_communication_comparison.png
  L3c_edge_heatmap.png
  L4_knowledge_utilization.png
  L5_selfhealing.png
  L6a_pareto_frontier.png
  L6b_cost_latency.png
  L7_novel_metrics.png
  DASHBOARD_all_layers.png

Dependencies: pip install matplotlib seaborn pandas numpy scipy
"""

import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_DIR = Path("metrics")
VIZ_DIR     = Path("visualizations")
VIZ_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "none":      "#6C757D",
    "raw":       "#E07B39",
    "adaptive":  "#2E86AB",
    "highlight": "#D62828",
    "good":      "#2DC653",
    "warn":      "#F4A261",
    "neutral":   "#A8DADC",
}
MODE_ORDER  = ["none", "raw", "adaptive"]
MODE_LABELS = {"none": "NONE", "raw": "RAW", "adaptive": "ADAPTIVE (CUGCR)"}
MODE_COLORS = [PALETTE["none"], PALETTE["raw"], PALETTE["adaptive"]]

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(name: str) -> pd.DataFrame:
    p = METRICS_DIR / name
    if not p.exists():
        print(f"  [missing] {name} — returning empty DataFrame")
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "ablation_mode" in df.columns:
        df["ablation_mode"] = df["ablation_mode"].str.lower().str.strip()
    return df


def load_json(name: str) -> dict:
    p = METRICS_DIR / name
    if not p.exists():
        print(f"  [missing] {name}")
        return {}
    with open(p) as f:
        return json.load(f)


def save(fig: plt.Figure, name: str):
    path = VIZ_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✔  {name}")


def _mode_color(mode: str) -> str:
    return PALETTE.get(mode.lower(), "#999999")


def _add_value_labels(ax, bars, fmt="{:.2f}", fontsize=7, offset=0.05):
    for bar in bars:
        h = bar.get_height()
        if math.isnan(h) or h < 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize, color="#333333",
        )


def _no_data(ax, label="No data available"):
    ax.text(0.5, 0.5, label, ha="center", va="center",
            transform=ax.transAxes, color="#999999", fontsize=9)
    ax.set_axis_off()


# ═══════════════════════════════════════════════════════════════════════════════
# L1 — System-level overview
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L1():
    df  = load_csv("L1_system.csv")
    met = load_json("all_metrics.json")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Layer 1 — System-Level Evaluation", fontsize=13, fontweight="bold", y=1.01)

    abl = df[df["protocol"] == "A_ablation"] if not df.empty else pd.DataFrame()

    # 1a GTS violin per mode
    ax = axes[0, 0]
    if not abl.empty:
        data  = [abl[abl["ablation_mode"] == m]["gts"].dropna().values for m in MODE_ORDER]
        parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)
        for pc, c in zip(parts["bodies"], MODE_COLORS):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        parts["cmedians"].set_color("#222222")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        ax.set_ylabel("Global Task Score (0–10)")
        ax.set_title("GTS Distribution by Mode")
        ax.set_ylim(0, 11)
    else:
        _no_data(ax, "No ablation data")

    # 1b GTS mean ± std bar chart
    ax = axes[0, 1]
    if not abl.empty:
        means = [abl[abl["ablation_mode"] == m]["gts"].mean() for m in MODE_ORDER]
        stds  = [abl[abl["ablation_mode"] == m]["gts"].std()  for m in MODE_ORDER]
        x    = np.arange(len(MODE_ORDER))
        bars = ax.bar(x, means, color=MODE_COLORS, alpha=0.85, width=0.55,
                      yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        ax.set_ylabel("Mean GTS ± Std")
        ax.set_title("Mean GTS per Mode")
        ax.set_ylim(0, 11)
        _add_value_labels(ax, bars, fmt="{:.2f}", offset=0.1)
    else:
        _no_data(ax)

    # 1c Task Success Rate
    ax = axes[0, 2]
    if not abl.empty:
        tsrs = [abl[abl["ablation_mode"] == m]["tsr"].mean() for m in MODE_ORDER]
        x    = np.arange(len(MODE_ORDER))
        bars = ax.bar(x, tsrs, color=MODE_COLORS, alpha=0.85, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        ax.set_ylabel("Task Success Rate")
        ax.set_title("Task Success Rate per Mode")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        _add_value_labels(ax, bars, fmt="{:.1%}", offset=0.01)
    else:
        _no_data(ax)

    # 1d Pairwise win rate pie
    ax = axes[1, 0]
    wr = met.get("L1_system", {}).get("win_rates", {})
    if wr:
        sizes  = [wr.get("adaptive_win_rate", 0), wr.get("raw_win_rate", 0), wr.get("none_win_rate", 0)]
        labels = [f"ADAPTIVE ({sizes[0]:.0%})", f"RAW ({sizes[1]:.0%})", f"NONE ({sizes[2]:.0%})"]
        ax.pie(sizes, labels=labels, colors=MODE_COLORS, autopct="%1.0f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title(f"Pairwise Win Rate\n(n={wr.get('total_comparisons', 0)} queries)")
    else:
        _no_data(ax, "No win rate data")

    # 1e Consistency stdev
    ax = axes[1, 1]
    cons_df = df[df["protocol"] == "B_consistency"] if not df.empty else pd.DataFrame()
    if not cons_df.empty:
        stats = cons_df.groupby("query_id")["gts"].agg(["mean", "std"]).reset_index()
        ax.bar(stats["query_id"], stats["std"], color=PALETTE["adaptive"], alpha=0.8)
        ax.axhline(0.5, color=PALETTE["highlight"], linestyle="--", linewidth=1.2,
                   label="Instability threshold (σ=0.5)")
        ax.set_ylabel("GTS Std Dev (lower = more stable)")
        ax.set_title("Answer Consistency (3 reps per query)")
        ax.legend()
    else:
        _no_data(ax, "No consistency run data")

    # 1f Depth budget utilisation
    ax = axes[1, 2]
    if not abl.empty and "depth_utilization_pct" in abl.columns:
        data  = [abl[abl["ablation_mode"] == m]["depth_utilization_pct"].dropna().values for m in MODE_ORDER]
        parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)
        for pc, c in zip(parts["bodies"], MODE_COLORS):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        ax.set_ylabel("Depth Budget Used (%)")
        ax.set_title("Depth Budget Utilisation")
    else:
        _no_data(ax, "No depth data")

    fig.tight_layout()
    save(fig, "L1_system_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L2 — Agent-level performance
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L2():
    df  = load_csv("L2_agent.csv")
    met = load_json("all_metrics.json")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Layer 2 — Agent-Level Evaluation", fontsize=13, fontweight="bold", y=1.01)

    if df.empty:
        for ax in axes.flat:
            _no_data(ax)
        save(fig, "L2_agent_performance.png")
        return

    first_pass   = df[(df["is_refine"] == 0) & (df["nqs"] >= 0)]
    agents       = sorted(df["agent_type"].unique())
    agent_colors = sns.color_palette("Set2", len(agents))
    a_color      = {a: agent_colors[i] for i, a in enumerate(agents)}

    # 2a NQS boxplot per agent (first pass)
    ax = axes[0, 0]
    data = [first_pass[first_pass["agent_type"] == a]["nqs"].dropna().values for a in agents]
    bp   = ax.boxplot(data, patch_artist=True)
    for patch, a in zip(bp["boxes"], agents):
        patch.set_facecolor(a_color[a]); patch.set_alpha(0.8)
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_ylabel("Node Quality Score (0–10)")
    ax.set_title("NQS Distribution per Agent\n(first-pass only)")
    ax.axhline(7.5, color=PALETTE["highlight"], linestyle="--", linewidth=1, label="Quality gate (7.5)")
    ax.legend(fontsize=7)

    # 2b First-pass success rate
    ax = axes[0, 1]
    fps  = {a: ((first_pass[first_pass["agent_type"] == a]["nqs"] >= 7.5).mean()
                if len(first_pass[first_pass["agent_type"] == a]) > 0 else 0)
            for a in agents}
    bars = ax.bar(agents, [fps[a] for a in agents],
                  color=[a_color[a] for a in agents], alpha=0.85)
    ax.set_ylabel("First-Pass Success Rate")
    ax.set_title("First-Pass Success Rate per Agent\n(NQS ≥ 7.5 on attempt #1)")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=30, ha="right")
    _add_value_labels(ax, bars, fmt="{:.0%}", offset=0.01)

    # 2c Hallucination + DB refusal grouped bar
    ax = axes[0, 2]
    hall = [df[df["agent_type"] == a]["ai_warning"].mean() for a in agents]
    ref  = [df[df["agent_type"] == a]["db_refusal"].mean() for a in agents]
    x    = np.arange(len(agents)); w = 0.35
    ax.bar(x - w/2, hall, width=w, label="AI Extrapolation", color=PALETTE["warn"],      alpha=0.85)
    ax.bar(x + w/2, ref,  width=w, label="DB Refusal",       color=PALETTE["highlight"], alpha=0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Hallucination & DB Refusal Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()

    # 2d Placeholder leak rate
    ax = axes[1, 0]
    leak = [df[df["agent_type"] == a]["placeholder_leak"].mean() for a in agents]
    bars = ax.bar(agents, leak, color=[a_color[a] for a in agents], alpha=0.85)
    ax.set_ylabel("Placeholder Leak Rate")
    ax.set_title("Placeholder Leak Rate per Agent")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=30, ha="right")
    _add_value_labels(ax, bars, fmt="{:.0%}", offset=0.005)

    # 2e NQS: first attempt vs after REFINE
    ax = axes[1, 1]
    all_valid = df[df["nqs"] >= 0]
    fp_nqs    = all_valid[all_valid["is_refine"] == 0]["nqs"]
    rf_nqs    = all_valid[all_valid["is_refine"] == 1]["nqs"]
    ax.hist(fp_nqs, bins=10, range=(0, 10), alpha=0.7, color=PALETTE["raw"],  label="1st attempt")
    ax.hist(rf_nqs, bins=10, range=(0, 10), alpha=0.7, color=PALETTE["good"], label="After REFINE")
    ax.set_xlabel("NQS"); ax.set_ylabel("Count")
    ax.set_title("NQS: First Attempt vs After REFINE")
    ax.legend()

    # 2f Token usage per agent
    ax = axes[1, 2]
    tok  = [df[df["agent_type"] == a]["tokens_used"].dropna().values for a in agents]
    bp   = ax.boxplot(tok, patch_artist=True)
    for patch, a in zip(bp["boxes"], agents):
        patch.set_facecolor(a_color[a]); patch.set_alpha(0.8)
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_ylabel("Tokens Used")
    ax.set_title("Token Usage per Agent")

    fig.tight_layout()
    save(fig, "L2_agent_performance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L3a — CUS inverted U-curve (the central empirical result)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L3a_u_curve():
    df = load_csv("L3_edges.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Layer 3 — The 'Sweet Spot' Hypothesis\n"
        "Context Utilisation (CUS) vs Node Quality Score (NQS)",
        fontsize=13, fontweight="bold"
    )

    for col_idx, mode in enumerate(["raw", "adaptive"]):
        ax  = axes[col_idx]
        sub = df[(df["ablation_mode"] == mode) &
                 (df["cus"] >= 0) & (df["cus"] <= 1) &
                 (df["run_gts"] >= 0)] if not df.empty else pd.DataFrame()

        if sub.empty:
            _no_data(ax, f"No data for {mode.upper()}"); continue

        x = sub["cus"].values
        y = sub["run_gts"].values

        ax.scatter(x * 100, y, alpha=0.45, s=25, color=_mode_color(mode),
                   linewidths=0, zorder=3)

        if len(x) >= 4:
            try:
                z  = np.polyfit(x * 100, y, 2)
                xs = np.linspace(0, 100, 300)
                ax.plot(xs, np.clip(np.poly1d(z)(xs), 0, 10),
                        color="#222222", linewidth=2, label="Poly fit (deg 2)", zorder=4)
            except Exception:
                pass

        ax.axvspan(20, 40, alpha=0.12, color=PALETTE["good"], label="Sweet spot (20–40%)")
        ax.axvline(20, color=PALETTE["good"], linestyle=":", linewidth=1)
        ax.axvline(40, color=PALETTE["good"], linestyle=":", linewidth=1)

        for xpos, lbl in [(8, "Context\nBlindness"), (62, "Over-\nConstrained")]:
            ax.text(xpos, 1.5, lbl, ha="center", fontsize=7.5,
                    color=PALETTE["highlight"], style="italic")

        ax.set_xlim(0, 100); ax.set_ylim(0, 10.5)
        ax.set_xlabel("Context Utilisation Score — CUS (%)")
        ax.set_ylabel("Node Quality Score — NQS (0–10)")
        ax.set_title(f"Mode: {MODE_LABELS[mode]}\n(n={len(sub)} edges)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, "L3a_cus_inverted_u_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L3b — Communication metrics comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L3b_comm():
    df  = load_csv("L3_edges.csv")
    met = load_json("all_metrics.json")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Layer 3 — Agent-to-Agent Communication Metrics",
                 fontsize=13, fontweight="bold", y=1.01)

    def _metric_bar(ax, metric_col, title, ylabel, fmt="{:.2f}"):
        if df.empty or metric_col not in df.columns:
            _no_data(ax, f"No {metric_col}"); return
        vals = [df[(df["ablation_mode"] == m) & (df[metric_col] >= 0)][metric_col].mean()
                for m in MODE_ORDER]
        bars = ax.bar(range(len(MODE_ORDER)), vals, color=MODE_COLORS, alpha=0.85, width=0.55)
        ax.set_xticks(range(len(MODE_ORDER)))
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title)
        _add_value_labels(ax, bars, fmt=fmt,
                          offset=max(vals) * 0.03 if any(v > 0 for v in vals) else 0.01)

    _metric_bar(axes[0, 0], "crs",               "Context Relevance Score (CRS)",    "Mean CRS")
    _metric_bar(axes[0, 1], "cus",               "Context Utilisation Score (CUS)",  "Mean CUS")
    _metric_bar(axes[0, 2], "dsr",               "Dependency Satisfaction Rate (DSR)","Mean DSR")
    _metric_bar(axes[1, 0], "compression_ratio", "Token Compression Ratio",           "RAW / Injected", fmt="{:.1f}×")

    # ΔQ per edge pair
    ax = axes[1, 1]
    dq_rows = met.get("L3_communication", {}).get("delta_q_by_edge", [])
    if dq_rows:
        dq_df = pd.DataFrame(dq_rows)
        top10 = dq_df.nlargest(10, "delta_q")
        colors = [PALETTE["good"] if v >= 0 else PALETTE["highlight"] for v in top10["delta_q"]]
        ax.barh(top10["edge_pair"], top10["delta_q"], color=colors, alpha=0.85)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("ΔQ (Information Gain)")
        ax.set_title("Information Gain per Edge (ΔQ)\nAdaptive vs None")
    else:
        _no_data(ax, "No ΔQ data")

    # Protected atom preservation pie
    ax = axes[1, 2]
    if not df.empty and "prot_preservation_rate" in df.columns:
        prot = df[(df["prot_preservation_rate"] >= 0) & (df["ablation_mode"] == "adaptive")]
        if not prot.empty:
            perfect = (prot["prot_preservation_rate"] == 1.0).mean()
            partial = ((prot["prot_preservation_rate"] < 1.0) &
                       (prot["prot_preservation_rate"] > 0)).mean()
            lost    = (prot["prot_preservation_rate"] == 0.0).mean()
            ax.pie([perfect, partial, lost],
                   labels=[f"Perfect (100%)\n{perfect:.0%}",
                           f"Partial\n{partial:.0%}",
                           f"Lost\n{lost:.0%}"],
                   colors=[PALETTE["good"], PALETTE["warn"], PALETTE["highlight"]],
                   startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            ax.set_title("Protected Atom\nPreservation Rate (ADAPTIVE)")
        else:
            _no_data(ax, "No protected atom data")
    else:
        _no_data(ax, "No protected atom data")

    fig.tight_layout()
    save(fig, "L3b_communication_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L3c — Edge heatmap (CRS and CUS per agent pair)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L3c_heatmap():
    df = load_csv("L3_edges.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Layer 3 — Edge-Level Communication Heatmaps",
                 fontsize=13, fontweight="bold")

    for col_idx, (metric, label) in enumerate([("crs", "CRS"), ("cus", "CUS")]):
        ax  = axes[col_idx]
        sub = df[(df["ablation_mode"] == "adaptive") & (df[metric] >= 0)] \
              if not df.empty else pd.DataFrame()

        if sub.empty:
            _no_data(ax, f"No {metric} data"); continue

        pivot = sub.pivot_table(values=metric, index="from_agent",
                                columns="to_agent", aggfunc="mean")
        if pivot.empty:
            _no_data(ax, "Insufficient edge variety"); continue

        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Mean {label} per Agent Pair\n(ADAPTIVE mode)")
        ax.set_xlabel("To Agent"); ax.set_ylabel("From Agent")

    fig.tight_layout()
    save(fig, "L3c_edge_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L4 — Knowledge utilization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L4():
    df  = load_csv("L4_knowledge.csv")
    met = load_json("all_metrics.json")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Layer 4 — Knowledge Utilisation & Provenance",
                 fontsize=13, fontweight="bold")

    # 4a Stacked bar: DB hit / miss / extrapolation per agent
    ax = axes[0]
    if not df.empty:
        agents = sorted(df["agent_type"].unique())
        hit    = [df[df["agent_type"] == a]["db_hit"].mean()          for a in agents]
        miss   = [df[df["agent_type"] == a]["db_miss"].mean()         for a in agents]
        extrap = [df[df["agent_type"] == a]["ai_extrapolated"].mean()  for a in agents]
        x      = np.arange(len(agents))
        ax.bar(x, hit,    color=PALETTE["good"],      alpha=0.85, label="DB Hit")
        ax.bar(x, miss,   bottom=hit,
               color=PALETTE["highlight"], alpha=0.75, label="DB Refusal")
        ax.bar(x, extrap, bottom=[h + m for h, m in zip(hit, miss)],
               color=PALETTE["warn"],      alpha=0.75, label="AI Extrapolation")
        ax.set_xticks(x); ax.set_xticklabels(agents, rotation=30, ha="right")
        ax.set_ylabel("Rate"); ax.set_title("Knowledge Source per Agent")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=7)
    else:
        _no_data(ax)

    # 4b NQS by knowledge source
    ax = axes[1]
    if not df.empty and "nqs" in df.columns:
        hit_nqs  = df[(df["db_hit"]          == 1) & (df["nqs"] >= 0)]["nqs"]
        miss_nqs = df[(df["db_miss"]         == 1) & (df["nqs"] >= 0)]["nqs"]
        ext_nqs  = df[(df["ai_extrapolated"] == 1) & (df["nqs"] >= 0)]["nqs"]
        groups   = {"DB Hit": hit_nqs, "DB Refusal": miss_nqs, "AI Extrapolation": ext_nqs}
        bp_data  = [v.values for v in groups.values() if len(v) > 0]
        bp_lbls  = [k for k, v in groups.items() if len(v) > 0]
        colors   = [PALETTE["good"], PALETTE["highlight"], PALETTE["warn"]][:len(bp_lbls)]
        bp = ax.boxplot(bp_data, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        ax.set_xticklabels(bp_lbls, rotation=20, ha="right")
        ax.set_ylabel("NQS"); ax.set_title("Output Quality by Knowledge Source")
        ax.axhline(7.5, color="#555555", linestyle="--", linewidth=1)
    else:
        _no_data(ax)

    # 4c Overall knowledge coverage donut
    ax = axes[2]
    L4 = met.get("L4_knowledge", {})
    if L4:
        cov   = L4.get("knowledge_coverage_rate",  0)
        ref   = L4.get("db_refusal_rate",           0)
        ext   = L4.get("ai_extrapolation_rate",     0)
        total = cov + ref + ext
        if total > 0:
            ax.pie([cov / total, ref / total, ext / total],
                   labels=[f"DB Hit\n{cov:.0%}", f"DB Refusal\n{ref:.0%}", f"AI Extrap.\n{ext:.0%}"],
                   colors=[PALETTE["good"], PALETTE["highlight"], PALETTE["warn"]],
                   startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
            centre = plt.Circle((0, 0), 0.55, fc="white")
            ax.add_artist(centre)
            gs = L4.get("mean_grounding_score")
            if gs:
                ax.text(0, 0, f"Grounding\n{gs:.1f}/10",
                        ha="center", va="center", fontsize=9, fontweight="bold")
        ax.set_title("Knowledge Coverage Summary")
    else:
        _no_data(ax)

    fig.tight_layout()
    save(fig, "L4_knowledge_utilization.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L5 — Self-healing evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L5():
    df  = load_csv("L5_selfhealing.csv")
    met = load_json("all_metrics.json")
    L5  = met.get("L5_selfhealing", {})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Layer 5 — Self-Healing & Robustness",
                 fontsize=13, fontweight="bold", y=1.01)

    agent_rows = df[df["agent_type"] != "SYSTEM"] if not df.empty else pd.DataFrame()

    # 5a Recovery funnel
    ax = axes[0, 0]
    if L5:
        tf  = L5.get("total_node_failures", 0)
        rr  = L5.get("recovery_rate", 0)
        rec = int(tf * rr)
        ax.bar(["Failed", "Recovered\n(NQS≥8)", "Unrecovered"],
               [tf, rec, tf - rec],
               color=[PALETTE["warn"], PALETTE["good"], PALETTE["highlight"]],
               alpha=0.85)
        ax.set_ylabel("Node Count")
        ax.set_title(f"Recovery Funnel\n(Recovery Rate: {rr:.0%})")
    else:
        _no_data(ax)

    # 5b Recovery depth distribution
    ax = axes[0, 1]
    if not agent_rows.empty:
        depths = agent_rows["recovery_depth"].value_counts().sort_index()
        ax.bar(depths.index.astype(str), depths.values,
               color=PALETTE["adaptive"], alpha=0.85)
        ax.set_xlabel("REFINE iterations"); ax.set_ylabel("Count")
        ax.set_title("Recovery Depth Distribution\n(0 = no retry needed)")
    else:
        _no_data(ax)

    # 5c NQS before vs after recovery
    ax = axes[0, 2]
    if not agent_rows.empty:
        rec = agent_rows[(agent_rows["recovery_attempted"] == 1) &
                         (agent_rows["original_nqs"]  >= 0) &
                         (agent_rows["recovered_nqs"] >= 0)]
        if not rec.empty:
            ax.scatter(rec["original_nqs"], rec["recovered_nqs"],
                       alpha=0.6, s=40, color=PALETTE["adaptive"])
            ax.plot([0, 10], [0,  10], "k--", linewidth=0.8, label="No change")
            ax.plot([0, 10], [8,   8], "--",  color=PALETTE["good"],
                    linewidth=0.8, label="Recovery threshold (8)")
            ax.set_xlabel("Original NQS"); ax.set_ylabel("Recovered NQS")
            ax.set_title("NQS Before vs After REFINE")
            ax.legend(fontsize=7)
        else:
            _no_data(ax, "No recovery pairs")
    else:
        _no_data(ax)

    # 5d Reflexion quality histogram
    ax = axes[1, 0]
    rq = L5.get("reflexion_quality_scores", [])
    if rq:
        rq_df      = pd.DataFrame(rq)
        scores_yes = rq_df[rq_df["recovered"] == 1]["rq_score"]
        scores_no  = rq_df[rq_df["recovered"] == 0]["rq_score"]
        ax.hist(scores_yes, bins=8, range=(0, 10), alpha=0.75,
                color=PALETTE["good"],      label="Eventually recovered")
        ax.hist(scores_no,  bins=8, range=(0, 10), alpha=0.75,
                color=PALETTE["highlight"], label="Still failed")
        ax.set_xlabel("Reflexion Quality Score (LLM Judge)")
        ax.set_ylabel("Count")
        ax.set_title("Reflexion Block Quality vs Recovery Outcome")
        ax.legend()
    else:
        _no_data(ax, "No reflexion quality data")

    # 5e False positive REFINE rate per agent
    ax = axes[1, 1]
    if not agent_rows.empty:
        agents   = sorted(agent_rows["agent_type"].unique())
        fp_rates = [agent_rows[agent_rows["agent_type"] == a]["false_positive_refine"].mean()
                    for a in agents]
        bars = ax.bar(agents, fp_rates, color=PALETTE["warn"], alpha=0.85)
        ax.set_ylabel("False Positive REFINE Rate")
        ax.set_title("Over-triggering Rate per Agent\n(REFINE when NQS already ≥ 7.5)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=30, ha="right")
        _add_value_labels(ax, bars, fmt="{:.0%}", offset=0.005)
    else:
        _no_data(ax)

    # 5f Cascading failures + Diagnostician
    ax = axes[1, 2]
    sys_rows = df[df["agent_type"] == "SYSTEM"] if not df.empty else pd.DataFrame()
    if not sys_rows.empty and "cascading_failure_count" in sys_rows.columns:
        cascade_total  = sys_rows["cascading_failure_count"].sum()
        diag_triggered = sys_rows["recovery_attempted"].sum()
        diag_success   = (sys_rows["recovery_success"] == 1).sum()
        labels = ["Cascading\nFailures", "Diagnostician\nTriggered", "Diagnostician\nSuccess"]
        vals   = [cascade_total, diag_triggered, diag_success]
        colors = [PALETTE["highlight"], PALETTE["warn"], PALETTE["good"]]
        bars   = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_ylabel("Count")
        ax.set_title("System-Level Failure & Recovery\n(Global Diagnostician)")
        _add_value_labels(ax, bars, fmt="{:.0f}", offset=0.1)
    else:
        _no_data(ax)

    fig.tight_layout()
    save(fig, "L5_selfhealing.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L6a — Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L6a_pareto():
    df = load_csv("L6_pareto.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Layer 6 — Pareto Frontier: Quality vs Token Cost",
                 fontsize=13, fontweight="bold")

    if df.empty:
        _no_data(ax); save(fig, "L6a_pareto_frontier.png"); return

    for mode in MODE_ORDER:
        sub = df[df["ablation_mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(sub["total_tokens"], sub["gts"],
                   alpha=0.55, s=35, color=_mode_color(mode),
                   label=MODE_LABELS[mode], zorder=3)
        if len(sub) >= 3:
            try:
                from scipy.spatial import ConvexHull
                pts  = sub[["total_tokens", "gts"]].values
                hull = ConvexHull(pts)
                hp   = np.append(hull.vertices, hull.vertices[0])
                ax.plot(pts[hp, 0], pts[hp, 1], linestyle="--",
                        linewidth=1, alpha=0.4, color=_mode_color(mode))
            except Exception:
                pass

    ax.set_xlabel("Total Tokens (cost proxy)")
    ax.set_ylabel("Global Task Score (GTS)")
    ax.set_title("Pareto Frontier — ADAPTIVE should dominate upper-left\n"
                 "(high quality, low token cost)")
    ax.legend()
    ax.annotate("← Ideal region\n(high quality,\nlow cost)",
                xy=(0.15, 0.85), xycoords="axes fraction", fontsize=8, color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    fig.tight_layout()
    save(fig, "L6a_pareto_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L6b — Cost and latency breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L6b_cost():
    df  = load_csv("L6_cost_latency.csv")
    met = load_json("all_metrics.json")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Layer 6 — Cost & Latency Trade-Off",
                 fontsize=13, fontweight="bold", y=1.01)

    abl = df[df["protocol"] == "A_ablation"] if not df.empty else pd.DataFrame()

    def mode_means(col):
        return [abl[abl["ablation_mode"] == m][col].mean()
                if not abl.empty and col in abl.columns else 0
                for m in MODE_ORDER]

    # 6b-1 Token cost per mode
    ax    = axes[0, 0]
    means = mode_means("total_tokens")
    bars  = ax.bar(range(3), means, color=MODE_COLORS, alpha=0.85, width=0.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Mean Total Tokens"); ax.set_title("Token Cost per Mode")
    _add_value_labels(ax, bars, fmt="{:.0f}", offset=max(means) * 0.01 if any(means) else 1)

    # 6b-2 Latency per mode
    ax    = axes[0, 1]
    means = mode_means("latency_ms")
    bars  = ax.bar(range(3), means, color=MODE_COLORS, alpha=0.85, width=0.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Mean Latency (ms)"); ax.set_title("End-to-End Latency per Mode")
    _add_value_labels(ax, bars, fmt="{:.0f}ms", offset=50)

    # 6b-3 CUGCR overhead tax histogram
    ax = axes[0, 2]
    if not df.empty and "cugcr_overhead_pct" in df.columns:
        cugcr_df = df[(df["cugcr_triggered"] == 1) & (df["cugcr_overhead_pct"] > 0)]
        if not cugcr_df.empty:
            mean_oh = cugcr_df["cugcr_overhead_pct"].mean()
            ax.hist(cugcr_df["cugcr_overhead_pct"], bins=10,
                    color=PALETTE["adaptive"], alpha=0.8)
            ax.axvline(mean_oh, color=PALETTE["highlight"], linestyle="--",
                       label=f"Mean: {mean_oh:.1f}%")
            ax.set_xlabel("CUGCR Overhead (% of total latency)")
            ax.set_ylabel("Count")
            ax.set_title("CUGCR Compression Overhead Tax")
            ax.legend()
        else:
            _no_data(ax, "CUGCR not triggered in data")
    else:
        _no_data(ax)

    # 6b-4 Communication Efficiency CE distribution
    ax = axes[1, 0]
    L3      = met.get("L3_communication", {})
    dq_rows = L3.get("delta_q_by_edge", [])
    if dq_rows:
        ce_vals = [r["communication_efficiency_ce"]
                   for r in dq_rows
                   if r.get("communication_efficiency_ce") is not None]
        if ce_vals:
            ax.hist(ce_vals, bins=12, color=PALETTE["adaptive"], alpha=0.8)
            ax.axvline(0, color=PALETTE["highlight"], linestyle="--", linewidth=1)
            ax.set_xlabel("CE = ΔQ / 1k tokens")
            ax.set_ylabel("Count")
            ax.set_title("Communication Efficiency (CE)\nDistribution across Edges")
        else:
            _no_data(ax, "No CE values computed")
    else:
        _no_data(ax, "No CE data")

    # 6b-5 Depth budget utilisation distribution
    ax = axes[1, 1]
    if not abl.empty and "depth_utilization_pct" in abl.columns:
        for mode in MODE_ORDER:
            sub = abl[abl["ablation_mode"] == mode]["depth_utilization_pct"].dropna()
            if not sub.empty:
                ax.hist(sub, bins=8, alpha=0.5, label=MODE_LABELS[mode],
                        color=_mode_color(mode))
        ax.set_xlabel("Depth Budget Used (%)")
        ax.set_ylabel("Count")
        ax.set_title("Depth Budget Utilisation Distribution")
        ax.legend()
    else:
        _no_data(ax)

    # 6b-6 GTS vs tokens scatter
    ax = axes[1, 2]
    if not abl.empty:
        for mode in MODE_ORDER:
            sub = abl[(abl["ablation_mode"] == mode) & (abl["gts"] >= 0)]
            ax.scatter(sub["total_tokens"], sub["gts"], alpha=0.55, s=30,
                       color=_mode_color(mode), label=MODE_LABELS[mode])
        ax.set_xlabel("Total Tokens"); ax.set_ylabel("GTS")
        ax.set_title("GTS vs Cost (Ablation Runs)")
        ax.legend(fontsize=7)
    else:
        _no_data(ax)

    fig.tight_layout()
    save(fig, "L6b_cost_latency.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L7 — Novel metrics
# ═══════════════════════════════════════════════════════════════════════════════

def plot_L7():
    df  = load_csv("L7_novel.csv")
    met = load_json("all_metrics.json")
    L7  = met.get("L7_novel", {})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Layer 7 — Novel System-Specific Metrics",
                 fontsize=13, fontweight="bold", y=1.01)

    # 7a Router determinism pie
    ax = axes[0, 0]
    det_df    = df[df["metric"] == "router_determinism"] if not df.empty else pd.DataFrame()
    det_score = L7.get("router_determinism_score")
    if not det_df.empty:
        unique = det_df["value"].nunique()
        total  = len(det_df)
        same   = total - (unique - 1)
        diff   = unique - 1
        if unique > 1:
            ax.pie([max(same, 0), max(diff, 0)],
                   labels=[f"Identical ({same}/{total})", f"Different ({diff})"],
                   colors=[PALETTE["good"], PALETTE["highlight"]],
                   wedgeprops={"edgecolor": "white", "linewidth": 2}, startangle=90)
        else:
            ax.pie([1], labels=[f"All identical ({total}/{total})"],
                   colors=[PALETTE["good"]],
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
        suffix = f"\n(Score: {det_score:.3f})" if det_score else ""
        ax.set_title(f"Router Determinism{suffix}")
    elif det_score is not None:
        ax.text(0.5, 0.5, f"Determinism Score\n{det_score:.3f}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=18, fontweight="bold",
                color=PALETTE["good"] if det_score > 0.9 else PALETTE["warn"])
        ax.set_title("Router Determinism Score")
        ax.set_axis_off()
    else:
        _no_data(ax, "No Protocol D data")

    # 7b Query complexity scaling
    ax   = axes[0, 1]
    comp = L7.get("complexity_scaling", {})
    if comp:
        groups = list(comp.keys())
        means  = [comp[g]["mean_gts"] for g in groups]
        ns     = [comp[g]["n"] for g in groups]
        colors = sns.color_palette("Blues_d", len(groups))
        bars   = ax.bar(groups, means, color=colors, alpha=0.85)
        ax.set_ylabel("Mean GTS"); ax.set_title("GTS by Query Complexity Group")
        ax.set_ylim(0, 10.5)
        ax.set_xticklabels(groups, rotation=20, ha="right")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"n={n}", ha="center", fontsize=7)
    else:
        _no_data(ax, "No complexity scaling data")

    # 7c Profile sensitivity
    ax   = axes[0, 2]
    prof = L7.get("profile_sensitivity", {}).get("gts_by_profile", {})
    if prof:
        profiles = [p for p in prof if prof[p] >= 0]
        gtss     = [prof[p] for p in profiles]
        colors   = [PALETTE["adaptive"], PALETTE["raw"], PALETTE["warn"]][:len(profiles)]
        bars     = ax.bar(profiles, gtss, color=colors, alpha=0.85)
        ax.set_ylabel("GTS"); ax.set_title("GTS by User Profile\n(Same query, different profiles)")
        ax.set_ylim(0, 10.5)
        _add_value_labels(ax, bars, fmt="{:.2f}", offset=0.15)
        gts_range = L7.get("profile_sensitivity", {}).get("gts_range", 0)
        ax.text(0.98, 0.97, f"Range: {gts_range:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    else:
        _no_data(ax, "No profile sensitivity data")

    # 7d Adversarial failure floor
    ax      = axes[1, 0]
    adv_df  = df[df["metric"] == "adversarial_robustness"] if not df.empty else pd.DataFrame()
    adv_sum = L7.get("adversarial_failure_floor", {})
    if not adv_df.empty:
        ax.bar(range(len(adv_df)), adv_df["numeric"].values,
               color=[PALETTE["highlight"] if v < 4 else
                      PALETTE["warn"]      if v < 7 else
                      PALETTE["good"]
                      for v in adv_df["numeric"].values],
               alpha=0.85)
        ax.axhline(4, color=PALETTE["highlight"], linestyle="--",
                   linewidth=1, label="Catastrophic threshold (4)")
        ax.axhline(7, color=PALETTE["warn"],      linestyle="--",
                   linewidth=1, label="Acceptable threshold (7)")
        ax.set_xlabel("Adversarial Query"); ax.set_ylabel("GTS")
        ax.set_ylim(0, 10.5)
        ax.set_title(f"Adversarial Robustness\n(Mean: {adv_sum.get('mean_gts', '?')})")
        ax.legend(fontsize=7)
    elif adv_sum:
        ax.text(0.5, 0.5,
                f"Mean GTS: {adv_sum.get('mean_gts', '?')}\n"
                f"Min: {adv_sum.get('min_gts', '?')}  Max: {adv_sum.get('max_gts', '?')}\n"
                f"n = {adv_sum.get('n', '?')}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Adversarial Failure Floor"); ax.set_axis_off()
    else:
        _no_data(ax, "No adversarial data")

    # 7e Context boundary gateway decisions
    ax       = axes[1, 1]
    bound_df = df[df["metric"] == "boundary_behavior"] if not df.empty else pd.DataFrame()
    if not bound_df.empty:
        decisions = bound_df["value"].value_counts()
        bars = ax.bar(decisions.index, decisions.values,
                      color=[PALETTE["adaptive"] if "cugcr" in str(k) else PALETTE["raw"]
                             for k in decisions.index],
                      alpha=0.85)
        ax.set_ylabel("Count")
        ax.set_title("Gateway Decisions on Boundary Queries\n(Protocol F)")
        _add_value_labels(ax, bars, fmt="{:.0f}", offset=0.05)
    else:
        _no_data(ax, "No boundary data (Protocol F)")

    # 7f Novel metrics scorecard
    ax = axes[1, 2]
    ax.set_axis_off()
    items = []
    if det_score is not None:
        items.append(("Router Determinism",    f"{det_score:.3f}", det_score > 0.9))
    adv_mean = adv_sum.get("mean_gts")
    if adv_mean is not None:
        items.append(("Adversarial Mean GTS", f"{adv_mean:.2f}", float(adv_mean) >= 5))
    prof_range = L7.get("profile_sensitivity", {}).get("gts_range")
    if prof_range is not None:
        items.append(("Profile GTS Range",    f"±{prof_range:.2f}", float(prof_range) < 2))
    if comp:
        pf = comp.get("perfect_hit",      {}).get("mean_gts")
        ma = comp.get("missing_algorithm", {}).get("mean_gts")
        if pf and ma:
            items.append(("Complexity Degradation",
                           f"{pf:.1f} → {ma:.1f}",
                           float(pf) - float(ma) < 3))

    if items:
        ax.set_title("Novel Metrics Scorecard", fontsize=11, fontweight="bold", pad=10)
        y = 0.88
        for label, value, is_ok in items:
            icon  = "✔" if is_ok else "✘"
            color = PALETTE["good"] if is_ok else PALETTE["highlight"]
            ax.text(0.05, y, f"{icon}  {label}:", transform=ax.transAxes,
                    fontsize=9, color=color, fontweight="bold")
            ax.text(0.72, y, value, transform=ax.transAxes, fontsize=9)
            y -= 0.18
    else:
        ax.text(0.5, 0.5, "Run all protocols\nto populate scorecard",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#999999")
        ax.set_title("Novel Metrics Scorecard")

    fig.tight_layout()
    save(fig, "L7_novel_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — single-page overview for paper / slides
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dashboard():
    df_sys  = load_csv("L1_system.csv")
    df_ag   = load_csv("L2_agent.csv")
    df_edge = load_csv("L3_edges.csv")
    df_sh   = load_csv("L5_selfhealing.csv")
    df_par  = load_csv("L6_pareto.csv")
    df_nov  = load_csv("L7_novel.csv")
    met     = load_json("all_metrics.json")

    fig = plt.figure(figsize=(22, 17))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)
    fig.suptitle("System Evaluation Dashboard — All 7 Layers",
                 fontsize=16, fontweight="bold", y=1.005)

    abl = df_sys[df_sys["protocol"] == "A_ablation"] if not df_sys.empty else pd.DataFrame()

    # ── Row 0 ─────────────────────────────────────────────────────────────────

    # [0,0] GTS mean ± std
    ax = fig.add_subplot(gs[0, 0])
    if not abl.empty:
        means = [abl[abl["ablation_mode"] == m]["gts"].mean() for m in MODE_ORDER]
        stds  = [abl[abl["ablation_mode"] == m]["gts"].std()  for m in MODE_ORDER]
        bars  = ax.bar(["NONE","RAW","ADAPTIVE"], means, color=MODE_COLORS,
                       alpha=0.85, yerr=stds, capsize=4)
        ax.set_ylim(0, 11); ax.set_ylabel("GTS")
        _add_value_labels(ax, bars, fmt="{:.1f}", offset=0.2)
    else:
        _no_data(ax)
    ax.set_title("L1 — GTS per Mode", fontweight="bold")

    # [0,1] Win rate pie
    ax = fig.add_subplot(gs[0, 1])
    wr = met.get("L1_system", {}).get("win_rates", {})
    if wr:
        sizes = [wr.get("adaptive_win_rate",0),
                 wr.get("raw_win_rate",     0),
                 wr.get("none_win_rate",    0)]
        ax.pie(sizes, labels=["ADAPTIVE","RAW","NONE"],
               colors=MODE_COLORS, autopct="%1.0f%%", startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":1.5})
    else:
        _no_data(ax)
    ax.set_title("L1 — Pairwise Win Rate", fontweight="bold")

    # [0,2] Agent NQS violin
    ax = fig.add_subplot(gs[0, 2])
    if not df_ag.empty:
        fp     = df_ag[(df_ag["is_refine"] == 0) & (df_ag["nqs"] >= 0)]
        agents = sorted(fp["agent_type"].unique())
        data   = [fp[fp["agent_type"] == a]["nqs"].values for a in agents]
        colors = sns.color_palette("Set2", len(agents))
        parts  = ax.violinplot(data, showmedians=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(agents) + 1))
        ax.set_xticklabels(agents, rotation=20, ha="right")
        ax.axhline(7.5, color=PALETTE["highlight"], linestyle="--", linewidth=0.8)
        ax.set_ylabel("NQS")
    else:
        _no_data(ax)
    ax.set_title("L2 — Agent NQS", fontweight="bold")

    # [0,3] Sweet spot mini U-curve
    ax = fig.add_subplot(gs[0, 3])
    sub = df_edge[(df_edge["ablation_mode"] == "adaptive") &
                  (df_edge["cus"] >= 0) & (df_edge["run_gts"] >= 0)] \
          if not df_edge.empty else pd.DataFrame()
    if not sub.empty:
        ax.scatter(sub["cus"] * 100, sub["run_gts"],
                   alpha=0.4, s=15, color=PALETTE["adaptive"])
        if len(sub) >= 4:
            try:
                z  = np.polyfit(sub["cus"].values * 100, sub["run_gts"].values, 2)
                xs = np.linspace(0, 100, 200)
                ax.plot(xs, np.clip(np.poly1d(z)(xs), 0, 10), "k-", linewidth=1.5)
            except Exception:
                pass
        ax.axvspan(20, 40, alpha=0.15, color=PALETTE["good"])
        ax.set_xlabel("CUS (%)"); ax.set_ylabel("NQS")
    else:
        _no_data(ax)
    ax.set_title("L3 — Sweet Spot Curve ★", fontweight="bold")

    # ── Row 1 ─────────────────────────────────────────────────────────────────

    # [1,0] DSR comparison
    ax = fig.add_subplot(gs[1, 0])
    if not df_edge.empty and "dsr" in df_edge.columns:
        dsrs = [df_edge[(df_edge["ablation_mode"] == m) & (df_edge["dsr"] >= 0)]["dsr"].mean()
                for m in MODE_ORDER]
        bars = ax.bar(["NONE","RAW","ADAPTIVE"], dsrs, color=MODE_COLORS, alpha=0.85)
        ax.set_ylabel("DSR"); ax.set_ylim(0, 1.1)
        _add_value_labels(ax, bars, fmt="{:.2f}", offset=0.02)
    else:
        _no_data(ax)
    ax.set_title("L3 — Dep. Satisfaction Rate", fontweight="bold")

    # [1,1] Knowledge coverage donut
    ax = fig.add_subplot(gs[1, 1])
    L4 = met.get("L4_knowledge", {})
    if L4:
        cov   = L4.get("knowledge_coverage_rate",  0)
        ref   = L4.get("db_refusal_rate",           0)
        ext   = L4.get("ai_extrapolation_rate",     0)
        total = cov + ref + ext
        if total > 0:
            ax.pie([cov/total, ref/total, ext/total],
                   labels=["DB Hit","Refusal","AI Extrap."],
                   colors=[PALETTE["good"], PALETTE["highlight"], PALETTE["warn"]],
                   startangle=90,
                   wedgeprops={"edgecolor":"white","linewidth":1.5})
            ax.add_artist(plt.Circle((0,0), 0.55, fc="white"))
        gs_val = L4.get("mean_grounding_score")
        if gs_val:
            ax.text(0, 0, f"Grounding\n{gs_val:.1f}/10",
                    ha="center", va="center", fontsize=8, fontweight="bold")
    else:
        _no_data(ax)
    ax.set_title("L4 — Knowledge Coverage", fontweight="bold")

    # [1,2] Recovery funnel
    ax = fig.add_subplot(gs[1, 2])
    L5 = met.get("L5_selfhealing", {})
    if L5:
        tf  = L5.get("total_node_failures", 0)
        rr  = L5.get("recovery_rate", 0)
        rec = int(tf * rr)
        ax.bar(["Failed","Recovered","Unrecovered"],
               [tf, rec, tf - rec],
               color=[PALETTE["warn"], PALETTE["good"], PALETTE["highlight"]],
               alpha=0.85)
        ax.set_ylabel("Nodes")
    else:
        _no_data(ax)
    ax.set_title(f"L5 — Recovery ({L5.get('recovery_rate', 0):.0%})", fontweight="bold")

    # [1,3] Reflexion quality
    ax = fig.add_subplot(gs[1, 3])
    rq = L5.get("reflexion_quality_scores", [])
    if rq:
        rq_df = pd.DataFrame(rq)
        ax.hist(rq_df["rq_score"], bins=8, range=(0, 10),
                color=PALETTE["adaptive"], alpha=0.8)
        ax.axvline(rq_df["rq_score"].mean(), color=PALETTE["highlight"],
                   linestyle="--", linewidth=1.2,
                   label=f"Mean: {rq_df['rq_score'].mean():.1f}")
        ax.set_xlabel("Reflexion Quality"); ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    else:
        _no_data(ax)
    ax.set_title("L5 — Reflexion Quality", fontweight="bold")

    # ── Row 2 ─────────────────────────────────────────────────────────────────

    # [2,0:2] Pareto frontier (wide)
    ax = fig.add_subplot(gs[2, 0:2])
    if not df_par.empty:
        for mode in MODE_ORDER:
            sub = df_par[(df_par["ablation_mode"] == mode) & (df_par["gts"] >= 0)]
            if not sub.empty:
                ax.scatter(sub["total_tokens"], sub["gts"],
                           alpha=0.6, s=30, color=_mode_color(mode),
                           label=MODE_LABELS[mode])
        ax.set_xlabel("Total Tokens"); ax.set_ylabel("GTS")
        ax.legend(fontsize=7)
    else:
        _no_data(ax)
    ax.set_title("L6 — Pareto Frontier (Quality vs Token Cost)", fontweight="bold")

    # [2,2] Adversarial robustness
    ax = fig.add_subplot(gs[2, 2])
    adv_df  = df_nov[df_nov["metric"] == "adversarial_robustness"] \
              if not df_nov.empty else pd.DataFrame()
    adv_sum = met.get("L7_novel", {}).get("adversarial_failure_floor", {})
    if not adv_df.empty:
        ax.bar(range(len(adv_df)), adv_df["numeric"].values,
               color=[PALETTE["highlight"] if v < 5 else PALETTE["good"]
                      for v in adv_df["numeric"].values],
               alpha=0.85)
        ax.axhline(4, color=PALETTE["highlight"], linestyle="--", linewidth=0.8)
        ax.set_xlabel("Query"); ax.set_ylabel("GTS"); ax.set_ylim(0, 11)
    else:
        _no_data(ax)
    ax.set_title("L7 — Adversarial Robustness", fontweight="bold")

    # [2,3] Complexity scaling
    ax   = fig.add_subplot(gs[2, 3])
    comp = met.get("L7_novel", {}).get("complexity_scaling", {})
    if comp:
        groups = list(comp.keys())
        means  = [comp[g]["mean_gts"] for g in groups]
        ax.bar(groups, means,
               color=sns.color_palette("Blues_d", len(groups)), alpha=0.85)
        ax.set_ylim(0, 10.5); ax.set_ylabel("GTS")
        ax.set_xticklabels(groups, rotation=20, ha="right")
    else:
        _no_data(ax)
    ax.set_title("L7 — Complexity Scaling", fontweight="bold")

    save(fig, "DASHBOARD_all_layers.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       FULL SYSTEM EVALUATION — VISUALIZATIONS           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Reading from : {METRICS_DIR.resolve()}")
    print(f"Output to    : {VIZ_DIR.resolve()}\n")

    print("Generating figures...")
    plot_L1()
    plot_L2()
    plot_L3a_u_curve()
    plot_L3b_comm()
    plot_L3c_heatmap()
    plot_L4()
    plot_L5()
    plot_L6a_pareto()
    plot_L6b_cost()
    plot_L7()
    plot_dashboard()

    print(f"\n✅ All figures saved to {VIZ_DIR.resolve()}")
    print("\nFigure index:")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()