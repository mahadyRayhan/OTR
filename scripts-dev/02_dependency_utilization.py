"""
02_dependency_utilization.py — Context Dependency Utilization Analysis
=======================================================================
Analyses how much of the upstream context injected between agents was
actually USED to produce the final answer, broken down by ablation mode
and receiving agent type.

Metrics extracted from communication_edges per run
---------------------------------------------------
  context_utilized_pct  — CUS: % of injected context that appears in the
                          agent's output (token overlap score × 100).
                          Measures how much of what was sent was actually used.
  context_relevance_pct — CRS: cosine similarity between the injected context
                          and the downstream query (× 100).
                          Measures how pertinent the sent content was.
  context_tokens        — Raw token count of injected context per edge.
                          Measures the volume of inter-agent communication.
  dependency_accuracy   — Bool: True = no unresolved placeholder in output.
                          Measures whether the agent properly consumed context.

Key insight
-----------
  NONE mode:     0 tokens injected, 0% CUS, 0% CRS  (no context passing)
  RAW  mode:     high token volume, moderate CUS (noise dilutes usage)
  ADAPTIVE mode: lower token volume (CUGCR compressed), higher CUS
                 (only relevant atoms injected → more efficiently used)

Run:   python scripts/eval/02_dependency_utilization.py
Input: raw_telemetry/run_*.json
Out:   figures/02_dep_utilization_cus.{png,pdf}
       figures/02_dep_utilization_crs.{png,pdf}
       figures/02_dep_utilization_tokens.{png,pdf}
       figures/02_dep_utilization_accuracy.{png,pdf}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
TELEMETRY_DIR = ROOT / "raw_telemetry"
FIG_DIR       = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
MODES       = ["none", "raw", "adaptive"]
# Receiving agents that participate in context injection (not root nodes)
AGENTS      = ["rag", "code", "cloud", "scholar"]
AGENT_LABELS = {
    "rag":     "RAG",
    "code":    "Code",
    "cloud":   "Cloud",
    "scholar": "Scholar",
    "overall": "Overall",
}
ALL_AGENTS  = AGENTS + ["overall"]

STYLES = {
    "none":     {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx",  "label": "NONE"},
    "raw":      {"face": "#ffcc99", "edge": "#cc3300", "hatch": "//",  "label": "RAW"},
    "adaptive": {"face": "#99ffcc", "edge": "#006633", "hatch": "..",  "label": "ADAPTIVE (Ours)"},
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         20,
    "font.weight":       "bold",
    "axes.labelsize":    34,
    "axes.titlesize":    38,
    "xtick.labelsize":   28,
    "ytick.labelsize":   28,
    "axes.labelweight":  "bold",
    "axes.titleweight":  "bold",
    "legend.fontsize":   26,
    "hatch.linewidth":   2.0,
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

def extract_edge_metrics(records: list[dict]) -> dict:
    """
    Returns {mode: {agent: {metric: [values]}}}
    Metrics: cus, crs, tokens, accuracy
    """
    data: dict = {
        m: {a: {"cus": [], "crs": [], "tokens": [], "accuracy": []}
            for a in ALL_AGENTS}
        for m in MODES
    }

    for rec in records:
        mode = rec.get("ablation_mode", "")
        if mode not in MODES:
            continue

        tel   = rec.get("engine_telemetry", {})
        edges = tel.get("communication_edges", [])

        # Also pull from got_dag_nodes which has the enriched per-node fields
        nodes = {n["node_id"]: n for n in tel.get("got_dag_nodes", [])}

        # Primary source: communication_edges (has context text and raw metrics)
        edge_seen = set()
        for edge in edges:
            agent  = edge.get("to_agent", "").lower()
            to_id  = edge.get("to_id", "")
            if agent not in AGENTS:
                continue
            edge_seen.add(to_id)

            cus      = (edge.get("utilization_score") or 0) * 100
            crs      = (edge.get("context_relevance")  or 0) * 100
            tokens   = edge.get("context_token_count", 0) or 0
            accurate = 1 if edge.get("dependency_accuracy", True) else 0

            for bucket in (agent, "overall"):
                data[mode][bucket]["cus"].append(cus)
                data[mode][bucket]["crs"].append(crs)
                data[mode][bucket]["tokens"].append(tokens)
                data[mode][bucket]["accuracy"].append(accurate)

        # Supplement with enriched dag node fields for any nodes not in edges
        for n_id, node in nodes.items():
            agent = node.get("agent", "").lower()
            op    = node.get("operator", "")
            if agent not in AGENTS or op == "aggregate":
                continue
            if n_id in edge_seen:
                continue  # already counted from edges above

            cus    = node.get("context_utilized_pct") or 0
            crs    = node.get("context_relevance_pct") or 0
            tokens = node.get("context_tokens", 0) or 0
            acc    = 1 if node.get("dependency_accuracy", True) else 0

            for bucket in (agent, "overall"):
                data[mode][bucket]["cus"].append(cus)
                data[mode][bucket]["crs"].append(crs)
                data[mode][bucket]["tokens"].append(tokens)
                data[mode][bucket]["accuracy"].append(acc)

    return data


def compute_stats(data: dict) -> dict:
    """
    Returns {mode: {agent: {metric: (mean, sem, n)}}}
    """
    stats: dict = {}
    for mode in MODES:
        stats[mode] = {}
        for agent in ALL_AGENTS:
            stats[mode][agent] = {}
            for metric in ("cus", "crs", "tokens", "accuracy"):
                vals = data[mode][agent].get(metric, [])
                if vals:
                    arr = np.array(vals, dtype=float)
                    stats[mode][agent][metric] = (
                        float(np.mean(arr)),
                        float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
                        len(arr),
                    )
                else:
                    stats[mode][agent][metric] = (np.nan, 0.0, 0)
    return stats


# ── Generic bar plotter ────────────────────────────────────────────────────

def _bar_chart(
    stats:     dict,
    metric:    str,
    ylabel:    str,
    title:     str,
    stem:      str,
    ylim:      tuple,
    y_fmt:     str,          # e.g. "{:.1f}%" or "{:.0f}"
    out_dir:   Path,
) -> None:
    n_groups  = len(ALL_AGENTS)
    bar_width = 0.25
    x         = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(20, 10))

    for i, mode in enumerate(MODES):
        s      = STYLES[mode]
        offset = (i - 1) * bar_width
        means  = [stats[mode][a][metric][0] for a in ALL_AGENTS]
        errs   = [stats[mode][a][metric][1] for a in ALL_AGENTS]

        bars = ax.bar(
            x + offset, means, bar_width,
            color=s["face"], edgecolor=s["edge"], linewidth=2,
            hatch=s["hatch"], label=s["label"],
            yerr=errs, capsize=6,
            error_kw={"elinewidth": 2, "ecolor": s["edge"], "capthick": 2},
            zorder=3,
        )

        for bar, mean in zip(bars, means):
            if np.isnan(mean) or mean == 0:
                continue
            pad = bar.get_height() * 0.04 if ylim[1] is None else ylim[1] * 0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad,
                y_fmt.format(mean),
                ha="center", va="bottom",
                color=s["edge"], fontsize=20, fontweight="bold",
            )

    # Separator before Overall
    ax.axvline(n_groups - 1 - 0.5, color="#888888",
               linestyle="--", linewidth=1.8, zorder=2)

    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_title(title, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in ALL_AGENTS], ha="center")
    ax.set_ylim(*ylim)
    sns.despine()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3, frameon=True,
        edgecolor="black", fancybox=False,
    )

    plt.tight_layout()
    _save(fig, out_dir, stem)


# ── Accuracy rate bar (special — y is 0–100%) ─────────────────────────────

def plot_accuracy(stats: dict, out_dir: Path) -> None:
    # Convert mean (0–1) to percentage
    pct_stats: dict = {}
    for mode in MODES:
        pct_stats[mode] = {}
        for agent in ALL_AGENTS:
            pct_stats[mode][agent] = {}
            m, s, n = stats[mode][agent]["accuracy"]
            pct_stats[mode][agent]["accuracy"] = (m * 100, s * 100, n)
            # carry other metrics unchanged
            for k in ("cus", "crs", "tokens"):
                pct_stats[mode][agent][k] = stats[mode][agent][k]

    _bar_chart(
        pct_stats, "accuracy",
        ylabel="Dependency Accuracy Rate (%)",
        title="Dependency Accuracy Rate by Ablation Mode\n"
              "(% of edges with no unresolved placeholder leaks)",
        stem="02_dep_utilization_accuracy",
        ylim=(0, 120),
        y_fmt="{:.0f}%",
        out_dir=out_dir,
    )


# ── Combined 4-panel summary ───────────────────────────────────────────────

def plot_combined(stats: dict, out_dir: Path) -> None:
    """
    2×2 panel figure: CUS | CRS | Tokens | Accuracy
    Compact summary for a single publication figure.
    """
    metrics = [
        ("cus",      "CUS — Context Utilized (%)",      (0, 110), "{:.0f}%"),
        ("crs",      "CRS — Context Relevance (%)",      (0, 110), "{:.0f}%"),
        ("tokens",   "Context Volume (tokens injected)", (0, None), "{:.0f}"),
        ("accuracy", "Dependency Accuracy (%)",          (0, 120), "{:.0f}%"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(30, 22))
    axes = axes.flatten()

    for ax, (metric, ylabel, ylim, fmt) in zip(axes, metrics):
        n_groups  = len(ALL_AGENTS)
        bar_width = 0.25
        x         = np.arange(n_groups)

        for i, mode in enumerate(MODES):
            s      = STYLES[mode]
            offset = (i - 1) * bar_width
            vals   = stats[mode]

            if metric == "accuracy":
                means = [vals[a]["accuracy"][0] * 100 for a in ALL_AGENTS]
                errs  = [vals[a]["accuracy"][1] * 100 for a in ALL_AGENTS]
            else:
                means = [vals[a][metric][0] for a in ALL_AGENTS]
                errs  = [vals[a][metric][1] for a in ALL_AGENTS]

            # Auto y-limit for tokens
            actual_ylim = ylim
            if metric == "tokens" and ylim[1] is None:
                max_val = max((v for v in means if not np.isnan(v)), default=1)
                actual_ylim = (0, max_val * 1.35)

            bars = ax.bar(
                x + offset, means, bar_width,
                color=s["face"], edgecolor=s["edge"], linewidth=2,
                hatch=s["hatch"], label=s["label"],
                yerr=errs, capsize=5,
                error_kw={"elinewidth": 1.8, "ecolor": s["edge"], "capthick": 2},
                zorder=3,
            )

            for bar, mean in zip(bars, means):
                if np.isnan(mean) or mean == 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + actual_ylim[1] * 0.015,
                    fmt.format(mean),
                    ha="center", va="bottom",
                    color=s["edge"], fontsize=18, fontweight="bold",
                )

        ax.axvline(n_groups - 1 - 0.5, color="#888888",
                   linestyle="--", linewidth=1.8, zorder=2)
        ax.set_ylabel(ylabel, labelpad=8, fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[a] for a in ALL_AGENTS],
                           ha="center", fontsize=24)
        if metric == "tokens" and ylim[1] is None:
            max_val = max(
                (stats[m][a]["tokens"][0] for m in MODES for a in ALL_AGENTS
                 if not np.isnan(stats[m][a]["tokens"][0])),
                default=1
            )
            ax.set_ylim(0, max_val * 1.35)
        else:
            ax.set_ylim(*ylim)
        sns.despine(ax=ax)

    # Shared legend at bottom
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       facecolor=STYLES[m]["face"],
                       edgecolor=STYLES[m]["edge"],
                       hatch=STYLES[m]["hatch"],
                       linewidth=2,
                       label=STYLES[m]["label"])
        for m in MODES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3, frameon=True,
        edgecolor="black", fancybox=False,
        fontsize=28,
    )

    fig.suptitle(
        "Context Dependency Utilization — How Upstream Agent Outputs\n"
        "Were Injected and Used Across Ablation Modes",
        fontsize=36, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    _save(fig, out_dir, "02_dep_utilization_combined")


# ── Summary table ──────────────────────────────────────────────────────────

def print_table(stats: dict) -> None:
    print("\n── Dependency Utilization Summary (mean ± SEM) ──")
    header = (
        f"{'Agent':<10}  "
        f"{'NONE CUS%':>12}  {'RAW CUS%':>12}  {'ADAP CUS%':>12}  "
        f"{'NONE CRS%':>12}  {'RAW CRS%':>12}  {'ADAP CRS%':>12}  "
        f"{'NONE tok':>10}  {'RAW tok':>10}  {'ADAP tok':>10}"
    )
    print(header)
    print("─" * len(header))

    def fmt(mode: str, agent: str, metric: str, scale: float = 1.0) -> str:
        m, s, _ = stats[mode][agent].get(metric, (None, 0, 0))
        if m is None or np.isnan(m):
            return "   N/A   "
        return f"{m*scale:6.1f}±{s*scale:.1f}"

    for agent in ALL_AGENTS:
        label = AGENT_LABELS[agent]
        print(
            f"{label:<10}  "
            f"{fmt('none',     agent, 'cus'):>12}  "
            f"{fmt('raw',      agent, 'cus'):>12}  "
            f"{fmt('adaptive', agent, 'cus'):>12}  "
            f"{fmt('none',     agent, 'crs'):>12}  "
            f"{fmt('raw',      agent, 'crs'):>12}  "
            f"{fmt('adaptive', agent, 'crs'):>12}  "
            f"{fmt('none',     agent, 'tokens'):>10}  "
            f"{fmt('raw',      agent, 'tokens'):>10}  "
            f"{fmt('adaptive', agent, 'tokens'):>10}"
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight",
                    format=ext if ext == "pdf" else None)
        print(f"Saved {path}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TELEMETRY_DIR.exists():
        print(f"[ERROR] Telemetry dir not found: {TELEMETRY_DIR}")
        sys.exit(1)

    records = load_records(TELEMETRY_DIR)
    if not records:
        print("[ERROR] No run_*.json files found.")
        sys.exit(1)

    edge_data = extract_edge_metrics(records)
    stats     = compute_stats(edge_data)
    print_table(stats)

    # Individual figures (one metric each — for selective inclusion in paper)
    _bar_chart(
        stats, "cus",
        ylabel="Context Utilization Score — CUS (%)",
        title="How Much Injected Context Was Used by Each Agent\n"
              "(CUS: token overlap between injected context and agent output)",
        stem="02_dep_utilization_cus",
        ylim=(0, 110),
        y_fmt="{:.1f}%",
        out_dir=FIG_DIR,
    )

    _bar_chart(
        stats, "crs",
        ylabel="Context Relevance Score — CRS (%)",
        title="How Relevant the Injected Context Was to Each Agent's Query\n"
              "(CRS: cosine similarity between upstream context and downstream query)",
        stem="02_dep_utilization_crs",
        ylim=(0, 110),
        y_fmt="{:.1f}%",
        out_dir=FIG_DIR,
    )

    _bar_chart(
        stats, "tokens",
        ylabel="Mean Context Tokens Injected per Edge",
        title="Volume of Context Passed Between Agents\n"
              "(ADAPTIVE compresses via CUGCR; RAW dumps everything; NONE passes nothing)",
        stem="02_dep_utilization_tokens",
        ylim=(0, None),
        y_fmt="{:.0f}",
        out_dir=FIG_DIR,
    )

    plot_accuracy(stats, FIG_DIR)

    # Combined 2×2 panel for the paper
    plot_combined(stats, FIG_DIR)

    print("\nDone. Figures saved to figures/")
