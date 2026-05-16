"""
04_self_healing.py — Self-Healing & First-Pass Success Analysis
================================================================
Analyses the Reflexion / global-retry mechanism and first-pass
success rates across ablation modes.

Metrics extracted per run
-------------------------
  first_pass_success  — bool: did the pipeline pass the judge on the first try?
  global_retries      — int: how many Reflexion retries were triggered?
  retry_events        — list: per-retry detail (failing_agent, score before/after)
  final_judge_score   — int 0-10: final score after all retries

Figures produced
----------------
  04_fpsr.{png,pdf}           — first-pass success rate per mode (bar)
  04_retry_dist.{png,pdf}     — retry count distribution (grouped histogram)
  04_score_delta.{png,pdf}    — score improvement per Reflexion event (scatter)
  04_agent_failures.{png,pdf} — which agents triggered retries (stacked bar)
  04_panel.{png,pdf}          — 2×2 combined panel

Run:   python scripts/eval/04_self_healing.py
Input: raw_telemetry/run_*.json
Out:   figures/04_*.{png,pdf}
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

def extract_healing_data(records: list[dict]) -> dict:
    """
    Returns {mode: {
        "fpsr":           [0|1],        1 = first-pass success
        "n_retries":      [int],
        "final_score":    [float],
        "retry_deltas":   [(before, after, agent)],
        "failing_agents": [str],        per retry event
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

        fps   = et.get("first_pass_success")
        retries = et.get("global_retries", 0) or 0
        score = et.get("final_judge_score")
        retry_events = et.get("retry_events") or []

        if fps is None or score is None:
            continue

        b = data[mode]
        b["fpsr"].append(1 if fps else 0)
        b["n_retries"].append(retries)
        b["final_score"].append(float(score))

        for ev in retry_events:
            bef = ev.get("score_before", 0) or 0
            aft = ev.get("score_after",  0) or 0
            agent = (ev.get("failing_agent") or "unknown").lower()
            b["retry_deltas"].append((bef, aft, agent))
            b["failing_agents"].append(agent)

    return data


# ── Figure 1: First-Pass Success Rate ─────────────────────────────────────

def fig_fpsr(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    x = np.arange(len(MODES))
    w = 0.55

    rates = []
    ns    = []
    for m in MODES:
        vals = data[m]["fpsr"]
        rates.append(np.mean(vals) * 100 if vals else 0)
        ns.append(len(vals))

    for i, (m, r, n) in enumerate(zip(MODES, rates, ns)):
        s = STYLES[m]
        bar = ax.bar(i, r, width=w,
                     color=s["face"], hatch=s["hatch"],
                     edgecolor=s["edge"], linewidth=2)
        ax.text(i, r + 1.5, f"{r:.1f}%\n(n={n})",
                ha="center", va="bottom",
                fontsize=20, fontweight="bold", color=s["edge"])

    ax.set_xticks(x)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylim(0, 115)
    ax.set_ylabel("First-Pass Success Rate (%)")
    ax.set_title("First-Pass Success Rate by Mode")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 2: Retry Count Distribution ────────────────────────────────────

def fig_retry_dist(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    max_retries = max(
        (max(data[m]["n_retries"]) if data[m]["n_retries"] else 0)
        for m in MODES
    )
    bins = np.arange(-0.5, max_retries + 1.5, 1)
    width = 0.25

    x_centers = np.arange(max_retries + 1)
    offsets = [-width, 0, width]

    for m, offset in zip(MODES, offsets):
        vals = np.array(data[m]["n_retries"])
        if not len(vals):
            continue
        counts = np.array([np.sum(vals == k) for k in range(max_retries + 1)], dtype=float)
        pcts   = counts / len(vals) * 100
        s = STYLES[m]
        ax.bar(x_centers + offset, pcts, width=width,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=1.5, label=s["label"])

    ax.set_xlabel("Number of Reflexion Retries")
    ax.set_ylabel("Fraction of Runs (%)")
    ax.set_title("Retry Count Distribution")
    ax.set_xticks(x_centers)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 3: Score Delta from Reflexion ──────────────────────────────────

def fig_score_delta(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    handles = []
    has_data = False
    for m in MODES:
        deltas = data[m]["retry_deltas"]
        if not deltas:
            continue
        has_data = True
        s = STYLES[m]
        befs = [d[0] for d in deltas]
        afts = [d[1] for d in deltas]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(befs))
        for b, a, j in zip(befs, afts, jitter):
            ax.annotate("", xy=(a, j), xytext=(b, j),
                        arrowprops=dict(arrowstyle="->",
                                        color=s["edge"], lw=1.8))
        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"], label=s["label"]))

    if not has_data:
        ax.text(0.5, 0.5, "No retry events recorded",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=24, color="gray")
    else:
        ax.axvline(7, color="steelblue", linestyle="--", linewidth=1.8, alpha=0.7,
                   label="Pass threshold (7)")
        handles.append(mpatches.Patch(facecolor="steelblue", label="Pass threshold (7)"))
        ax.legend(handles=handles, loc="upper left", framealpha=0.85)

    ax.set_xlabel("Judge Score (before → after retry)")
    ax.set_ylabel("Retry Events (jittered)")
    ax.set_title("Score Improvement via Reflexion")
    ax.set_xlim(0, 11)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 4: Which Agents Required Retries ────────────────────────────────

def fig_agent_failures(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    # Collect all agents that appear
    all_agents: set[str] = set()
    for m in MODES:
        all_agents.update(data[m]["failing_agents"])
    agents_sorted = sorted(all_agents)

    if not agents_sorted:
        ax.text(0.5, 0.5, "No retry events recorded",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=24, color="gray")
        if standalone:
            fig.tight_layout()
        return fig

    x = np.arange(len(MODES))
    w = 0.55

    # stacked bars
    agent_colors = {
        "rag":     "#6baed6",
        "code":    "#fd8d3c",
        "cloud":   "#74c476",
        "scholar": "#9e9ac8",
        "unknown": "#bdbdbd",
    }
    agent_hatches = {
        "rag":     "//",
        "code":    "..",
        "cloud":   "xx",
        "scholar": "--",
        "unknown": "",
    }

    bottoms = np.zeros(len(MODES))
    for agent in agents_sorted:
        counts = np.array([
            data[m]["failing_agents"].count(agent) for m in MODES
        ], dtype=float)
        color  = agent_colors.get(agent, "#bdbdbd")
        hatch  = agent_hatches.get(agent, "")
        bars   = ax.bar(x, counts, width=w, bottom=bottoms,
                        color=color, hatch=hatch,
                        edgecolor="black", linewidth=1.2, label=agent.title())
        for bar, v, bot in zip(bars, counts, bottoms):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + v / 2, str(int(v)),
                        ha="center", va="center",
                        fontsize=18, fontweight="bold", color="black")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Count of Retry Events")
    ax.set_title("Retries Triggered per Agent Type")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Combined 2×2 Panel ─────────────────────────────────────────────────────

def fig_panel(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(
        "Self-Healing & First-Pass Success Analysis",
        fontsize=36, fontweight="bold", y=1.01,
    )

    fig_fpsr(data,           ax=axes[0, 0])
    fig_retry_dist(data,     ax=axes[0, 1])
    fig_score_delta(data,    ax=axes[1, 0])
    fig_agent_failures(data, ax=axes[1, 1])

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
    print("=== 04_self_healing.py ===")
    records = load_records(TELEMETRY_DIR)
    data    = extract_healing_data(records)

    for m in MODES:
        fpsr   = np.mean(data[m]["fpsr"]) * 100 if data[m]["fpsr"] else 0
        n_ret  = np.mean(data[m]["n_retries"]) if data[m]["n_retries"] else 0
        n_evts = len(data[m]["retry_deltas"])
        print(f"  {m.upper():8s}: FPSR={fpsr:.1f}%, "
              f"mean_retries={n_ret:.2f}, retry_events={n_evts}")

    print("\nPlotting figures...")
    save(fig_fpsr(data),           "04_fpsr")
    save(fig_retry_dist(data),     "04_retry_dist")
    save(fig_score_delta(data),    "04_score_delta")
    save(fig_agent_failures(data), "04_agent_failures")
    save(fig_panel(data),          "04_panel")
    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
