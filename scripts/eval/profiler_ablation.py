"""
05_profiler_ablation.py — Profiler Ablation Study
===================================================
Analyses whether the Neuro-Symbolic Query Profiler meaningfully improves
pipeline quality, and at what latency cost.

The profiler classifies each query into a (conceptual, procedural, analytical)
cognitive triplet and uses the dominant skill to bias task decomposition.
This script tests whether that classification actually leads to better outcomes.

Metrics
-------
  use_profiler        — bool: was the profiler active for this run?
  profiler_latency_ms — int: time taken by the profiler sub-call
  profiler_state      — {conceptual, procedural, analytical}: % distribution
  final_judge_score   — 0-10: downstream answer quality
  global_retries      — int: number of Reflexion retries needed

Figures produced
----------------
  05_quality_lift.{png,pdf}     — judge score violin: profiler ON vs OFF
  05_latency_cost.{png,pdf}     — profiler latency as % of total latency
  05_profile_dist.{png,pdf}     — ternary-style bar of profiler states per mode
  05_retry_impact.{png,pdf}     — retries needed with vs without profiler
  05_panel.{png,pdf}            — 2×2 combined panel

Run:   python scripts/eval/05_profiler_ablation.py
Input: raw_telemetry/run_*.json
Out:   figures/05_*.{png,pdf}
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as scipy_stats

# ── Paths ──────────────────────────────────────────────────────────────────
import argparse as _ap
ROOT          = Path(__file__).resolve().parent.parent.parent
FIG_DIR       = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--dir", type=Path,
                default=ROOT / "raw_telemetry_profiler_ablation")
_p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
_pargs, _ = _p.parse_known_args()
TELEMETRY_DIR = _pargs.dir
FIG_DIR       = _pargs.fig_dir
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["none", "raw", "adaptive"]
STYLES = {
    "none":     {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx",  "label": "NONE"},
    "raw":      {"face": "#ffcc99", "edge": "#cc3300", "hatch": "//",  "label": "RAW"},
    "adaptive": {"face": "#99ffcc", "edge": "#006633", "hatch": "..",  "label": "ADAPTIVE (Ours)"},
}

PROFILER_STYLES = {
    True:  {"face": "#74c476", "edge": "#006d2c", "hatch": "..", "label": "Profiler ON"},
    False: {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx", "label": "Profiler OFF"},
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

def extract_profiler_data(records: list[dict]) -> dict:
    """
    Returns {
        "by_flag": {True: {"scores":[float], "latency_pct":[float], "retries":[int]},
                    False: {...}},
        "by_mode": {mode: {"scores_on":[float], "scores_off":[float]}},
        "profile_states": {mode: {"conceptual":[float], "procedural":[float], "analytical":[float]}}
    }
    """
    by_flag  = {True: defaultdict(list), False: defaultdict(list)}
    by_mode  = {m: {"scores_on": [], "scores_off": []} for m in MODES}
    states   = {m: defaultdict(list) for m in MODES}

    for rec in records:
        mode = rec.get("ablation_mode", "")
        if mode not in MODES:
            continue
        et = rec.get("engine_telemetry", {})
        if not et or rec.get("failed"):
            continue

        use_profiler = et.get("use_profiler")
        if use_profiler is None:
            continue

        score   = et.get("final_judge_score")
        retries = et.get("global_retries", 0) or 0
        wall_ms = et.get("latency_seconds", 0) * 1000
        prof_ms = et.get("profiler_latency_ms") or 0

        if score is None:
            continue

        lat_pct = (prof_ms / wall_ms * 100) if (use_profiler and wall_ms > 0) else 0.0

        b = by_flag[use_profiler]
        b["scores"].append(float(score))
        b["latency_pct"].append(lat_pct)
        b["retries"].append(retries)

        if use_profiler:
            by_mode[mode]["scores_on"].append(float(score))
            ps = et.get("profiler_state") or {}
            for dim in ("conceptual", "procedural", "analytical"):
                val = ps.get(dim, 0) or 0
                states[mode][dim].append(val)
        else:
            by_mode[mode]["scores_off"].append(float(score))

    return {"by_flag": by_flag, "by_mode": by_mode, "profile_states": states}


# ── Figure 1: Quality Lift Violin ─────────────────────────────────────────

def fig_quality_lift(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    bf = data["by_flag"]
    groups = [True, False]
    vals   = [bf[g]["scores"] for g in groups]

    positions = [0, 1]
    parts = ax.violinplot(
        [v if v else [0] for v in vals],
        positions=positions,
        widths=0.55,
        showmedians=False,
        showextrema=False,
    )
    for pc, g in zip(parts["bodies"], groups):
        s = PROFILER_STYLES[g]
        pc.set_facecolor(s["face"])
        pc.set_edgecolor(s["edge"])
        pc.set_linewidth(2)
        pc.set_alpha(0.85)

    bp = ax.boxplot(
        [v if v else [0] for v in vals],
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
    )
    for patch, g in zip(bp["boxes"], groups):
        s = PROFILER_STYLES[g]
        patch.set_facecolor(s["face"])
        patch.set_hatch(s["hatch"])
        patch.set_edgecolor(s["edge"])

    for i, g in enumerate(groups):
        v = bf[g]["scores"]
        if v:
            med = np.median(v)
            ax.text(i, med + 0.3, f"{med:.1f}",
                    ha="center", va="bottom",
                    fontsize=20, fontweight="bold",
                    color=PROFILER_STYLES[g]["edge"])

    # significance test
    on_v  = bf[True]["scores"]
    off_v = bf[False]["scores"]
    if len(on_v) >= 3 and len(off_v) >= 3:
        t, p = scipy_stats.mannwhitneyu(on_v, off_v, alternative="greater")
        sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ymax  = max(max(on_v, default=0), max(off_v, default=0)) + 0.5
        ax.annotate("", xy=(1, ymax), xytext=(0, ymax),
                    arrowprops=dict(arrowstyle="-", lw=1.5, color="black"))
        ax.text(0.5, ymax + 0.1, sig, ha="center", fontsize=22, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([PROFILER_STYLES[g]["label"] for g in groups])
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Final Judge Score (0–10)")
    ax.set_title("Quality: Profiler ON vs. OFF")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 2: Latency Cost ─────────────────────────────────────────────────

def fig_latency_cost(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    pcts = [p for p in data["by_flag"][True]["latency_pct"] if p > 0]
    if not pcts:
        ax.text(0.5, 0.5, "No profiler latency data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=24, color="gray")
        if standalone:
            fig.tight_layout()
        return fig

    ax.hist(pcts, bins=12, color="#74c476", edgecolor="#006d2c",
            hatch="..", linewidth=1.5)
    ax.axvline(np.mean(pcts), color="crimson", linestyle="--", linewidth=2.5,
               label=f"Mean = {np.mean(pcts):.1f}%")
    ax.axvline(np.median(pcts), color="navy", linestyle=":", linewidth=2.5,
               label=f"Median = {np.median(pcts):.1f}%")
    ax.set_xlabel("Profiler Latency (% of Total Wall Time)")
    ax.set_ylabel("Run Count")
    ax.set_title("Profiler Overhead Distribution")
    ax.legend(framealpha=0.85)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 3: Profiler State Distribution by Mode ─────────────────────────

def fig_profile_dist(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    dims = ["conceptual", "procedural", "analytical"]
    dim_colors  = {"conceptual": "#6baed6", "procedural": "#fd8d3c", "analytical": "#74c476"}
    dim_hatches = {"conceptual": "//",      "procedural": "..",      "analytical": "xx"}

    x = np.arange(len(MODES))
    w = 0.55
    states = data["profile_states"]
    bottoms = np.zeros(len(MODES))

    has_any = any(states[m]["conceptual"] for m in MODES)
    if not has_any:
        ax.text(0.5, 0.5, "No profiler state data\n(profiler may not have been used)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=22, color="gray")
        ax.set_title("Profiler State Distribution by Mode")
        if standalone:
            fig.tight_layout()
        return fig

    for dim in dims:
        vals = np.array([
            np.mean(states[m][dim]) if states[m][dim] else 0 for m in MODES
        ])
        bars = ax.bar(x, vals, width=w, bottom=bottoms,
                      color=dim_colors[dim], hatch=dim_hatches[dim],
                      edgecolor="black", linewidth=1.2, label=dim.title())
        for bar, v, bot in zip(bars, vals, bottoms):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + v / 2, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=16, fontweight="bold", color="black")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([STYLES[m]["label"] for m in MODES])
    ax.set_ylabel("Mean Profiler State Score (%)")
    ax.set_title("Profiler State Distribution by Mode")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 4: Retries: Profiler ON vs OFF ─────────────────────────────────

def fig_retry_impact(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    bf = data["by_flag"]
    groups = [True, False]
    all_retries = [bf[g]["retries"] for g in groups]
    max_r = max((max(r) if r else 0) for r in all_retries)

    x_centers = np.arange(max_r + 1)
    width = 0.35
    offsets = [-0.18, 0.18]

    for g, offset in zip(groups, offsets):
        vals = np.array(bf[g]["retries"])
        if not len(vals):
            continue
        counts = np.array([np.sum(vals == k) for k in range(max_r + 1)], dtype=float)
        pcts   = counts / len(vals) * 100
        s = PROFILER_STYLES[g]
        ax.bar(x_centers + offset, pcts, width=width,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=1.5, label=s["label"])

    mean_on  = np.mean(bf[True]["retries"])  if bf[True]["retries"]  else 0
    mean_off = np.mean(bf[False]["retries"]) if bf[False]["retries"] else 0
    ax.set_xlabel("Number of Reflexion Retries")
    ax.set_ylabel("Fraction of Runs (%)")
    ax.set_title(f"Retries: ON μ={mean_on:.2f} | OFF μ={mean_off:.2f}")
    ax.set_xticks(x_centers)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Combined 2×2 Panel ─────────────────────────────────────────────────────

def fig_panel(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(
        "Neuro-Symbolic Profiler Ablation Study",
        fontsize=36, fontweight="bold", y=1.01,
    )

    fig_quality_lift(data,  ax=axes[0, 0])
    fig_latency_cost(data,  ax=axes[0, 1])
    fig_profile_dist(data,  ax=axes[1, 0])
    fig_retry_impact(data,  ax=axes[1, 1])

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
    print(f"=== 05_profiler_ablation.py  ({TELEMETRY_DIR.name}) ===")
    records = load_records(TELEMETRY_DIR)
    data    = extract_profiler_data(records)

    bf = data["by_flag"]
    for g in (True, False):
        n   = len(bf[g]["scores"])
        med = np.median(bf[g]["scores"]) if bf[g]["scores"] else 0
        mean_lat = np.mean(bf[g]["latency_pct"]) if bf[g]["latency_pct"] else 0
        print(f"  Profiler {'ON ' if g else 'OFF'}: n={n:3d}, "
              f"median_score={med:.1f}, mean_latency_pct={mean_lat:.1f}%")

    print("\nPlotting figures...")
    save(fig_quality_lift(data),  "05_quality_lift")
    save(fig_latency_cost(data),  "05_latency_cost")
    save(fig_profile_dist(data),  "05_profile_dist")
    save(fig_retry_impact(data),  "05_retry_impact")
    save(fig_panel(data),         "05_panel")
    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
