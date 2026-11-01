"""
07_query_group.py — Query Group Performance Analysis
=====================================================
Breaks down pipeline performance across the 6 benchmark query groups.
Each group stresses a different system property:

  G1  (BQ01-05) — Full-Stack RAG→Code→Cloud pipeline
  G2  (BQ06-10) — Coreference / knowledge transfer
  G3  (BQ11-15) — Context overload stress test
  G4  (BQ16-20) — Missing data / KB gaps / refusals
  G5  (BQ21-25) — Adversarial / ambiguous routing
  G6  (BQ26-30) — Single-agent baselines

Figures produced
----------------
  07_group_score.{png,pdf}      — final judge score per group × mode (grouped bar)
  07_group_fpsr.{png,pdf}       — first-pass success rate per group × mode
  07_group_latency.{png,pdf}    — wall latency per group × mode (heat-map)
  07_group_retries.{png,pdf}    — mean retries per group × mode
  07_panel.{png,pdf}            — 2×2 combined panel

Run:   python scripts/eval/07_query_group.py
Input: raw_telemetry/run_*.json
Out:   figures/07_*.{png,pdf}
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

GROUPS = {
    "G1": {"ids": [f"BQ{i:02d}" for i in range(1, 6)],
           "label": "G1\nFull-Stack"},
    "G2": {"ids": [f"BQ{i:02d}" for i in range(6, 11)],
           "label": "G2\nCoreference"},
    "G3": {"ids": [f"BQ{i:02d}" for i in range(11, 16)],
           "label": "G3\nCtx Overload"},
    "G4": {"ids": [f"BQ{i:02d}" for i in range(16, 21)],
           "label": "G4\nMissing Data"},
    "G5": {"ids": [f"BQ{i:02d}" for i in range(21, 26)],
           "label": "G5\nAdversarial"},
    "G6": {"ids": [f"BQ{i:02d}" for i in range(26, 31)],
           "label": "G6\nBaseline"},
}
GROUP_KEYS = list(GROUPS.keys())

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        20,
    "font.weight":      "bold",
    "axes.labelsize":   32,
    "axes.titlesize":   36,
    "xtick.labelsize":  24,
    "ytick.labelsize":  26,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.fontsize":  24,
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


def query_id_to_group(qid: str) -> str | None:
    for gk, gv in GROUPS.items():
        if qid in gv["ids"]:
            return gk
    return None


# ── Extraction ─────────────────────────────────────────────────────────────

def extract_group_data(records: list[dict]) -> dict:
    """
    Returns {group: {mode: {
        "scores":   [float],
        "fpsr":     [0|1],
        "latency_s":[float],
        "retries":  [int],
    }}}
    """
    data: dict = {
        gk: {m: defaultdict(list) for m in MODES}
        for gk in GROUP_KEYS
    }

    for rec in records:
        mode = rec.get("ablation_mode", "")
        if mode not in MODES:
            continue
        et = rec.get("engine_telemetry", {})
        if not et or rec.get("failed"):
            continue

        qid = rec.get("query_id", "")
        grp = query_id_to_group(qid)
        if grp is None:
            continue

        score   = et.get("final_judge_score")
        fps     = et.get("first_pass_success")
        wall_s  = et.get("latency_seconds") or rec.get("wall_latency_ms", 0) / 1000
        retries = et.get("global_retries", 0) or 0

        if score is None or fps is None:
            continue

        b = data[grp][mode]
        b["scores"].append(float(score))
        b["fpsr"].append(1 if fps else 0)
        b["latency_s"].append(wall_s)
        b["retries"].append(retries)

    return data


# ── Figure 1: Judge Score per Group × Mode ────────────────────────────────

def fig_group_score(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(16, 7))
    else:
        fig = None

    n_groups = len(GROUP_KEYS)
    group_w  = 1.0
    mode_w   = group_w / (len(MODES) + 1)
    offsets  = np.linspace(-group_w / 3, group_w / 3, len(MODES))
    x_base   = np.arange(n_groups) * (group_w + 0.4)

    handles = []
    for m_i, m in enumerate(MODES):
        s    = STYLES[m]
        means = [np.mean(data[gk][m]["scores"]) if data[gk][m]["scores"] else 0
                 for gk in GROUP_KEYS]
        stds  = [np.std(data[gk][m]["scores"])  if data[gk][m]["scores"] else 0
                 for gk in GROUP_KEYS]
        bars  = ax.bar(x_base + offsets[m_i], means, width=mode_w * 0.9,
                       yerr=stds, capsize=5,
                       color=s["face"], hatch=s["hatch"],
                       edgecolor=s["edge"], linewidth=1.5)
        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"],
            hatch=s["hatch"], label=s["label"]))

    ax.set_xticks(x_base)
    ax.set_xticklabels([GROUPS[gk]["label"] for gk in GROUP_KEYS])
    ax.set_ylabel("Mean Judge Score (0–10)")
    ax.set_title("Final Answer Quality by Query Group")
    ax.set_ylim(0, 12)
    ax.legend(handles=handles, loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.axhline(7, color="steelblue", linestyle="--", linewidth=1.5,
               alpha=0.5, label="Pass threshold (7)")

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 2: FPSR per Group × Mode ───────────────────────────────────────

def fig_group_fpsr(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(16, 7))
    else:
        fig = None

    n_groups = len(GROUP_KEYS)
    mode_w   = 0.22
    offsets  = np.linspace(-mode_w, mode_w, len(MODES))
    x_base   = np.arange(n_groups)

    handles = []
    for m_i, m in enumerate(MODES):
        s     = STYLES[m]
        rates = [
            np.mean(data[gk][m]["fpsr"]) * 100 if data[gk][m]["fpsr"] else 0
            for gk in GROUP_KEYS
        ]
        ax.bar(x_base + offsets[m_i], rates, width=mode_w * 0.9,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=1.5)
        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"],
            hatch=s["hatch"], label=s["label"]))

    ax.set_xticks(x_base)
    ax.set_xticklabels([GROUPS[gk]["label"] for gk in GROUP_KEYS])
    ax.set_ylabel("First-Pass Success Rate (%)")
    ax.set_title("First-Pass Success Rate by Query Group")
    ax.set_ylim(0, 110)
    ax.legend(handles=handles, loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 3: Latency Heatmap (group × mode) ──────────────────────────────

def fig_group_latency(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = None

    matrix = np.zeros((len(MODES), len(GROUP_KEYS)))
    for m_i, m in enumerate(MODES):
        for g_i, gk in enumerate(GROUP_KEYS):
            lats = data[gk][m]["latency_s"]
            matrix[m_i, g_i] = np.mean(lats) if lats else 0

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(GROUP_KEYS)))
    ax.set_xticklabels([GROUPS[gk]["label"] for gk in GROUP_KEYS], fontsize=18)
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([STYLES[m]["label"] for m in MODES], fontsize=22)
    ax.set_title("Mean Wall Latency (s) — Group × Mode")

    for i in range(len(MODES)):
        for j in range(len(GROUP_KEYS)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0f}s",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="black" if v < 0.6 * matrix.max() else "white")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Latency (s)", fontsize=20)

    if standalone:
        fig.tight_layout()
    return fig


# ── Figure 4: Mean Retries per Group × Mode ───────────────────────────────

def fig_group_retries(data: dict, ax=None) -> plt.Figure | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(16, 7))
    else:
        fig = None

    n_groups = len(GROUP_KEYS)
    mode_w   = 0.22
    offsets  = np.linspace(-mode_w, mode_w, len(MODES))
    x_base   = np.arange(n_groups)

    handles = []
    for m_i, m in enumerate(MODES):
        s = STYLES[m]
        means = [
            np.mean(data[gk][m]["retries"]) if data[gk][m]["retries"] else 0
            for gk in GROUP_KEYS
        ]
        ax.bar(x_base + offsets[m_i], means, width=mode_w * 0.9,
               color=s["face"], hatch=s["hatch"],
               edgecolor=s["edge"], linewidth=1.5)
        handles.append(mpatches.Patch(
            facecolor=s["face"], edgecolor=s["edge"],
            hatch=s["hatch"], label=s["label"]))

    ax.set_xticks(x_base)
    ax.set_xticklabels([GROUPS[gk]["label"] for gk in GROUP_KEYS])
    ax.set_ylabel("Mean Reflexion Retries")
    ax.set_title("Reflexion Retries by Query Group")
    ax.legend(handles=handles, loc="upper right", framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        fig.tight_layout()
    return fig


# ── Combined 2×2 Panel ─────────────────────────────────────────────────────

def fig_panel(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    fig.suptitle(
        "Query Group Breakdown — 6 Benchmark Categories",
        fontsize=36, fontweight="bold", y=1.01,
    )

    fig_group_score(data,   ax=axes[0, 0])
    fig_group_fpsr(data,    ax=axes[0, 1])
    fig_group_latency(data, ax=axes[1, 0])
    fig_group_retries(data, ax=axes[1, 1])

    labels = ["(A)", "(B)", "(C)", "(D)"]
    for ax, lbl in zip(axes.flat, labels):
        ax.text(-0.06, 1.04, lbl, transform=ax.transAxes,
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
    print("=== 07_query_group.py ===")
    records = load_records(TELEMETRY_DIR)
    data    = extract_group_data(records)

    for gk in GROUP_KEYS:
        for m in MODES:
            sc = data[gk][m]["scores"]
            print(f"  {gk} | {m.upper():8s}: n={len(sc):2d}, "
                  f"mean_score={np.mean(sc):.1f}" if sc else
                  f"  {gk} | {m.upper():8s}: n=0")

    print("\nPlotting figures...")
    save(fig_group_score(data),   "07_group_score")
    save(fig_group_fpsr(data),    "07_group_fpsr")
    save(fig_group_latency(data), "07_group_latency")
    save(fig_group_retries(data), "07_group_retries")
    save(fig_panel(data),         "07_panel")
    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
