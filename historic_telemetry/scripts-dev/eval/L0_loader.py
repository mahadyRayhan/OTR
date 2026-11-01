"""
L0_loader.py — Shared data loading for the 6-layer paper evaluation suite.

Import this in every Lx_*.py script:
    from L0_loader import load_all, MODES, STYLES, AGENTS, GROUPS, ...

All paths resolve relative to the project root (3 levels above this file).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Directories ────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
TELEMETRY_DIR = ROOT / "raw_telemetry"
FIG_DIR       = ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────
PASS_THRESHOLD       = 7     # final_judge_score >= this → run "passed"
NODE_PASS_THRESHOLD  = 0.70  # per-node quality_score  >= this → node "passed"

# ── Ablation modes ─────────────────────────────────────────────────────────
MODES = ["none", "raw", "adaptive"]
STYLES = {
    "none":     {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx",  "label": "NONE"},
    "raw":      {"face": "#ffcc99", "edge": "#cc3300", "hatch": "//",  "label": "RAW"},
    "adaptive": {"face": "#99ffcc", "edge": "#006633", "hatch": "..",  "label": "ADAPTIVE (Ours)"},
}

# ── Agents ─────────────────────────────────────────────────────────────────
AGENTS = ["rag", "scholar", "code", "cloud"]
AGENT_LABELS = {"rag": "RAG", "scholar": "Scholar", "code": "Code", "cloud": "Cloud"}
AGENT_COLORS = {
    "rag":     "#6baed6",
    "scholar": "#9e9ac8",
    "code":    "#fd8d3c",
    "cloud":   "#74c476",
}

# ── Query groups ───────────────────────────────────────────────────────────
GROUPS = {
    "G1": {"ids": [f"BQ{i:02d}" for i in range(1,  6)],  "label": "G1\nFull-Stack"},
    "G2": {"ids": [f"BQ{i:02d}" for i in range(6,  11)], "label": "G2\nCoreference"},
    "G3": {"ids": [f"BQ{i:02d}" for i in range(11, 16)], "label": "G3\nCtx Stress"},
    "G4": {"ids": [f"BQ{i:02d}" for i in range(16, 21)], "label": "G4\nMissing Data"},
    "G5": {"ids": [f"BQ{i:02d}" for i in range(21, 26)], "label": "G5\nAdversarial"},
    "G6": {"ids": [f"BQ{i:02d}" for i in range(26, 31)], "label": "G6\nBaseline"},
    "G7": {"ids": [f"BQ{i:02d}" for i in range(31, 36)], "label": "G7\nVol Stress"},
}

def qid_to_group(qid: str) -> str | None:
    for gk, gv in GROUPS.items():
        if qid in gv["ids"]:
            return gk
    return None


# ── Publication style ──────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         22,
        "font.weight":       "bold",
        "axes.labelsize":    32,
        "axes.titlesize":    36,
        "xtick.labelsize":   26,
        "ytick.labelsize":   26,
        "axes.labelweight":  "bold",
        "axes.titleweight":  "bold",
        "legend.fontsize":   22,
        "hatch.linewidth":   2.0,
        "figure.facecolor":  "white",
    })
    sns.set_style("whitegrid", {"grid.linestyle": "--"})


def mode_legend_handles():
    return [
        mpatches.Patch(
            facecolor=STYLES[m]["face"],
            edgecolor=STYLES[m]["edge"],
            hatch=STYLES[m]["hatch"],
            label=STYLES[m]["label"],
        )
        for m in MODES
    ]


def save_fig(fig: plt.Figure, stem: str):
    p = FIG_DIR / f"{stem}.png"
    fig.savefig(p, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  → figures/paper/{stem}.png")
    plt.close(fig)


def save_table(df, stem: str):
    """Save DataFrame as CSV and pretty-print to console."""
    p = FIG_DIR / f"{stem}.csv"
    df.to_csv(p, index=False)
    print(f"\n  → figures/paper/{stem}.csv")
    print(df.to_string(index=False))
    print()


# ── Core data loader ───────────────────────────────────────────────────────

def load_all(directory: Path = TELEMETRY_DIR) -> list[dict]:
    """
    Returns a list of run dicts, one per valid telemetry file.
    Each dict has flat fields (mode, judge_score, …) plus
    nested 'nodes' and 'edges' lists.
    """
    runs = []
    for f in sorted(directory.glob("run_*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception as exc:
            print(f"  [WARN] {f.name}: {exc}")
            continue

        et = rec.get("engine_telemetry", {})
        if not et or rec.get("failed"):
            continue

        mode = rec.get("ablation_mode", "")
        if mode not in MODES:
            continue

        score = et.get("final_judge_score")
        if score is None:
            continue

        qid   = rec.get("query_id", "")
        group = qid_to_group(qid)
        tok   = et.get("orchestrator_tokens") or {}

        # ── Extract nodes (exclude aggregate) ──────────────────────────────
        nodes = []
        for n in et.get("got_dag_nodes", []):
            agent = (n.get("agent") or "").lower()
            if agent in ("aggregate", ""):
                continue
            nodes.append({
                "agent":           agent,
                "quality_score":   n.get("quality_score"),
                "elapsed_s":       n.get("elapsed_s", 0) or 0,
                "failure_mode":    n.get("failure_mode"),
                "cugcr_triggered": n.get("cugcr_triggered", False),
                "crs_blocked":     n.get("crs_gate_blocked", False),
                "paper_count":     n.get("paper_count"),
                "rag_strategy":    n.get("rag_strategy") or n.get("strategy"),
            })

        # ── Extract communication edges ────────────────────────────────────
        edges = []
        for e in et.get("communication_edges", []):
            edges.append({
                "to_agent":       (e.get("to_agent") or "").lower(),
                "context_tokens": e.get("context_token_count", 0) or 0,
                "utilization":    e.get("utilization_score",   0.0) or 0.0,
                "relevance":      e.get("context_relevance",   0.0) or 0.0,
                "low_confidence": bool(e.get("low_confidence", False)),
                "dep_accuracy":   bool(e.get("dependency_accuracy", True)),
            })

        in_tok  = tok.get("input",  0) or 0
        out_tok = tok.get("output", 0) or 0

        runs.append({
            "mode":        mode,
            "query_id":    qid,
            "group":       group,
            "judge_score": float(score),
            "passed":      float(score) >= PASS_THRESHOLD,
            "wall_s":      et.get("latency_seconds") or rec.get("wall_latency_ms", 0) / 1000,
            "in_tokens":   in_tok,
            "out_tokens":  out_tok,
            "tokens":      in_tok + out_tok,
            "retries":     et.get("global_retries", 0) or 0,
            "fps":         bool(et.get("first_pass_success", False)),
            "cugcr":       bool(et.get("cugcr_triggered", False)),
            "retry_events":et.get("retry_events") or [],
            "agent_calls": len(nodes),
            "nodes":       nodes,
            "edges":       edges,
        })

    print(f"Loaded {len(runs)} valid runs  ({len(runs)//3} queries × 3 modes)")
    return runs


# ── Convenience helpers ────────────────────────────────────────────────────

def by_mode(runs: list[dict]) -> dict[str, list[dict]]:
    """Returns {mode: [run, ...]}"""
    return {m: [r for r in runs if r["mode"] == m] for m in MODES}


def recovery_success_rate(runs_m: list[dict]) -> float:
    """Fraction of retry events where score improved."""
    events = [ev for r in runs_m for ev in r["retry_events"]]
    if not events:
        return 0.0
    return sum(1 for ev in events if ev.get("improved", False)) / len(events) * 100
