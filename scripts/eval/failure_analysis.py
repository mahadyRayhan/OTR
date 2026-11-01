"""
failure_analysis.py
===================
Diagnoses WHY the adaptive pipeline fails on specific query types.

Four failure categories derived from telemetry signals:
  1. SCOPE_OVERFLOW   — too many GoT nodes + retries on exhaustive queries
  2. KB_GAP           — zero/near-zero RAG papers for the core topic
  3. SCHOLAR_CHAIN    — scholar-heavy DAGs where expert-lookup fails downstream
  4. ADVERSARIAL      — contradictory / category-mismatch / impossible queries

Outputs:
  results/failure_analysis/failure_map.png      — scatter: score vs nodes, colored by category
  results/failure_analysis/category_breakdown.png — bar charts per failure class
  results/failure_analysis/root_cause_signals.png — signal correlations
  results/failure_analysis/report.txt            — plain-text summary + fix recommendations

Usage:
    python scripts/eval/failure_analysis.py
    python scripts/eval/failure_analysis.py --dir historic_telemetry/raw_telemetry_best
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Failure-category rules (applied in priority order) ───────────────────────
# Each rule: (category_name, color, test_fn(run_dict) -> bool)

def _is_adversarial(r: dict) -> bool:
    q = r["query"].lower()
    adversarial_signals = [
        "simultaneously un-align",
        "video game",
        "multiplayer",
        "secondary structure of the resulting de genes",
        "quantum computing in genomics",
        "only if deseq2 can also",
        "doesn't use the cloud",
        "using deseq2",        # game backend query
    ]
    return any(s in q for s in adversarial_signals)

def _is_kb_gap(r: dict) -> bool:
    # Zero total RAG papers across all nodes, or known out-of-KB topics
    rag_papers = sum(
        n.get("paper_count", 0) or 0
        for n in r["dag_nodes"]
        if n.get("agent") in ("rag",)
    )
    q = r["query"].lower()
    kb_gap_signals = [
        "quantum computing", "qiskit", "monocle3", "salmon versus kallisto",
        "kallisto", "magic imputation", "bqsr", "haplotypecaller", "gatk",
        "snakemake workflow for bqsr",
    ]
    # Also flag if rag agents got 0 papers total
    return rag_papers == 0 or any(s in q for s in kb_gap_signals)

def _is_scholar_chain(r: dict) -> bool:
    # Scholar-heavy DAG (>4 scholar nodes) with retries = fragile name-chain
    scholar_count = sum(1 for n in r["dag_nodes"] if n.get("agent") == "scholar")
    return scholar_count >= 4 and r["retries"] >= 1

def _is_scope_overflow(r: dict) -> bool:
    # Many nodes + retries + exhaustive-language query
    scope_signals = [
        "retrieve all", "retrieve both", "all mandatory fields",
        "every cigar", "all 12 bits", "complete python library",
        "minimum 12 functions", "retrieve comprehensive",
        "retrieve all available", "retrieve the full", "every statistical formula",
    ]
    q = r["query"].lower()
    has_scope_language = any(s in q for s in scope_signals)
    return (len(r["dag_nodes"]) >= 8 or r["retries"] >= 2) and has_scope_language

CATEGORIES = [
    ("ADVERSARIAL",    "#e74c3c", _is_adversarial),
    ("KB_GAP",         "#e67e22", _is_kb_gap),
    ("SCHOLAR_CHAIN",  "#9b59b6", _is_scholar_chain),
    ("SCOPE_OVERFLOW", "#3498db", _is_scope_overflow),
    ("SUCCESS",        "#2ecc71", lambda r: True),  # catch-all
]

CAT_LABELS = {
    "ADVERSARIAL":    "Adversarial / Impossible",
    "KB_GAP":         "Knowledge Base Gap",
    "SCHOLAR_CHAIN":  "Scholar Chain Fragility",
    "SCOPE_OVERFLOW": "Scope Overflow",
    "SUCCESS":        "Success",
}

# ── Fixes (rendered in report) ────────────────────────────────────────────────
FIXES = {
    "ADVERSARIAL": """Root cause: The query is either contradictory, nonsensical, or a category mismatch
  (e.g. "game backend using DESeq2", "align and un-align simultaneously").
  The system tries to comply rather than reject, wasting cycles on impossible outputs.

  Fix — Add a Query Intent Validator before decomposition:
    1. In decompose_only(), before calling decomposer.decompose(), run a lightweight
       LLM call asking: "Is this question coherent and answerable by a bioinformatics
       AI? Reply YES or NO with one-line reason."
    2. If NO → return a structured rejection answer immediately with score=0 flagged.
    3. Cost: one tiny LLM call per query. Saves 3-5 full pipeline runs per batch.
    4. Bonus: The adversarial GROUP 5 queries are known test cases — you could also
       hard-code a keyword blacklist (["video game", "un-align", "secondary structure of DE genes"])
       as a free O(1) first pass.""",

    "KB_GAP": """Root cause: Topics not represented in the FAISS knowledge base (Monocle3, GATK,
  Kallisto, Quantum Genomics). RAG returns 0-1 papers → agent answers from LLM weights
  alone → high hallucination risk, low grounding score.

  Fix — Two-tier RAG with graceful fallback:
    1. When paper_count < 2 after FAISS search, auto-trigger the PubMed/Semantic Scholar
       live API path (already exists in KnowledgeBase). The live path is slower but not
       gated by what's in the local index.
    2. In rag_agent/agent.py the fallback to general LLM knowledge is currently commented
       out. Un-comment it but add a clear provenance marker:
         "⚠️ FALLBACK: No papers in local KB. Answer from LLM general knowledge — verify independently."
    3. Long-term: expand the FAISS index with GATK, Kallisto, Monocle3, MAGIC papers.""",

    "SCHOLAR_CHAIN": """Root cause: Scholar nodes spawn 4-7 separate API lookups to find researchers.
  If a researcher name is wrong or not found, the downstream code/cloud node receives
  an empty or fictional name and builds on a broken foundation. High retry count
  reflects GoT trying to fix downstream failures caused by the bad upstream name.

  Fix — Scholar output validation + chain guard:
    1. scholar_agent/agent.py already prepends [STRUCTURAL_FAIL] when experts_list is empty.
       Extend this: also flag when any returned expert name looks synthetic
       (no real publication list, name not found in Semantic Scholar).
    2. Add a scholar_agent result validator in _execute_agent() that checks:
       - Expert names are non-empty strings
       - At least one paper URL or DOI is present in the output
       If validation fails, mark the node as STRUCTURAL_FAIL before the downstream
       code node reads from it.
    3. In the decomposer, for scholar→code chains, add an intermediate RAG node that
       verifies the researcher's claimed methods exist in the literature before the
       code node uses them as ground truth.""",

    "SCOPE_OVERFLOW": """Root cause: "Retrieve ALL / BOTH / comprehensive" queries cause the GoT DAG
  to spawn 10-16 nodes, each covering a narrow slice. The AGGREGATE node then receives
  12 partial fragments and cannot synthesize a coherent whole — especially for code,
  where 5 separate code snippets need to form one library.

  Fix — Three complementary changes:
    1. Node budget cap: In AdaptiveGoTOrchestrator, add max_initial_nodes=6.
       Force the decomposer prompt to say: "Split into AT MOST 5 subtasks regardless
       of how many parts the question has. Merge related parts."
    2. Code consolidation node: When multiple code nodes exist in the DAG, add a
       dedicated CONSOLIDATE operator (already have AGGREGATE) that explicitly tells
       the LLM: "Merge these code snippets into one unified module with no duplicate
       imports and a single entry point."
    3. Query pre-simplification: Detect scope-overload language before decomposition
       ("retrieve ALL", "minimum N functions") and rewrite the query to:
       "Provide a representative and complete answer — you do not need to be exhaustive." """,
}


# ── Load telemetry ─────────────────────────────────────────────────────────────

def load_runs(tel_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(tel_dir.glob("run_*.json")):
        try:
            d  = json.loads(f.read_text())
            et = d.get("engine_telemetry", {})
            if d.get("ablation_mode") != "adaptive":
                continue
            nodes = et.get("got_dag_nodes", [])
            runs.append({
                "query":        (et.get("user_query") or d.get("query", "")).strip(),
                "score":        et.get("final_judge_score", 0) or 0,
                "retries":      et.get("global_retries", 0) or 0,
                "dag_nodes":    nodes,
                "n_nodes":      len(nodes),
                "n_scholar":    sum(1 for n in nodes if n.get("agent") == "scholar"),
                "n_rag":        sum(1 for n in nodes if n.get("agent") == "rag"),
                "n_code":       sum(1 for n in nodes if n.get("agent") == "code"),
                "total_papers": sum((n.get("paper_count") or 0) for n in nodes),
                "cugcr":        et.get("cugcr_triggered", False),
            })
        except Exception:
            pass
    return runs


def classify(run: dict) -> str:
    for name, _, test in CATEGORIES:
        if test(run):
            return name
    return "SUCCESS"


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_failure_map(runs: list[dict], out: Path):
    """Scatter: n_nodes vs score, colored by category."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    cat_color = {name: color for name, color, _ in CATEGORIES}
    jitter = np.random.default_rng(42)

    for r in runs:
        cat = r["category"]
        x = r["n_nodes"] + jitter.uniform(-0.25, 0.25)
        y = r["score"]   + jitter.uniform(-0.1, 0.1)
        ax.scatter(x, y, color=cat_color[cat], s=70, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)

    patches = [mpatches.Patch(color=cat_color[n], label=CAT_LABELS[n]) for n, _, _ in CATEGORIES]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    ax.set_xlabel("Number of GoT DAG Nodes", fontsize=10)
    ax.set_ylabel("Final Judge Score (0–10)", fontsize=10)
    ax.set_title("Failure Map: DAG Complexity vs Answer Quality", fontsize=11, fontweight="bold")
    ax.axhline(6, color="#aaa", linestyle="--", linewidth=0.8, label="Score 6 threshold")
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_category_breakdown(runs: list[dict], out: Path):
    """3-panel: count per category, mean score per category, retry rate per category."""
    cats = [n for n, _, _ in CATEGORIES if n != "SUCCESS"]
    cat_color = {name: color for name, color, _ in CATEGORIES}

    cat_runs = defaultdict(list)
    for r in runs:
        cat_runs[r["category"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("white")

    panels = [
        ("count",      "Query Count",           lambda rs: len(rs),                    None),
        ("mean_score", "Mean Score (0–10)",      lambda rs: np.mean([r["score"] for r in rs]), lambda rs: np.std([r["score"] for r in rs])),
        ("retry_rate", "Mean Retries per Query", lambda rs: np.mean([r["retries"] for r in rs]), None),
    ]

    all_cats = cats + ["SUCCESS"]
    for ax, (_, ylabel, val_fn, std_fn) in zip(axes, panels):
        vals   = [val_fn(cat_runs[c])  for c in all_cats if cat_runs[c]]
        errs   = [std_fn(cat_runs[c])  for c in all_cats if cat_runs[c]] if std_fn else None
        colors = [cat_color[c]          for c in all_cats if cat_runs[c]]
        labels = [CAT_LABELS[c]         for c in all_cats if cat_runs[c]]
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, 0.55, color=colors, edgecolor="white",
                      yerr=errs, capsize=4, error_kw={"linewidth": 1, "color": "#555"}, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=18, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.3 if vals else 1)
        ax.yaxis.grid(True, color="#eee", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Failure Category Breakdown  (adaptive mode, N=35)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_root_cause_signals(runs: list[dict], out: Path):
    """2-panel scatter: total_papers vs score, n_nodes vs retries, with regression lines."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    cat_color = {name: color for name, color, _ in CATEGORIES}

    # Panel 1: papers vs score
    ax = axes[0]
    xs = [r["total_papers"] for r in runs]
    ys = [r["score"]         for r in runs]
    cs = [cat_color[r["category"]] for r in runs]
    ax.scatter(xs, ys, c=cs, s=65, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
    if len(xs) > 2:
        m, b = np.polyfit(xs, ys, 1)
        xfit = np.linspace(min(xs), max(xs), 100)
        ax.plot(xfit, m * xfit + b, color="#555", linewidth=1.2, linestyle="--", label=f"r={np.corrcoef(xs,ys)[0,1]:.2f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("Total RAG Papers Retrieved", fontsize=10)
    ax.set_ylabel("Final Judge Score (0–10)", fontsize=10)
    ax.set_title("Paper Coverage vs Answer Quality", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: n_nodes vs retries (bubble = score)
    ax2 = axes[1]
    xs2 = [r["n_nodes"]  for r in runs]
    ys2 = [r["retries"]  for r in runs]
    ss2 = [(r["score"]+1) * 25 for r in runs]
    cs2 = [cat_color[r["category"]] for r in runs]
    sc = ax2.scatter(xs2, ys2, s=ss2, c=cs2, alpha=0.75, edgecolors="white", linewidth=0.5, zorder=3)
    ax2.set_xlabel("Number of DAG Nodes", fontsize=10)
    ax2.set_ylabel("Global Retries", fontsize=10)
    ax2.set_title("DAG Complexity vs Retry Cost\n(bubble size = score)", fontsize=10, fontweight="bold")
    ax2.yaxis.grid(True, color="#eee", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=cat_color[n], label=CAT_LABELS[n]) for n, _, _ in CATEGORIES]
    fig.legend(handles=patches, fontsize=7, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def write_report(runs: list[dict], out: Path):
    cat_runs = defaultdict(list)
    for r in runs:
        cat_runs[r["category"]].append(r)

    lines = [
        "FAILURE ANALYSIS REPORT — Adaptive Pipeline",
        "=" * 60,
        f"Total adaptive runs: {len(runs)}",
        f"Mean score: {np.mean([r['score'] for r in runs]):.2f}/10",
        "",
    ]

    for name, _, _ in CATEGORIES:
        rs = cat_runs[name]
        if not rs:
            continue
        scores  = [r["score"]   for r in rs]
        retries = [r["retries"] for r in rs]
        lines += [
            f"── {CAT_LABELS[name]} ({len(rs)} queries) ──",
            f"   Mean score : {np.mean(scores):.1f}  (min={min(scores)}, max={max(scores)})",
            f"   Mean retries: {np.mean(retries):.1f}",
            "   Queries:",
        ]
        for r in sorted(rs, key=lambda x: x["score"]):
            lines.append(f"     [{r['score']:2d}] {r['query'][:85]}")
        lines += ["", "   FIX:", FIXES.get(name, "No fix available."), ""]

    out.write_text("\n".join(lines))
    print(f"  → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=ROOT / "raw_telemetry")
    args = p.parse_args()

    out_dir = ROOT / "results" / "failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading adaptive runs from {args.dir}...")
    runs = load_runs(args.dir)
    if not runs:
        print("No adaptive runs found.")
        return

    # Classify
    for r in runs:
        r["category"] = classify(r)

    # Summary
    cat_counts = defaultdict(int)
    for r in runs:
        cat_counts[r["category"]] += 1
    print(f"\n  {len(runs)} queries classified:")
    for name, _, _ in CATEGORIES:
        if cat_counts[name]:
            rs = [r for r in runs if r["category"] == name]
            print(f"    {CAT_LABELS[name]:<30} n={cat_counts[name]:2d}  mean_score={np.mean([r['score'] for r in rs]):.1f}")

    print("\nGenerating plots...")
    plot_failure_map(runs,          out_dir / "failure_map.png")
    plot_category_breakdown(runs,   out_dir / "category_breakdown.png")
    plot_root_cause_signals(runs,   out_dir / "root_cause_signals.png")
    write_report(runs,              out_dir / "report.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
