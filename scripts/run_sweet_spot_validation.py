"""
run_sweet_spot_validation.py
============================
Controlled experiment to empirically validate the CUS sweet spot range.

Strategy: Run the same query multiple times under ADAPTIVE mode, but
artificially vary the token budget B to force different CUS levels.
A tight budget forces low utilization; a loose budget allows high utilization.
This gives us real NQS measurements across the full 0-100% CUS range
instead of relying on a sparse polynomial fit.

Output:
  sweet_spot_validation/raw_results.json
  sweet_spot_validation/fig_sweet_spot_validated.png
"""

import asyncio
import json
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv

# ── Path fix ──────────────────────────────────────────────────────────────────
# Works whether the script is run from scripts/, the project root, or anywhere.
# Assumes structure:  <project_root>/scripts/run_sweet_spot_validation.py
#                     <project_root>/orchestrator/engine.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")  # load .env from project root

from orchestrator.engine import OrchestratorEngine, AGENT_MANIFESTS, DependencyManifest, AtomType

# ── Silence the CommunicationEvaluator import error inside engine.py ─────────
# If the class name in your communication_evaluator.py differs from what
# engine.py expects, patch it here before the engine loads.
import importlib, types
try:
    import orchestrator.evaluation.communication_evaluator as _ce_mod
    # If the real class exists under a different name, alias it so engine.py finds it
    if not hasattr(_ce_mod, "CommunicationEvaluator"):
        # Find whatever evaluator class IS defined there and alias it
        for _name, _obj in vars(_ce_mod).items():
            if isinstance(_obj, type) and "eval" in _name.lower():
                _ce_mod.CommunicationEvaluator = _obj
                print(f"  [Path fix] Aliased '{_name}' → CommunicationEvaluator")
                break
except Exception as _e:
    print(f"  [warn] Could not patch CommunicationEvaluator: {_e}")

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("sweet_spot_validation")
OUT_DIR.mkdir(exist_ok=True)

# ── Experiment design ─────────────────────────────────────────────────────────
# These are the queries most likely to produce RAG→CODE or SCHOLAR→CODE edges
# i.e. queries where context passing actually happens between agents.
VALIDATION_QUERIES = [
    "What is the standard statistical method used by Scanpy to find marker genes? "
    "Explain the method, then write a Python script that applies it to find "
    "differentially expressed genes across Leiden clusters.",

    "Explain the theoretical difference between PCA and UMAP in single-cell analysis. "
    "Based on this, write a Python script using Scanpy to perform PCA, construct a "
    "neighborhood graph, and compute the UMAP embeddings.",

    "Write a complete Nextflow pipeline script for trimming Illumina FASTQ reads using "
    "FastQC and Trimmomatic, then design an AWS Batch deployment architecture.",
]

# Token budget sweep — from very tight (forces low CUS) to very loose (allows high CUS).
# These map approximately to the following CUS ranges:
#   50   tokens → CUS ≈ 0–10%   (almost nothing injected → agent ignores it)
#   150  tokens → CUS ≈ 15–30%  (below sweet spot)
#   300  tokens → CUS ≈ 35–55%  (target sweet spot)
#   500  tokens → CUS ≈ 50–70%  (upper sweet spot)
#   800  tokens → CUS ≈ 70–85%  (above sweet spot)
#   1200 tokens → CUS ≈ 85–100% (over-constrained region)
BUDGET_SWEEP = [50, 150, 300, 500, 800, 1200]

INTER_RUN_SLEEP = 35   # seconds — respects API rate limits
ERROR_SLEEP     = 90


# ── Override budgets at runtime ───────────────────────────────────────────────

def set_all_budgets(budget: int):
    """Temporarily override every agent's token budget to the given value."""
    for key in AGENT_MANIFESTS:
        original = AGENT_MANIFESTS[key]
        AGENT_MANIFESTS[key] = DependencyManifest(
            agent_type=original.agent_type,
            required_atom_types=original.required_atom_types,
            token_budget=budget
        )
    print(f"      [Budget Override] All agent budgets → {budget} tokens")


# ── Read the last edge's CUS and NQS from telemetry ──────────────────────────

def extract_last_run_edges(telemetry_file: str = "experiment_telemetry.json") -> list[dict]:
    """Read the most recently appended run from the telemetry file."""
    p = Path(telemetry_file)
    if not p.exists():
        return []
    with open(p) as f:
        all_runs = json.load(f)
    if not all_runs:
        return []
    last_run = all_runs[-1]

    # Build NQS map from got_dag_nodes
    nqs_map = {}
    for node in last_run.get("got_dag_nodes", []):
        score = node.get("quality_score")
        if score is not None:
            nqs_map[node["node_id"]] = round(score * 10, 2)  # scale to 0-10

    edges = []
    for edge in last_run.get("communication_edges", []):
        to_id = edge.get("to_id")
        nqs   = nqs_map.get(to_id)
        if nqs is None:
            continue
        cus = (edge.get("utilization_score") or 0) * 100
        crs = (edge.get("context_relevance")  or 0) * 100
        edges.append({
            "to_agent":      edge.get("to_agent", "unknown"),
            "budget_used":   None,   # filled in by caller
            "cus":           round(cus, 1),
            "crs":           round(crs, 1),
            "nqs":           nqs,
            "dsr":           1 if edge.get("dependency_accuracy") else 0,
            "ablation_mode": last_run.get("ablation_mode", "adaptive"),
        })
    return edges


# ── Main experiment loop ──────────────────────────────────────────────────────

async def run_validation():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       SWEET SPOT VALIDATION — CONTROLLED BUDGET SWEEP       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Queries  : {len(VALIDATION_QUERIES)}")
    print(f"Budgets  : {BUDGET_SWEEP}")
    print(f"Total runs: {len(VALIDATION_QUERIES) * len(BUDGET_SWEEP)}\n")

    engine      = OrchestratorEngine()
    all_results = []
    run_number  = 0
    total_runs  = len(VALIDATION_QUERIES) * len(BUDGET_SWEEP)

    for budget in BUDGET_SWEEP:
        set_all_budgets(budget)

        for q_idx, query in enumerate(VALIDATION_QUERIES):
            run_number += 1
            print(f"\n── Run {run_number}/{total_runs} | Budget={budget} | Query {q_idx+1}")
            try:
                start = time.time()
                await engine.run(query, ablation_mode="adaptive", start_time=start)

                edges = extract_last_run_edges()
                for edge in edges:
                    edge["budget_used"] = budget
                    edge["query_id"]    = f"Q{q_idx+1}"
                all_results.extend(edges)

                if edges:
                    print(f"  ✔  {len(edges)} edge(s) | " +
                          " | ".join([f"CUS={e['cus']:.0f}% NQS={e['nqs']}" for e in edges]))
                else:
                    print("  ⚠  No edges logged for this run")

                print(f"  ⏳ Sleeping {INTER_RUN_SLEEP}s...")
                await asyncio.sleep(INTER_RUN_SLEEP)

            except Exception as exc:
                print(f"  ✘  Error: {exc}")
                await asyncio.sleep(ERROR_SLEEP)

    # Save raw results
    with open(OUT_DIR / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Raw results saved: {OUT_DIR / 'raw_results.json'}")
    print(f"   Total edges collected: {len(all_results)}")

    plot_validation(all_results)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_validation(results: list[dict]):
    if not results:
        print("No results to plot.")
        return

    cus_vals = np.array([r["cus"] for r in results])
    nqs_vals = np.array([r["nqs"] for r in results])
    budgets  = np.array([r["budget_used"] for r in results])

    # ── Figure: 2 panels ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Sweet Spot Validation — Controlled Budget Sweep\n"
        "Each point = one real agent execution at a forced token budget",
        fontsize=13, fontweight="bold"
    )

    # ── Panel A: CUS vs NQS scatter + fitted curve ────────────────────────────
    ax = axes[0]

    # Colour by budget level
    norm    = plt.Normalize(vmin=min(BUDGET_SWEEP), vmax=max(BUDGET_SWEEP))
    cmap    = plt.cm.coolwarm_r
    scatter = ax.scatter(cus_vals, nqs_vals,
                         c=budgets, cmap=cmap, norm=norm,
                         s=90, alpha=0.8, edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(scatter, ax=ax, label="Token Budget B")

    # Polynomial fit — only plot if we have enough spread
    if len(cus_vals) >= 5 and cus_vals.std() > 5:
        try:
            coeffs   = np.polyfit(cus_vals, nqs_vals, 2)
            poly     = np.poly1d(coeffs)
            x_line   = np.linspace(0, 100, 300)
            y_line   = np.clip(poly(x_line), 0, 10)
            ax.plot(x_line, y_line, "k--", linewidth=2.5, label="Quadratic fit", zorder=4)

            # Find the empirical peak
            peak_x = -coeffs[1] / (2 * coeffs[0])
            peak_y = np.clip(poly(peak_x), 0, 10)
            if 0 <= peak_x <= 100:
                ax.axvline(peak_x, color="darkred", linestyle=":", linewidth=1.5,
                           label=f"Empirical peak CUS = {peak_x:.0f}%")
                ax.annotate(f"Peak\n{peak_x:.0f}%",
                            xy=(peak_x, peak_y),
                            xytext=(peak_x + 5, peak_y - 1.5),
                            arrowprops=dict(arrowstyle="->", color="darkred"),
                            fontsize=9, color="darkred")
        except Exception as e:
            print(f"  [warn] Curve fit failed: {e}")

    # Shade the originally claimed sweet spot (40-80%) for comparison
    ax.axvspan(40, 80, alpha=0.10, color="green",
               label="Originally claimed sweet spot (40–80%)")

    ax.set_xlabel("Context Utilisation Score CUS (%)", fontsize=11)
    ax.set_ylabel("Downstream Node Quality Score NQS (0–10)", fontsize=11)
    ax.set_title("Panel A — CUS vs NQS (empirical validation)", fontweight="bold")
    ax.set_xlim(-5, 105); ax.set_ylim(-0.5, 10.5)
    ax.axhline(7.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Quality gate (7.5)")
    ax.legend(fontsize=8, loc="lower center")

    # ── Panel B: Mean NQS per budget level (bar chart) ───────────────────────
    ax = axes[1]
    budget_groups = {}
    for r in results:
        b = r["budget_used"]
        budget_groups.setdefault(b, []).append(r["nqs"])

    budgets_sorted = sorted(budget_groups.keys())
    means  = [np.mean(budget_groups[b]) for b in budgets_sorted]
    stds   = [np.std(budget_groups[b])  for b in budgets_sorted]
    # Estimate approximate CUS range per budget for x-axis labels
    labels = [f"B={b}\n(~{_budget_to_cus(b)}% CUS)" for b in budgets_sorted]

    bar_colors = [cmap(norm(b)) for b in budgets_sorted]
    bars = ax.bar(range(len(budgets_sorted)), means,
                  yerr=stds, capsize=5,
                  color=bar_colors, alpha=0.85, edgecolor="black",
                  error_kw={"elinewidth": 1.5})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(7.5, color="grey", linestyle="--", linewidth=1, label="Quality gate (7.5)")
    ax.set_xticks(range(len(budgets_sorted)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean NQS (0–10)", fontsize=11)
    ax.set_title("Panel B — Mean NQS per Token Budget Level\n(lower budget → lower CUS)",
                 fontweight="bold")
    ax.set_ylim(0, 11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = OUT_DIR / "fig_sweet_spot_validated.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Validation plot saved: {out_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("VALIDATION SUMMARY")
    print("="*55)
    print(f"{'Budget':>8} | {'~CUS%':>6} | {'Mean NQS':>9} | {'n':>4}")
    print("-"*55)
    for b in budgets_sorted:
        vals    = budget_groups[b]
        cus_est = _budget_to_cus(b)
        print(f"{b:>8} | {cus_est:>5}% | {np.mean(vals):>9.2f} | {len(vals):>4}")

    # Check if peak is in sweet spot
    if len(cus_vals) >= 5 and cus_vals.std() > 5:
        try:
            coeffs = np.polyfit(cus_vals, nqs_vals, 2)
            peak_x = -coeffs[1] / (2 * coeffs[0])
            print(f"\n  Empirical NQS peak at CUS = {peak_x:.1f}%")
            if 40 <= peak_x <= 80:
                print("  ✔  Peak is WITHIN the claimed 40–80% sweet spot — hypothesis SUPPORTED")
            else:
                print(f"  ✘  Peak is OUTSIDE the claimed 40–80% range — hypothesis needs revision")
                print(f"     Suggested revised range: {max(0, peak_x-20):.0f}% – {min(100, peak_x+20):.0f}%")
        except Exception:
            pass


def _budget_to_cus(budget: int) -> str:
    """Rough heuristic mapping token budget to expected CUS range."""
    mapping = {50: "0-10", 150: "15-30", 300: "35-55", 500: "50-70", 800: "70-85", 1200: "85-100"}
    return mapping.get(budget, "?")


if __name__ == "__main__":
    asyncio.run(run_validation())


# =======================================================
# VALIDATION SUMMARY
# =======================================================
#   Budget |  ~CUS% |  Mean NQS |    n
# -------------------------------------------------------
#       50 |  0-10% |      6.86 |   11
#      150 | 15-30% |      5.29 |   10
#      300 | 35-55% |      5.30 |   10
#      500 | 50-70% |      5.16 |   12
#      800 | 70-85% |      6.00 |   11
#     1200 | 85-100% |      5.58 |    9

#   Empirical NQS peak at CUS = -4.2%
#   ✘  Peak is OUTSIDE the claimed 40–80% range — hypothesis needs revision
#      Suggested revised range: 0% – 16%