"""
visualize_ablation_metrics.py
=============================
Generates 4 high-resolution, publication-ready graphs specifically for the 
Causal Ablation Study (RAW vs. ADAPTIVE).

Reads all run_*.json files from the `raw_telemetry/` directory.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as plt_sns
import pandas as pd
import numpy as np
from pathlib import Path
import random 

# Fix python path so it can find the root folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Styling ──────────────────────────────────────────────────────────────────
plt_sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 300})

# ── Corrected Paths (Pointing to the Parent Directory) ───────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
RAW_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# ── Data Loader ──────────────────────────────────────────────────────────────
def load_telemetry_data() -> list[dict]:
    runs = []
    if not RAW_DIR.exists():
        print(f"❌ Error: {RAW_DIR.resolve()} not found. Run the benchmark first.")
        return []
        
    for path in sorted(RAW_DIR.glob("run_*.json")):
        with open(path, "r") as f:
            try:
                runs.append(json.load(f))
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not read {path.name}")
    print(f"📦 Loaded {len(runs)} telemetry files.")
    return runs

def get_telemetry(run: dict) -> dict:
    """Helper to safely extract the nested engine_telemetry dict."""
    return run.get("engine_telemetry", {})

# ── Graph 1: Task Success & Mean GTS (Bar Chart) ─────────────────────────────
def plot_task_outcome(runs):
    """Compares the final Task Success Rate and Mean GTS between RAW and ADAPTIVE."""
    data = {"raw": [], "adaptive": []}
    for run in runs:
        mode = run.get("ablation_mode", "").lower()
        if mode in data:
            data[mode].append(get_telemetry(run).get("final_judge_score", 0))

    if not data["raw"] and not data["adaptive"]:
        print("⚠️ Not enough data to plot Task Outcome.")
        return

    modes = ["raw", "adaptive"]
    labels = ["Blind Passing (RAW)", "Dynamic Gateway (ADAPTIVE)"]
    colors = ["#4C72B0", "#E1812C"]
    
    means = [np.mean(data[m]) if data[m] else 0 for m in modes]
    stds = [np.std(data[m]) if len(data[m]) > 1 else 0 for m in modes]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.85, edgecolor='black')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("Mean Global Task Score (0-10)")
    ax.set_title("End-to-End System Performance by Protocol")
    ax.set_ylim(0, 10.5)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig1_task_outcome.png")
    print("✅ Generated: fig1_task_outcome.png")

# ── Graph 2: Pareto Frontier (Cost vs Quality) ───────────────────────────────
def plot_pareto_frontier(runs):
    """Plots Token Cost against Global Task Score."""
    plot_data = []
    for run in runs:
        mode = run.get("ablation_mode", "").upper()
        if mode not in ["RAW", "ADAPTIVE"]: continue
        
        t = get_telemetry(run)
        gts = t.get("final_judge_score", -1)
        tokens = sum(e.get("context_token_count", 0) for e in t.get("communication_edges", []))
        
        if gts >= 0:
            plot_data.append({"Mode": mode, "Tokens Passed": tokens, "GTS": gts})

    if not plot_data: 
        print("⚠️ Not enough data to plot Pareto Frontier.")
        return
        
    df = pd.DataFrame(plot_data)
    means_df = df.groupby('Mode').mean().reset_index()

    plt.figure(figsize=(8, 6))
    
    # 🚀 FIXED: Explicitly map colors so Seaborn doesn't sort them alphabetically
    color_map = {"RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}
    
    plt_sns.scatterplot(data=df, x='Tokens Passed', y='GTS', hue='Mode', s=100, palette=color_map, alpha=0.6)
    plt_sns.scatterplot(data=means_df, x='Tokens Passed', y='GTS', hue='Mode', s=300, palette=color_map, marker="X", legend=False)
    
    means_sorted = means_df.sort_values(by="Tokens Passed")
    plt.plot(means_sorted['Tokens Passed'], means_sorted['GTS'], color='gray', linestyle='--', alpha=0.5)

    plt.title('The Pareto Frontier: Inter-Agent Cost vs. Reasoning Quality')
    plt.xlabel('Token Overhead per Query (Edge Communications)')
    plt.ylabel('Global Task Score (0-10)')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig2_pareto_frontier.png")
    print("✅ Generated: fig2_pareto_frontier.png")

# ── Graph 3: System Resilience (Recovery Funnel) ─────────────────────────────
def plot_system_recovery(runs):
    """Plots how many failed nodes successfully healed to a passing score (>=8)."""
    fail_counts = {"raw": 0, "adaptive": 0}
    recovery_counts = {"raw": 0, "adaptive": 0}

    for run in runs:
        mode = run.get("ablation_mode", "").lower()
        if mode not in fail_counts: continue
        
        t = get_telemetry(run)
        gts = t.get("final_judge_score", 0)
        had_failure = any((n.get("quality_score") or 1.0) < 0.75 for n in t.get("got_dag_nodes", []))
        global_retry = t.get("global_retries", 0) > 0
        
        if had_failure or global_retry:
            fail_counts[mode] += 1
            if gts >= 8.0:
                recovery_counts[mode] += 1

    labels = ["RAW", "ADAPTIVE"]
    # 🚀 FIXED: Make sure the arrays map exactly to the labels
    failed = [fail_counts["raw"], fail_counts["adaptive"]]
    recovered = [recovery_counts["raw"], recovery_counts["adaptive"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(x - width/2, failed, width, label='Errors Caught by DAG', color='lightcoral', edgecolor='black')
    ax.bar(x + width/2, recovered, width, label='Successfully Healed (Score ≥ 8)', color='mediumseagreen', edgecolor='black')

    ax.set_ylabel('Number of Queries')
    ax.set_title('System Resilience: Autonomous GoT Self-Healing')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig3_recovery_funnel.png")
    print("✅ Generated: fig3_recovery_funnel.png")

# ── Graph 4: Context Sensitivity (The Sweet Spot U-Curve) ────────────────────
def plot_context_sweet_spot(runs):
    """Plots Context Utilization % vs Node Quality Score."""
    utilizations = []
    qualities = []

    for run in runs:
        if run.get("ablation_mode", "") != "adaptive": continue
        
        t = get_telemetry(run)
        node_qualities = {n["node_id"]: n["quality_score"] for n in t.get("got_dag_nodes", []) if n.get("quality_score")}
        
        for edge in t.get("communication_edges", []):
            util = edge.get("utilization_score")
            qual = node_qualities.get(edge.get("to_id"), 0.0)
            if util is not None and qual > 0:
                # 🚀 Apply microscopic mathematical noise so the NumPy solver never divides by zero
                safe_util = (util * 100) + random.uniform(-1.0, 1.0)
                safe_qual = qual + random.uniform(-0.01, 0.01)
                
                utilizations.append(safe_util)
                qualities.append(safe_qual)

    if len(utilizations) < 3: 
        print(f"⚠️ Not enough data to plot U-Curve. Found {len(utilizations)} edges, need at least 3.")
        return

    plt.figure(figsize=(8, 6))
    
    # 🚀 Set ci=None to stop the bootstrapping algorithm from crashing on small datasets
    plt_sns.regplot(x=utilizations, y=qualities, order=2, ci=None,
                    scatter_kws={"s": 100, "alpha": 0.6, "color": "royalblue"}, 
                    line_kws={"color": "darkred", "linewidth": 2})
    
    plt.axvspan(30, 70, alpha=0.15, color="green", label="Theoretical 'Sweet Spot'")
    
    plt.title('Agent Cognitive Load: Context Utilization vs. Reasoning Quality')
    plt.xlabel('Agent-to-Agent Context Utilization (%)')
    plt.ylabel('Node Quality Score (NQS)')
    plt.xlim(-5, 105)
    plt.ylim(0, 1.1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig4_context_sweet_spot.png")
    print("✅ Generated: fig4_context_sweet_spot.png")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing Ablation Visualization Suite...")
    data = load_telemetry_data()
    
    if data:
        plot_task_outcome(data)
        plot_pareto_frontier(data)
        plot_system_recovery(data)
        plot_context_sweet_spot(data)
        print(f"\n🎯 Complete! Check the '{VIZ_DIR.resolve()}' folder.")
    else:
        print("❌ No telemetry data found to process.")