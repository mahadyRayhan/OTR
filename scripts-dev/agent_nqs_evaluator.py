import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

# ── 1. Data Loader ───────────────────────────────────────────────────────────
def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists():
        print(f"❌ Error: {TELEMETRY_DIR} not found.")
        return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try:
                runs.append(json.load(f))
            except:
                pass
    return runs

# ── 2. Metrics Extraction ────────────────────────────────────────────────────
def extract_node_metrics(runs):
    """Extracts Node Quality Scores (NQS) for every agent across all runs."""
    node_data = []

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        nodes = t.get("got_dag_nodes", [])
        
        for node in nodes:
            agent = node.get("agent", "UNKNOWN").upper()
            score = node.get("quality_score")
            operator = node.get("operator", "UNKNOWN").upper()
            
            # The LLM judge scores nodes from 0.0 to 1.0. 
            # We scale it to 0-10 so it matches the Global Task Score (GTS) visual scale.
            if score is not None:
                node_data.append({
                    "Mode": mode,
                    "Agent": agent,
                    "Operator": operator,
                    "NQS": score * 10  
                })

    return pd.DataFrame(node_data)

# ── 3. Main Execution & Graphing ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          PER-AGENT NODE QUALITY SCORE (NQS) ANALYSIS     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    runs = load_telemetry()
    if not runs: return

    df = extract_node_metrics(runs)
    
    if df.empty:
        print("⚠️ No valid node scores found in telemetry.")
        return

    # --- TERMINAL PRINTOUT ---
    print("\n🔬 OVERALL AGENT PERFORMANCE (Aggregated across all modes)")
    overall_stats = df.groupby("Agent")["NQS"].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)
    
    for agent, row in overall_stats.iterrows():
        print(f"  ► {agent:<8} : Mean NQS = {row['mean']:.2f}/10 | StdDev = {row['std']:.2f} | Nodes Evaluated = {int(row['count'])}")

    print("\n🔬 AGENT PERFORMANCE BY ABLATION MODE")
    mode_stats = df.groupby(["Agent", "Mode"])["NQS"].mean().unstack()
    print(mode_stats.round(2).fillna("N/A").to_string())

    print("\n📊 Generating NQS Graphs...")

    # ── Graph G: Overall NQS Distribution (Boxplot) ──
    # A boxplot is perfect here to show the variance (are the scores consistent, or boom/bust?)
    plt.figure(figsize=(9, 6))
    plt_sns.set_theme(style="whitegrid")
    
    # Sort agents by their overall mean score for a cleaner graph
    order = df.groupby("Agent")["NQS"].mean().sort_values(ascending=False).index
    
    ax = plt_sns.boxplot(data=df, x="Agent", y="NQS", order=order, 
                         palette="Set2", linewidth=1.5, fliersize=4)
    
    # Overlay a swarmplot to show the actual data points
    plt_sns.swarmplot(data=df, x="Agent", y="NQS", order=order, 
                      color=".25", alpha=0.5, size=4)
                      
    plt.title('Overall Agent Competency: Node Quality Score Distribution', fontweight='bold')
    plt.ylabel('Node Quality Score (0 to 10)')
    plt.xlabel('Agent Type')
    plt.ylim(-0.5, 10.5)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig11_agent_nqs_boxplot.png")
    print("✅ Generated: fig11_agent_nqs_boxplot.png")

    # ── Graph H: Agent Performance by Ablation Mode (Grouped Bar Chart) ──
    plt.figure(figsize=(10, 6))
    
    ax2 = plt_sns.barplot(data=df, x="Agent", y="NQS", hue="Mode", 
                          order=order, palette=COLORS, edgecolor='black', alpha=0.85, capsize=0.1, errorbar=('ci', 95))
    
    plt.title('Agent Performance Breakdown by Context Mode (Mean NQS)', fontweight='bold')
    plt.ylabel('Mean Node Quality Score (0 to 10)')
    plt.xlabel('Agent Type')
    plt.ylim(0, 11)
    
    # Add labels on top of bars
    for p in ax2.patches:
        if p.get_height() > 0:
            ax2.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', xytext=(0, 3),
                         textcoords='offset points')

    plt.legend(title='Ablation Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig12_agent_mode_comparison.png")
    print("✅ Generated: fig12_agent_mode_comparison.png")

if __name__ == "__main__":
    main()

# ╔══════════════════════════════════════════════════════════╗
# ║          PER-AGENT NODE QUALITY SCORE (NQS) ANALYSIS     ║
# ╚══════════════════════════════════════════════════════════╝

# 🔬 OVERALL AGENT PERFORMANCE (Aggregated across all modes)
#   ► CLOUD    : Mean NQS = 9.14/10 | StdDev = 0.73 | Nodes Evaluated = 4
#   ► CODE     : Mean NQS = 8.74/10 | StdDev = 2.30 | Nodes Evaluated = 23
#   ► SCHOLAR  : Mean NQS = 3.80/10 | StdDev = 4.14 | Nodes Evaluated = 4
#   ► RAG      : Mean NQS = 3.52/10 | StdDev = 4.05 | Nodes Evaluated = 46

# 🔬 AGENT PERFORMANCE BY ABLATION MODE
# Mode     ADAPTIVE  NONE    RAW
# Agent                         
# CLOUD        9.30  8.62  10.00
# CODE         7.68  9.80   9.45
# RAG          3.92  3.12   3.59
# SCHOLAR      1.60  1.60   6.00