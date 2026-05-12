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
PASSING_THRESHOLD = 0.75  # NQS threshold for a node to pass without needing REFINE

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
def extract_fpsr_metrics(runs):
    """Extracts First-Pass Success Rates for each agent."""
    first_pass_data = []

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        nodes = t.get("got_dag_nodes", [])
        
        # We only want to look at the FIRST attempt (Root or Branch nodes)
        # We explicitly exclude REFINE nodes, as those are the second/third attempts
        for node in nodes:
            operator = node.get("operator", "UNKNOWN").lower()
            if operator == "refine": 
                continue
                
            agent = node.get("agent", "UNKNOWN").upper()
            score = node.get("quality_score")
            
            if score is not None:
                passed = 1 if score >= PASSING_THRESHOLD else 0
                first_pass_data.append({
                    "Mode": mode,
                    "Agent": agent,
                    "Operator": operator,
                    "NQS": score,
                    "Passed": passed
                })

    return pd.DataFrame(first_pass_data)

# ── 3. Main Execution & Graphing ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          FIRST-PASS SUCCESS RATE (FPSR) ANALYSIS         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    runs = load_telemetry()
    if not runs: return

    df = extract_fpsr_metrics(runs)
    
    if df.empty:
        print("⚠️ No valid first-pass node scores found in telemetry.")
        return

    # --- TERMINAL PRINTOUT ---
    print("\n🔬 OVERALL FIRST-PASS SUCCESS RATE (Aggregated across all modes)")
    
    # Calculate overall FPSR per agent
    overall_stats = df.groupby("Agent").agg(
        Total_Attempts=("Passed", "count"),
        Successful_First_Passes=("Passed", "sum")
    )
    overall_stats["FPSR (%)"] = (overall_stats["Successful_First_Passes"] / overall_stats["Total_Attempts"]) * 100
    overall_stats = overall_stats.sort_values(by="FPSR (%)", ascending=False)
    
    for agent, row in overall_stats.iterrows():
        print(f"  ► {agent:<8} : {row['FPSR (%)']:>5.1f}%  ({int(row['Successful_First_Passes'])}/{int(row['Total_Attempts'])} passed on attempt 1)")

    print("\n🔬 FIRST-PASS SUCCESS RATE BY ABLATION MODE (%)")
    mode_stats = df.groupby(["Agent", "Mode"])["Passed"].mean() * 100
    mode_stats_unstacked = mode_stats.unstack().round(1).fillna("N/A")
    print(mode_stats_unstacked.to_string())

    print("\n📊 Generating FPSR Graphs...")

    # ── Graph I: Overall FPSR (Bar Chart) ──
    plt.figure(figsize=(8, 5))
    plt_sns.set_theme(style="whitegrid")
    
    # Sort agents by their FPSR
    order = overall_stats.index
    
    ax = plt_sns.barplot(data=overall_stats.reset_index(), x="Agent", y="FPSR (%)", 
                         order=order, palette="viridis", edgecolor='black')
                         
    plt.title('Baseline Reliability: Overall First-Pass Success Rate', fontweight='bold')
    plt.ylabel('First-Pass Success Rate (%)')
    plt.xlabel('Agent Type')
    plt.ylim(0, 105)
    
    # Add percentage labels on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 3),
                        textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig13_agent_fpsr_overall.png")
    print("✅ Generated: fig13_agent_fpsr_overall.png")

    # ── Graph J: FPSR by Ablation Mode (Grouped Bar Chart) ──
    df_grouped = df.groupby(["Agent", "Mode"])["Passed"].mean().reset_index()
    df_grouped["FPSR (%)"] = df_grouped["Passed"] * 100

    plt.figure(figsize=(10, 6))
    
    # 🚀 FIXED: Enforce the order of the bars to NONE, RAW, ADAPTIVE
    hue_order = ["NONE", "RAW", "ADAPTIVE"]
    
    ax2 = plt_sns.barplot(data=df_grouped, x="Agent", y="FPSR (%)", hue="Mode", 
                          order=order, hue_order=hue_order, palette=COLORS, edgecolor='black', alpha=0.85)
    
    plt.title('First-Pass Success Rate by Context Mode', fontweight='bold')
    plt.ylabel('First-Pass Success Rate (%)')
    plt.xlabel('Agent Type')
    plt.ylim(0, 105)
    
    # Add labels on top of bars
    for p in ax2.patches:
        if p.get_height() > 0:
            ax2.annotate(f"{p.get_height():.0f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', xytext=(0, 3),
                         textcoords='offset points')

    plt.legend(title='Ablation Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig14_agent_fpsr_mode_comparison.png")
    print("✅ Generated: fig14_agent_fpsr_mode_comparison.png")

if __name__ == "__main__":
    main()

# ╔══════════════════════════════════════════════════════════╗
# ║          FIRST-PASS SUCCESS RATE (FPSR) ANALYSIS         ║
# ╚══════════════════════════════════════════════════════════╝

# 🔬 OVERALL FIRST-PASS SUCCESS RATE (Aggregated across all modes)
#   ► CLOUD    : 100.0%  (4/4 passed on attempt 1)
#   ► CODE     :  75.0%  (15/20 passed on attempt 1)
#   ► RAG      :   0.0%  (0/33 passed on attempt 1)
#   ► SCHOLAR  :   0.0%  (0/3 passed on attempt 1)

# 🔬 FIRST-PASS SUCCESS RATE BY ABLATION MODE (%)
# Mode     ADAPTIVE   NONE    RAW
# Agent                          
# CLOUD       100.0  100.0  100.0
# CODE         50.0  100.0   87.5
# RAG           0.0    0.0    0.0
# SCHOLAR       0.0    0.0    0.0