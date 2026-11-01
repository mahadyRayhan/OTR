# import os
# import json
# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import numpy as np

# # ── Configuration ────────────────────────────────────────────────────────────
# ROOT_DIR = Path(os.path.dirname(__file__)).parent
# TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
# VIZ_DIR = ROOT_DIR / "visualizations"
# VIZ_DIR.mkdir(exist_ok=True)

# COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}
# PASSING_THRESHOLD = 0.75

# def load_telemetry():
#     runs = []
#     if not TELEMETRY_DIR.exists(): return []
#     for path in TELEMETRY_DIR.glob("run_*.json"):
#         with open(path, "r") as f:
#             try: runs.append(json.load(f))
#             except: pass
#     return runs

# def extract_metrics(runs):
#     """Extracts both NQS and First-Pass data in a single pass."""
#     all_nodes = []
#     first_pass_nodes = []

#     for run in runs:
#         mode = run.get("ablation_mode", "UNKNOWN").upper()
#         if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
#         t = run.get("engine_telemetry", {})
#         nodes = t.get("got_dag_nodes", [])
        
#         for node in nodes:
#             agent = node.get("agent", "UNKNOWN").upper()
#             score = node.get("quality_score")
#             operator = node.get("operator", "UNKNOWN").lower()
            
#             if score is not None:
#                 # Store for NQS analysis (scale 0-10)
#                 all_nodes.append({
#                     "Mode": mode,
#                     "Agent": agent,
#                     "NQS": score * 10  
#                 })
                
#                 # Store for First-Pass analysis (only root/branch nodes)
#                 if operator != "refine":
#                     first_pass_nodes.append({
#                         "Mode": mode,
#                         "Agent": agent,
#                         "Passed": 1 if score >= PASSING_THRESHOLD else 0
#                     })

#     return pd.DataFrame(all_nodes), pd.DataFrame(first_pass_nodes)

# def main():
#     print("🎨 Generating Publication-Ready Micro Figure (Agent Competency)...")
    
#     runs = load_telemetry()
#     df_nqs, df_fps = extract_metrics(runs)
    
#     if df_nqs.empty or df_fps.empty:
#         print("⚠️ Not enough data to generate Micro Figures.")
#         return

#     # Determine the order of agents based on overall NQS mean (best to worst)
#     agent_order = df_nqs.groupby("Agent")["NQS"].mean().sort_values(ascending=False).index.tolist()
#     hue_order = ["NONE", "RAW", "ADAPTIVE"]

#     # =========================================================================
#     # MASTER FIGURE 3: Agent Competency & Ablation Impact (2x2 Grid)
#     # =========================================================================
#     fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#     plt_sns.set_theme(style="whitegrid")

#     # ── Panel A: Overall NQS Distribution (Boxplot + Swarm) ──
#     ax = axes[0, 0]
#     plt_sns.boxplot(data=df_nqs, x="Agent", y="NQS", order=agent_order, 
#                     palette="Set2", linewidth=1.5, fliersize=0, ax=ax)
#     plt_sns.swarmplot(data=df_nqs, x="Agent", y="NQS", order=agent_order, 
#                       color=".25", alpha=0.5, size=3, ax=ax)
#     ax.set_title("A. Overall Agent Competency (NQS Distribution)", fontweight='bold')
#     ax.set_ylabel("Node Quality Score (0-10)")
#     ax.set_xlabel("")
#     ax.set_ylim(-0.5, 10.5)

#     # ── Panel B: Overall First-Pass Success Rate (Bar) ──
#     ax = axes[0, 1]
#     overall_fps = df_fps.groupby("Agent").agg(FPSR=("Passed", "mean")).reset_index()
#     overall_fps["FPSR"] *= 100
    
#     sns_ax = plt_sns.barplot(data=overall_fps, x="Agent", y="FPSR", order=agent_order, 
#                              palette="viridis", edgecolor='black', ax=ax)
#     ax.set_title("B. Baseline Reliability (First-Pass Success Rate)", fontweight='bold')
#     ax.set_ylabel("Success Rate (%)")
#     ax.set_xlabel("")
#     ax.set_ylim(0, 105)
#     for p in sns_ax.patches:
#         if p.get_height() > 0:
#             ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='bottom', fontsize=10, fontweight='bold')

#     # ── Panel C: NQS by Ablation Mode (Grouped Bar) ──
#     ax = axes[1, 0]
#     sns_ax = plt_sns.barplot(data=df_nqs, x="Agent", y="NQS", hue="Mode", 
#                              order=agent_order, hue_order=hue_order, 
#                              palette=COLORS, edgecolor='black', capsize=0.05, errorbar=('ci', 95), ax=ax)
#     ax.set_title("C. Ablation Impact on Reasoning Quality (Mean NQS)", fontweight='bold')
#     ax.set_ylabel("Mean NQS (0-10)")
#     ax.set_xlabel("Agent Type")
#     ax.set_ylim(0, 10.5)
#     ax.get_legend().remove() # We will put one master legend at the top
#     for p in sns_ax.patches:
#         if p.get_height() > 0:
#             ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 2), textcoords='offset points')

#     # ── Panel D: FPSR by Ablation Mode (Grouped Bar) ──
#     ax = axes[1, 1]
#     df_fps_grouped = df_fps.groupby(["Agent", "Mode"])["Passed"].mean().reset_index()
#     df_fps_grouped["FPSR (%)"] = df_fps_grouped["Passed"] * 100
    
#     sns_ax = plt_sns.barplot(data=df_fps_grouped, x="Agent", y="FPSR (%)", hue="Mode", 
#                              order=agent_order, hue_order=hue_order, 
#                              palette=COLORS, edgecolor='black', ax=ax)
#     ax.set_title("D. Ablation Impact on Baseline Reliability (FPSR)", fontweight='bold')
#     ax.set_ylabel("Success Rate (%)")
#     ax.set_xlabel("Agent Type")
#     ax.set_ylim(0, 105)
#     ax.get_legend().remove()
#     for p in sns_ax.patches:
#         if p.get_height() > 0:
#             ax.annotate(f"{p.get_height():.0f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 2), textcoords='offset points')

#     # ── Master Legend & Formatting ──
#     handles, labels = axes[1, 1].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, title="Ablation Mode", fontsize=11, title_fontsize=12)
    
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92) # Leave room for the master legend
    
#     output_path = VIZ_DIR / "PUBLISH_Fig3_Micro_Agent_Competency.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Generated: {output_path.name}")

# if __name__ == "__main__":
#     main()


import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}
PASSING_THRESHOLD = 0.75

def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists(): return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

def extract_metrics(runs):
    all_nodes = []
    first_pass_nodes = []

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        nodes = t.get("got_dag_nodes", [])
        
        for node in nodes:
            agent = node.get("agent", "UNKNOWN").upper()
            score = node.get("quality_score")
            operator = node.get("operator", "UNKNOWN").lower()
            
            if score is not None:
                nqs_clamped = min(score * 10, 10.0) # Clamp
                all_nodes.append({"Mode": mode, "Agent": agent, "NQS": nqs_clamped})
                
                if operator != "refine":
                    first_pass_nodes.append({"Mode": mode, "Agent": agent, "Passed": 1 if score >= PASSING_THRESHOLD else 0})

    return pd.DataFrame(all_nodes), pd.DataFrame(first_pass_nodes)

def main():
    print("🎨 Generating Phase 1: Micro Figures (Final Publication)...")
    
    runs = load_telemetry()
    df_nqs, df_fps = extract_metrics(runs)
    
    if df_nqs.empty or df_fps.empty: return

    agent_order = df_nqs.groupby("Agent")["NQS"].mean().sort_values(ascending=False).index.tolist()
    hue_order = ["NONE", "RAW", "ADAPTIVE"]

    # =========================================================================
    # MASTER FIGURE 3: Agent Competency & Sensitivity (2x3 Grid)
    # =========================================================================
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.6)
    plt_sns.set_theme(style="whitegrid")

    # ── Panel A: Overall NQS Distribution (Boxplot + Swarm) ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    plt_sns.boxplot(data=df_nqs, x="Agent", y="NQS", order=agent_order, palette="Set2", linewidth=1.5, fliersize=0, ax=ax1)
    plt_sns.swarmplot(data=df_nqs, x="Agent", y="NQS", order=agent_order, color=".25", alpha=0.5, size=3, ax=ax1)
    ax1.set_title("A. Overall Agent Competency (NQS)", fontweight='bold')
    ax1.set_ylabel("Node Quality Score (0-10)")
    ax1.set_xlabel("")
    ax1.set_ylim(-0.5, 10.5)

    # ── Panel B: Overall First-Pass Success Rate (Bar) ──
    ax2 = fig.add_subplot(gs[0, 2:4])
    overall_fps = df_fps.groupby("Agent").agg(FPSR=("Passed", "mean")).reset_index()
    overall_fps["FPSR"] *= 100
    sns_ax2 = plt_sns.barplot(data=overall_fps, x="Agent", y="FPSR", order=agent_order, palette="viridis", edgecolor='black', ax=ax2)
    ax2.set_title("B. Baseline Reliability (FPSR)", fontweight='bold')
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_xlabel("")
    ax2.set_ylim(0, 105)
    for p in sns_ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Panel E: Agent Sensitivity Index (NEW!) ──
    ax3 = fig.add_subplot(gs[0, 4:6])
    df_fps_grouped = df_fps.groupby(["Agent", "Mode"])["Passed"].mean().reset_index()
    df_fps_grouped["FPSR"] = df_fps_grouped["Passed"] * 100
    
    sensitivity_data = []
    for agent in agent_order:
        agent_data = df_fps_grouped[df_fps_grouped["Agent"] == agent]
        sensitivity = agent_data["FPSR"].max() - agent_data["FPSR"].min()
        sensitivity_data.append({"Agent": agent, "Sensitivity Index": sensitivity})
        
    df_sens = pd.DataFrame(sensitivity_data)
    sns_ax3 = plt_sns.barplot(data=df_sens, x="Agent", y="Sensitivity Index", order=agent_order, color="crimson", edgecolor="black", alpha=0.8, ax=ax3)
    ax3.set_title("E. Agent Context Sensitivity Index", fontweight='bold')
    ax3.set_ylabel("Max(FPSR) - Min(FPSR) Δ%")
    ax3.set_xlabel("")
    ax3.set_ylim(0, 105)
    for p in sns_ax3.patches:
        ax3.annotate(f"Δ {p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Panel C: NQS by Ablation Mode (Grouped Bar) ──
    ax4 = fig.add_subplot(gs[1, 0:3])
    sns_ax4 = plt_sns.barplot(data=df_nqs, x="Agent", y="NQS", hue="Mode", order=agent_order, hue_order=hue_order, 
                              palette=COLORS, edgecolor='black', capsize=0.05, errorbar=('ci', 95), ax=ax4)
    ax4.set_title("C. Ablation Impact on Reasoning Quality (Mean NQS)", fontweight='bold')
    ax4.set_ylabel("Mean NQS (0-10)")
    ax4.set_xlabel("Agent Type")
    ax4.set_ylim(0, 10.5)
    ax4.get_legend().remove()
    for p in sns_ax4.patches:
        if p.get_height() > 0:
            ax4.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ── Panel D: FPSR by Ablation Mode (Grouped Bar) ──
    ax5 = fig.add_subplot(gs[1, 3:6])
    sns_ax5 = plt_sns.barplot(data=df_fps_grouped, x="Agent", y="FPSR", hue="Mode", order=agent_order, hue_order=hue_order, 
                              palette=COLORS, edgecolor='black', ax=ax5)
    ax5.set_title("D. Ablation Impact on Baseline Reliability (FPSR)", fontweight='bold')
    ax5.set_ylabel("Success Rate (%)")
    ax5.set_xlabel("Agent Type")
    ax5.set_ylim(0, 105)
    ax5.legend(title="Ablation Mode", loc='upper right')
    for p in sns_ax5.patches:
        if p.get_height() > 0:
            ax5.annotate(f"{p.get_height():.0f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle("Micro-Agent Competency & Context Dependency Profiling", fontweight='bold', fontsize=16, y=0.98)
    
    output_path = VIZ_DIR / "PUBLISH_Fig3_Micro_Agent_Competency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path.name}")

if __name__ == "__main__":
    main()