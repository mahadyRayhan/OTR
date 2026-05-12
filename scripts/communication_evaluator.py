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
def extract_communication_metrics(runs):
    """Extracts CRS, CUS, DSR, and NQS from every communication edge."""
    edge_data = []

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        
        # Build a fast lookup dictionary for Node Quality Scores (NQS)
        nqs_map = {}
        for node in t.get("got_dag_nodes", []):
            score = node.get("quality_score")
            if score is not None:
                nqs_map[node["node_id"]] = score * 10  # Scale 0.0-1.0 to 0-10

        # Loop through every communication edge
        for edge in t.get("communication_edges", []):
            to_id = edge.get("to_id")
            nqs = nqs_map.get(to_id)
            
            if nqs is None: continue # Skip if the node wasn't graded
            
            # The LLM Evaluator outputs 0.0 to 1.0. Scale to percentages.
            crs = (edge.get("context_relevance") or 0) * 100
            cus = (edge.get("utilization_score") or 0) * 100
            
            # Dependency Satisfaction Rate is a boolean True/False
            dsr = 100 if edge.get("dependency_accuracy") == True else 0
            
            edge_data.append({
                "Mode": mode,
                "Downstream Agent": edge.get("to_agent", "UNKNOWN").upper(),
                "CRS (%)": crs,
                "CUS (%)": cus,
                "DSR (%)": dsr,
                "NQS (0-10)": nqs
            })

    return pd.DataFrame(edge_data)

# ── 3. Main Execution & Graphing ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     AGENT-TO-AGENT COMMUNICATION & U-CURVE ANALYSIS      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    runs = load_telemetry()
    if not runs: return

    df = extract_communication_metrics(runs)
    
    if df.empty:
        print("⚠️ No valid communication edges found in telemetry.")
        return

    # --- TERMINAL PRINTOUT ---
    # We exclude NONE for CRS/CUS because NONE inherently passes 0 context.
    df_context = df[df["Mode"].isin(["RAW", "ADAPTIVE"])]
    
    print("\n🔬 CONTEXT RELEVANCE SCORE (CRS)")
    crs_stats = df_context.groupby("Mode")["CRS (%)"].mean()
    print(crs_stats.round(1).to_string())
    
    print("\n🔬 DEPENDENCY SATISFACTION RATE (DSR)")
    dsr_stats = df.groupby("Mode")["DSR (%)"].mean()
    print(dsr_stats.round(1).to_string())

    print("\n📊 Generating Communication Graphs...")

    # ── Graph K: Context Relevance Score (Bar Chart) ──
    plt.figure(figsize=(7, 5))
    plt_sns.set_theme(style="whitegrid")
    
    ax = plt_sns.barplot(data=df_context, x="Mode", y="CRS (%)", 
                         order=["RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', alpha=0.85)
                         
    plt.title('Context Filter Quality: Context Relevance Score (CRS)', fontweight='bold')
    plt.ylabel('Mean Relevance of Injected Context (%)')
    plt.ylim(0, 105)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 3),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig15_context_relevance_score.png")
    print("✅ Generated: fig15_context_relevance_score.png")

    # ── Graph L: Dependency Satisfaction Rate (Bar Chart) ──
    plt.figure(figsize=(8, 5))
    
    ax2 = plt_sns.barplot(data=df, x="Mode", y="DSR (%)", 
                          order=["NONE", "RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', alpha=0.85)
                          
    plt.title('Routing Success: Dependency Satisfaction Rate (DSR)', fontweight='bold')
    plt.ylabel('Dependencies Successfully Resolved (%)')
    plt.ylim(0, 105)
    
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 3),
                     textcoords='offset points')
                     
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig16_dependency_satisfaction_rate.png")
    print("✅ Generated: fig16_dependency_satisfaction_rate.png")

    # ── Graph M: The Inverted U-Curve (CUS vs NQS) ──
    plt.figure(figsize=(9, 6))
    
    # We only plot ADAPTIVE and RAW here to see how utilization affects the score.
    # Add minor jitter to CUS to prevent dots from stacking perfectly on top of each other
    df_curve = df_context.copy()
    df_curve["CUS Jittered"] = df_curve["CUS (%)"] + np.random.uniform(-1.5, 1.5, size=len(df_curve))
    df_curve["NQS Jittered"] = df_curve["NQS (0-10)"] + np.random.uniform(-0.1, 0.1, size=len(df_curve))

    # Plot Scatter Points
    plt_sns.scatterplot(data=df_curve, x="CUS Jittered", y="NQS Jittered", hue="Mode", 
                        palette=COLORS, s=80, alpha=0.7, edgecolor="black")

    # Fit a 2nd Degree Polynomial (Quadratic curve) to create the U-Shape
    try:
        x_vals = df_curve["CUS (%)"].values
        y_vals = df_curve["NQS (0-10)"].values
        
        if len(x_vals) >= 3:
            coeffs = np.polyfit(x_vals, y_vals, 2)
            poly_func = np.poly1d(coeffs)
            
            x_line = np.linspace(0, 100, 100)
            y_line = poly_func(x_line)
            
            # Clip the line so it doesn't draw above 10 or below 0
            y_line = np.clip(y_line, 0, 10)
            
            plt.plot(x_line, y_line, color="darkred", linewidth=2.5, linestyle="--", label="Quadratic Trend (U-Curve)")
    except Exception as e:
        print(f"⚠️ Could not fit polynomial curve: {e}")

    # Shaded Sweet Spot Area (e.g., 40% to 80% Utilization)
    plt.axvspan(40, 80, color='green', alpha=0.1, label="The 'Sweet Spot' (40-80% Utilization)")

    plt.title('Cognitive Load: Context Utilization vs. Reasoning Quality', fontweight='bold', fontsize=14)
    plt.xlabel('Agent-to-Agent Context Utilization Score (CUS %)', fontsize=12)
    plt.ylabel('Downstream Node Quality Score (NQS 0-10)', fontsize=12)
    plt.xlim(-5, 105)
    plt.ylim(-0.5, 10.5)
    plt.legend(loc='lower center')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig17_cus_u_curve.png")
    print("✅ Generated: fig17_cus_u_curve.png")

if __name__ == "__main__":
    main()

# ╔══════════════════════════════════════════════════════════╗
# ║     AGENT-TO-AGENT COMMUNICATION & U-CURVE ANALYSIS      ║
# ╚══════════════════════════════════════════════════════════╝

# 🔬 CONTEXT RELEVANCE SCORE (CRS)
# Mode
# ADAPTIVE    20.0
# RAW          6.2

# 🔬 DEPENDENCY SATISFACTION RATE (DSR)
# Mode
# ADAPTIVE    100.0
# NONE          0.0
# RAW         100.0