# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# from matplotlib.gridspec import GridSpec
# from pathlib import Path

# # ── Configuration ────────────────────────────────────────────────────────────
# ROOT_DIR = Path(os.path.dirname(__file__)).parent
# TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
# VALIDATION_DIR = ROOT_DIR / "sweet_spot_validation"
# VIZ_DIR = ROOT_DIR / "visualizations"
# VIZ_DIR.mkdir(exist_ok=True)

# # ── 1. Data Loaders ──────────────────────────────────────────────────────────
# def load_telemetry():
#     runs = []
#     if not TELEMETRY_DIR.exists(): return []
#     for path in TELEMETRY_DIR.glob("run_*.json"):
#         with open(path, "r") as f:
#             try: runs.append(json.load(f))
#             except: pass
#     return runs

# def load_validation_results():
#     val_file = VALIDATION_DIR / "raw_results.json"
#     if not val_file.exists(): return []
#     with open(val_file, "r") as f:
#         return json.load(f)

# def _budget_to_cus(budget: int) -> str:
#     mapping = {50: "0-10", 150: "15-30", 300: "35-55", 500: "50-70", 800: "70-85", 1200: "85-100"}
#     return mapping.get(budget, "?")

# # ── 3. Main Execution & Graphing ─────────────────────────────────────────────
# def main():
#     print("🎨 Generating Publication-Ready Plateau Figure...")
    
#     runs = load_telemetry()
#     val_results = load_validation_results()
    
#     if not runs or not val_results:
#         return

#     # Extract Baseline Data
#     combined_data = []
#     for run in runs:
#         if run.get("ablation_mode", "").upper() not in ["RAW", "ADAPTIVE"]: continue
#         t = run.get("engine_telemetry", {})
#         nqs_map = {n["node_id"]: n["quality_score"] * 10 for n in t.get("got_dag_nodes", []) if n.get("quality_score")}
#         for edge in t.get("communication_edges", []):
#             nqs = nqs_map.get(edge.get("to_id"))
#             if nqs is not None:
#                 combined_data.append({
#                     "CUS": (edge.get("utilization_score") or 0) * 100,
#                     "NQS": nqs,
#                     "Source": "Baseline"
#                 })

#     # Add Validation Data
#     for r in val_results:
#         nqs = r["nqs"] * 10 if r["nqs"] <= 1.0 else r["nqs"]
#         combined_data.append({
#             "CUS": r["cus"],
#             "NQS": nqs,
#             "Source": f"Budget {r['budget_used']}"
#         })

#     df = pd.DataFrame(combined_data)
#     if df.empty: return

#     # Create Bins for CUS
#     bins = [0, 20, 40, 60, 80, 100]
#     labels = ['0-20%\n(Starvation)', '20-40%\n(Optimal)', '40-60%\n(Optimal)', '60-80%\n(High Load)', '80-100%\n(Overload)']
#     df['CUS_Bin'] = pd.cut(df['CUS'], bins=bins, labels=labels, include_lowest=True)

#     # =========================================================================
#     # MASTER FIGURE 6: The Context Utilization Plateau (1x2 Grid)
#     # =========================================================================
#     fig = plt.figure(figsize=(16, 6))
#     gs = GridSpec(1, 2, width_ratios=[1.2, 1], figure=fig)
#     plt_sns.set_theme(style="whitegrid")

#     # ── Panel A: NQS Distribution by CUS Bin (Boxplot) ──
#     ax1 = fig.add_subplot(gs[0, 0])
    
#     # We use a boxplot to show that the MEDIAN score is highest in the 20-60% range
#     plt_sns.boxplot(data=df, x="CUS_Bin", y="NQS", palette="coolwarm", linewidth=1.5, fliersize=0, ax=ax1)
    
#     # Overlay swarmplot to show the actual 10/10 scores clustering in the middle
#     plt_sns.swarmplot(data=df, x="CUS_Bin", y="NQS", color=".25", alpha=0.5, size=4, ax=ax1)
    
#     ax1.axhline(7.5, color="grey", linestyle="--", linewidth=1.5, label="Quality Gate (7.5)")
#     ax1.axvspan(0.5, 2.5, color='green', alpha=0.1, label="Optimal Context Plateau (20-60% CUS)")
    
#     ax1.set_title("A. The Competence Plateau: NQS Distribution by Utilization", fontweight='bold')
#     ax1.set_xlabel("Context Utilization Score (CUS)")
#     ax1.set_ylabel("Downstream Node Quality Score (0-10)")
#     ax1.set_ylim(-0.5, 10.5)
#     ax1.legend(loc='lower left')

#     # ── Panel B: Mean NQS per Token Budget Level (Bar Chart) ──
#     ax2 = fig.add_subplot(gs[0, 1])
    
#     df_val = pd.DataFrame(val_results)
#     if df_val["nqs"].max() <= 1.0: df_val["nqs"] = df_val["nqs"] * 10
    
#     budget_stats = df_val.groupby("budget_used")["nqs"].agg(["mean", "std"]).reset_index()
#     budgets_sorted = budget_stats.sort_values(by="budget_used")
    
#     x_labels = [f"B={int(row['budget_used'])}\n(~{_budget_to_cus(int(row['budget_used']))}% CUS)" for _, row in budgets_sorted.iterrows()]
    
#     norm = plt.Normalize(vmin=budgets_sorted["budget_used"].min(), vmax=budgets_sorted["budget_used"].max())
#     cmap = plt.cm.coolwarm_r
#     bar_colors = [cmap(norm(b)) for b in budgets_sorted["budget_used"]]
    
#     bars = ax2.bar(range(len(budgets_sorted)), budgets_sorted["mean"],
#                    yerr=budgets_sorted["std"], capsize=5,
#                    color=bar_colors, alpha=0.85, edgecolor="black", error_kw={"elinewidth": 1.5})

#     for bar, mean in zip(bars, budgets_sorted["mean"]):
#         ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
#                  f"{mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight='bold')

#     ax2.axhline(7.5, color="grey", linestyle="--", linewidth=1.5, label="Quality Gate (7.5)")
#     ax2.set_xticks(range(len(budgets_sorted)))
#     ax2.set_xticklabels(x_labels)
#     ax2.set_ylabel("Mean NQS (0-10)")
#     ax2.set_title("B. The GIGO Effect: Mean NQS by Constrained Budget", fontweight='bold')
#     ax2.set_ylim(0, 11)
#     ax2.legend()

#     plt.suptitle("Targeted Ablation: Optimal Context Density for Generative Inference", fontweight='bold', fontsize=16, y=1.02)
#     plt.tight_layout()
    
#     output_path = VIZ_DIR / "PUBLISH_Fig6_Plateau_Validation.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Generated: {output_path.name}")

# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VALIDATION_DIR = ROOT_DIR / "sweet_spot_validation"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists(): return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

def load_validation_results():
    val_file = VALIDATION_DIR / "raw_results.json"
    if not val_file.exists(): return []
    with open(val_file, "r") as f:
        return json.load(f)

def _budget_to_cus(budget: int) -> str:
    mapping = {50: "0-10", 150: "15-30", 300: "35-55", 500: "50-70", 800: "70-85", 1200: "85-100"}
    return mapping.get(budget, "?")

def main():
    print("🎨 Generating Phase 3: The Cost-Efficient Context Frontier...")
    
    runs = load_telemetry()
    val_results = load_validation_results()
    if not runs or not val_results: return

    # Extract Baseline Data
    combined_data = []
    for run in runs:
        if run.get("ablation_mode", "").upper() not in ["RAW", "ADAPTIVE"]: continue
        t = run.get("engine_telemetry", {})
        nqs_map = {n["node_id"]: n["quality_score"] * 10 for n in t.get("got_dag_nodes", []) if n.get("quality_score")}
        for edge in t.get("communication_edges", []):
            nqs = nqs_map.get(edge.get("to_id"))
            if nqs is not None:
                combined_data.append({
                    "CUS": (edge.get("utilization_score") or 0) * 100,
                    "NQS": min(nqs, 10.0), # Clamp
                    "Tokens": edge.get("context_token_count", 0),
                    "Source": "Baseline"
                })

    for r in val_results:
        nqs = min(r["nqs"] * 10 if r["nqs"] <= 1.0 else r["nqs"], 10.0)
        combined_data.append({"CUS": r["cus"], "NQS": nqs, "Tokens": r.get("budget_used", 0), "Source": f"Budget {r['budget_used']}"})

    df = pd.DataFrame(combined_data)
    if df.empty: return

    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20%\n(Starvation)', '20-40%\n(Efficient)', '40-60%\n(Efficient)', '60-80%\n(High Load)', '80-100%\n(Overload)']
    df['CUS_Bin'] = pd.cut(df['CUS'], bins=bins, labels=labels, include_lowest=True)

    # =========================================================================
    # MASTER FIGURE 6: The Cost-Efficient Context Frontier
    # =========================================================================
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], figure=fig)
    plt_sns.set_theme(style="whitegrid")

    # ── Panel A: Dual-Axis Boxplot (Quality vs Cost) ──
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Primary Y-Axis (Quality)
    plt_sns.boxplot(data=df, x="CUS_Bin", y="NQS", palette="Blues", linewidth=1.5, fliersize=0, ax=ax1)
    plt_sns.swarmplot(data=df, x="CUS_Bin", y="NQS", color=".25", alpha=0.5, size=4, ax=ax1)
    ax1.set_ylabel("Downstream Node Quality Score (0-10)", color='darkblue', fontweight='bold')
    ax1.set_ylim(-0.5, 10.5)
    
    # Secondary Y-Axis (Token Cost)
    ax1_twin = ax1.twinx()
    mean_tokens = df.groupby("CUS_Bin")["Tokens"].mean().values
    ax1_twin.plot(range(len(labels)), mean_tokens, color="crimson", marker="D", markersize=8, linestyle="-", linewidth=2.5, label="Mean Token Cost")
    ax1_twin.set_ylabel("Mean Token Cost per Edge", color='crimson', fontweight='bold')
    ax1_twin.set_ylim(0, max(mean_tokens) * 1.2)
    ax1_twin.grid(False)

    ax1.axhline(7.5, color="grey", linestyle="--", linewidth=1.5)
    ax1.axvspan(0.5, 2.5, color='green', alpha=0.1)
    ax1.set_title("A. The Cost-Efficient Context Plateau", fontweight='bold')
    ax1.set_xlabel("Context Utilization Score (CUS Bin)")
    ax1_twin.legend(loc='lower right')

    # ── Panel B: Mean NQS per Token Budget Level ──
    ax2 = fig.add_subplot(gs[0, 1])
    df_val = pd.DataFrame(val_results)
    if df_val["nqs"].max() <= 1.0: df_val["nqs"] = df_val["nqs"] * 10
    
    budget_stats = df_val.groupby("budget_used")["nqs"].agg(["mean", "std"]).reset_index()
    budgets_sorted = budget_stats.sort_values(by="budget_used")
    x_labels = [f"B={int(row['budget_used'])}\n(~{_budget_to_cus(int(row['budget_used']))}%)" for _, row in budgets_sorted.iterrows()]
    
    norm = plt.Normalize(vmin=budgets_sorted["budget_used"].min(), vmax=budgets_sorted["budget_used"].max())
    cmap = plt.cm.coolwarm_r
    bars = ax2.bar(range(len(budgets_sorted)), budgets_sorted["mean"], yerr=budgets_sorted["std"], capsize=5,
                   color=[cmap(norm(b)) for b in budgets_sorted["budget_used"]], alpha=0.85, edgecolor="black")

    for bar, mean in zip(bars, budgets_sorted["mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight='bold')

    ax2.axhline(7.5, color="grey", linestyle="--", linewidth=1.5)
    ax2.set_xticks(range(len(budgets_sorted)))
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("Mean NQS (0-10)")
    ax2.set_title("B. Forced Constraints: NQS by Token Budget", fontweight='bold')
    ax2.set_ylim(0, 11)

    # Add the reviewer's requested GIGO annotation
    ax2.text(0.95, 0.05, "Note: Under GIGO regimes (CRS < 10%),\noptimal budget converges to minimum.", 
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.suptitle("The Cost-Efficient Context Frontier", fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = VIZ_DIR / "PUBLISH_Fig6_Plateau_Validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path.name}")

if __name__ == "__main__":
    main()