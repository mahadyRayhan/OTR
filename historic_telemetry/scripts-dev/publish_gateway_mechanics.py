# import os
# import json
# from pathlib import Path
# from collections import defaultdict   # 🚀 ADDED THIS IMPORT
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import numpy as np

# # ── Configuration ────────────────────────────────────────────────────────────
# ROOT_DIR = Path(os.path.dirname(__file__)).parent
# TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
# VIZ_DIR = ROOT_DIR / "visualizations"
# VIZ_DIR.mkdir(exist_ok=True)

# COLORS = {"RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

# def load_telemetry():
#     runs = []
#     if not TELEMETRY_DIR.exists(): return []
#     for path in TELEMETRY_DIR.glob("run_*.json"):
#         with open(path, "r") as f:
#             try: runs.append(json.load(f))
#             except: pass
#     return runs

# def extract_gateway_metrics(runs):
#     """Extracts CRS, Token counts, and GTS to build the 3 panels."""
#     edge_data = []
#     query_groups = defaultdict(dict)

#     for run in runs:
#         mode = run.get("ablation_mode", "UNKNOWN").upper()
#         if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
#         t = run.get("engine_telemetry", {})
#         gts = t.get("final_judge_score", 0)
#         total_tokens = sum(e.get("context_token_count", 0) for e in t.get("communication_edges", []))
        
#         query = run.get("query", "Unknown")
#         query_groups[query][mode] = {"gts": gts, "tokens": total_tokens}

#         for edge in t.get("communication_edges", []):
#             crs = (edge.get("context_relevance") or 0) * 100
#             edge_data.append({"Mode": mode, "CRS (%)": crs})

#     df_crs = pd.DataFrame(edge_data)
#     df_crs = df_crs[df_crs["Mode"].isin(["RAW", "ADAPTIVE"])] # NONE passes 0 context

#     # Calculate Advanced Metrics (Compression & CE)
#     advanced_metrics = []
#     for query, modes in query_groups.items():
#         if not all(m in modes for m in ["NONE", "RAW", "ADAPTIVE"]): continue
        
#         gts_none = modes["NONE"]["gts"]
        
#         for mode in ["RAW", "ADAPTIVE"]:
#             gts_current = modes[mode]["gts"]
#             tokens_current = modes[mode]["tokens"]
            
#             delta_q = gts_current - gts_none
#             ce = delta_q / (tokens_current / 1000) if tokens_current > 0 else 0
            
#             compression = 1.0
#             if mode == "ADAPTIVE" and tokens_current > 0:
#                 compression = modes["RAW"]["tokens"] / tokens_current

#             advanced_metrics.append({
#                 "Mode": mode,
#                 "CE (Efficiency)": ce,
#                 "Compression Ratio": compression
#             })

#     df_adv = pd.DataFrame(advanced_metrics)
#     return df_crs, df_adv

# def main():
#     print("🎨 Generating Publication-Ready Mechanics Figure...")
    
#     runs = load_telemetry()
#     df_crs, df_adv = extract_gateway_metrics(runs)
    
#     if df_crs.empty or df_adv.empty:
#         print("⚠️ Not enough data to generate Master Figure 4.")
#         return

#     # =========================================================================
#     # MASTER FIGURE 4: The CUGCR Gateway Mechanics (1x3 Grid)
#     # =========================================================================
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
#     plt_sns.set_theme(style="whitegrid")

#     # ── Panel A: Token Compression Ratio (Boxplot) ──
#     ax1 = axes[0]
#     df_adaptive = df_adv[df_adv["Mode"] == "ADAPTIVE"]
    
#     plt_sns.boxplot(y=df_adaptive["Compression Ratio"], color=COLORS["ADAPTIVE"], width=0.4, linewidth=1.5, ax=ax1)
#     plt_sns.swarmplot(y=df_adaptive["Compression Ratio"], color=".25", alpha=0.6, size=5, ax=ax1)
    
#     ax1.axhline(1.0, color='black', linestyle='--', label='1.0x (No Compression)')
#     ax1.set_title("A. Token Compression Ratio", fontweight='bold')
#     ax1.set_ylabel("Multiplier (RAW Tokens / ADAPTIVE Tokens)")
#     ax1.legend(loc='upper right')

#     # ── Panel B: Context Relevance Score (Bar) ──
#     ax2 = axes[1]
#     sns_ax2 = plt_sns.barplot(data=df_crs, x="Mode", y="CRS (%)", 
#                               order=["RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', ax=ax2)
#     ax2.set_title("B. Context Relevance Score (CRS)", fontweight='bold')
#     ax2.set_ylabel("Mean Relevance of Injected Context (%)")
#     ax2.set_xlabel("")
#     ax2.set_ylim(0, max(df_crs["CRS (%)"].max() * 1.3, 25)) # Give headroom for labels
    
#     for p in sns_ax2.patches:
#         ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
#                      ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')

#     # ── Panel C: Communication Efficiency (Bar) ──
#     ax3 = axes[2]
#     sns_ax3 = plt_sns.barplot(data=df_adv, x="Mode", y="CE (Efficiency)", 
#                               order=["RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', ax=ax3)
    
#     ax3.set_title("C. Communication Efficiency (CE)", fontweight='bold')
#     ax3.set_ylabel("Quality Gained per 1,000 Tokens Spent (ΔQ)")
#     ax3.set_xlabel("")
    
#     # Add a line at 0
#     ax3.axhline(0, color='black', linewidth=1.2)
    
#     for p in sns_ax3.patches:
#         ax3.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
#                      ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3 if p.get_height() > 0 else -12), textcoords='offset points')

#     plt.suptitle("Mechanics of Causal Utility-Gated Context Routing (CUGCR)", fontweight='bold', fontsize=15, y=1.05)
#     plt.tight_layout()
    
#     output_path = VIZ_DIR / "PUBLISH_Fig4_Gateway_Mechanics.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Generated: {output_path.name}")

# if __name__ == "__main__":
#     main()


import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists(): return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

def extract_gateway_metrics(runs):
    edge_data = []
    query_groups = defaultdict(lambda: defaultdict(dict))

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        gts = t.get("final_judge_score", 0)
        total_tokens = sum(e.get("context_token_count", 0) for e in t.get("communication_edges", []))
        
        # Calculate Mean CRS for this run
        crs_list = [(edge.get("context_relevance") or 0) * 100 for edge in t.get("communication_edges", [])]
        mean_crs = np.mean(crs_list) if crs_list else 0
        
        query = run.get("query", "Unknown")
        query_groups[query][mode] = {"gts": gts, "tokens": total_tokens, "crs": mean_crs}

        for crs_val in crs_list:
            edge_data.append({"Mode": mode, "CRS (%)": crs_val})

    df_crs = pd.DataFrame(edge_data)
    df_crs = df_crs[df_crs["Mode"].isin(["RAW", "ADAPTIVE"])] 

    # Calculate Advanced Metrics (Compression & CE split by CRS Tier)
    advanced_metrics = []
    for query, modes in query_groups.items():
        if not all(m in modes for m in ["NONE", "RAW", "ADAPTIVE"]): continue
        
        gts_none = modes["NONE"]["gts"]
        # Use RAW mode's CRS to establish the "baseline" retrieval quality of this query
        baseline_crs = modes["RAW"]["crs"]
        crs_tier = "High CRS (>15%)" if baseline_crs >= 15.0 else "Low CRS (GIGO Regime)"
        
        for mode in ["RAW", "ADAPTIVE"]:
            gts_current = modes[mode]["gts"]
            tokens_current = modes[mode]["tokens"]
            
            delta_q = gts_current - gts_none
            ce = delta_q / (tokens_current / 1000) if tokens_current > 0 else 0
            
            compression = 1.0
            if mode == "ADAPTIVE" and tokens_current > 0:
                compression = modes["RAW"]["tokens"] / tokens_current

            advanced_metrics.append({
                "Mode": mode,
                "CE (Efficiency)": ce,
                "Compression Ratio": compression,
                "CRS Tier": crs_tier
            })

    df_adv = pd.DataFrame(advanced_metrics)
    return df_crs, df_adv

def main():
    print("🎨 Generating Phase 2: Gateway Mechanics Figure...")
    
    runs = load_telemetry()
    df_crs, df_adv = extract_gateway_metrics(runs)
    
    if df_crs.empty or df_adv.empty: return

    # =========================================================================
    # MASTER FIGURE 4: The CUGCR Gateway Mechanics (1x3 Grid)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt_sns.set_theme(style="whitegrid")

    # ── Panel A: Token Compression Ratio (Boxplot) ──
    ax1 = axes[0]
    df_adaptive = df_adv[df_adv["Mode"] == "ADAPTIVE"]
    plt_sns.boxplot(y=df_adaptive["Compression Ratio"], color=COLORS["ADAPTIVE"], width=0.4, linewidth=1.5, ax=ax1)
    plt_sns.swarmplot(y=df_adaptive["Compression Ratio"], color=".25", alpha=0.6, size=5, ax=ax1)
    ax1.axhline(1.0, color='black', linestyle='--', label='1.0x (No Compression)')
    ax1.set_title("A. Token Compression Ratio", fontweight='bold')
    ax1.set_ylabel("Multiplier (RAW / ADAPTIVE Tokens)")
    ax1.legend(loc='upper right')

    # ── Panel B: Context Relevance Score (Bar) ──
    ax2 = axes[1]
    sns_ax2 = plt_sns.barplot(data=df_crs, x="Mode", y="CRS (%)", order=["RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', ax=ax2)
    ax2.set_title("B. Context Relevance Score (CRS)", fontweight='bold')
    ax2.set_ylabel("Mean Relevance of Injected Context (%)")
    ax2.set_xlabel("")
    ax2.set_ylim(0, max(df_crs["CRS (%)"].max() * 1.3, 20)) 
    for p in sns_ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # ── Panel C: Communication Efficiency gated by CRS (Grouped Bar) ──
    ax3 = axes[2]
    # Custom palette to show Good vs Bad outcomes
    tier_palette = {"High CRS (>15%)": "#2DC653", "Low CRS (GIGO Regime)": "#D62828"}
    
    sns_ax3 = plt_sns.barplot(data=df_adv, x="Mode", y="CE (Efficiency)", hue="CRS Tier", 
                              order=["RAW", "ADAPTIVE"], palette=tier_palette, edgecolor='black', alpha=0.85, ax=ax3)
    ax3.set_title("C. Comm. Efficiency (Gated by Retrieval Quality)", fontweight='bold')
    ax3.set_ylabel("Quality Gained per 1,000 Tokens (ΔQ)")
    ax3.set_xlabel("")
    ax3.axhline(0, color='black', linewidth=1.2)
    ax3.legend(title="Upstream Retrieval Quality", loc='lower left')
    
    for p in sns_ax3.patches:
        val = p.get_height()
        if pd.notnull(val) and val != 0:
            ax3.annotate(f"{val:+.2f}", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold', 
                         xytext=(0, 3 if val > 0 else -12), textcoords='offset points')

    plt.suptitle("Mechanics of Causal Utility-Gated Context Routing (CUGCR)", fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = VIZ_DIR / "PUBLISH_Fig4_Gateway_Mechanics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path.name}")

if __name__ == "__main__":
    main()