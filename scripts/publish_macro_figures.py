# import os
# import json
# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import numpy as np
# from matplotlib.gridspec import GridSpec

# # ── Configuration ────────────────────────────────────────────────────────────
# ROOT_DIR = Path(os.path.dirname(__file__)).parent
# TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
# VIZ_DIR = ROOT_DIR / "visualizations"
# VIZ_DIR.mkdir(exist_ok=True)
# PAIRWISE_FILE = ROOT_DIR / "pairwise_eval_results.json"

# COLORS = {"RAW": "#4C72B0", "ADAPTIVE": "#E1812C", "TIE": "#95a5a6"}

# def load_telemetry():
#     runs = []
#     if not TELEMETRY_DIR.exists(): return []
#     for path in TELEMETRY_DIR.glob("run_*.json"):
#         with open(path, "r") as f:
#             try: runs.append(json.load(f))
#             except: pass
#     return runs

# def main():
#     print("🎨 Generating Publication-Ready Macro Figures...")
    
#     # =========================================================================
#     # MASTER FIGURE 1: Enhanced Pareto JointPlot (Cost vs Quality)
#     # =========================================================================
#     runs = load_telemetry()
#     plot_data = []
#     for run in runs:
#         mode = run.get("ablation_mode", "").upper()
#         if mode not in ["RAW", "ADAPTIVE"]: continue
        
#         t = run.get("engine_telemetry", {})
#         gts = t.get("final_judge_score", -1)
#         tokens = sum(e.get("context_token_count", 0) for e in t.get("communication_edges", []))
        
#         if gts >= 0:
#             plot_data.append({"Mode": mode, "Tokens Passed": tokens, "GTS": gts})

#     if plot_data:
#         df_pareto = pd.DataFrame(plot_data)
        
#         # Create a JointGrid
#         g = plt_sns.JointGrid(data=df_pareto, x="Tokens Passed", y="GTS", hue="Mode", height=7, ratio=4, palette=COLORS)
        
#         # Add scatter to the center
#         g.plot_joint(plt_sns.scatterplot, s=100, alpha=0.7, edgecolor="black")
        
#         # Add density curves to the margins
#         g.plot_marginals(plt_sns.kdeplot, fill=True, alpha=0.5, linewidth=2)
        
#         # Plot Means as massive 'X' markers in the center
#         means_df = df_pareto.groupby('Mode').mean().reset_index()
#         g.ax_joint.scatter(means_df['Tokens Passed'], means_df['GTS'], 
#                            marker='X', s=300, c=[COLORS[m] for m in means_df['Mode']], 
#                            edgecolor='black', zorder=10)
        
#         # Connect the means with a dashed line to show the Pareto shift
#         means_sorted = means_df.sort_values(by="Tokens Passed")
#         g.ax_joint.plot(means_sorted['Tokens Passed'], means_sorted['GTS'], color='gray', linestyle='--', alpha=0.6, zorder=5)

#         g.ax_joint.set_xlabel('Token Overhead per Query (Edge Communications)', fontweight='bold')
#         g.ax_joint.set_ylabel('Global Task Score (0-10)', fontweight='bold')
#         g.figure.suptitle("The Pareto Shift: Cost Reduction via Context Compression", fontweight='bold', y=1.02)
        
#         plt.savefig(VIZ_DIR / "PUBLISH_Fig1_Pareto_Jointplot.png", dpi=300, bbox_inches='tight')
#         print("✅ Generated: PUBLISH_Fig1_Pareto_Jointplot.png")

#     # =========================================================================
#     # MASTER FIGURE 2: Head-to-Head A/B Evaluation (1x2 Subplot)
#     # =========================================================================
#     if PAIRWISE_FILE.exists():
#         with open(PAIRWISE_FILE, "r") as f:
#             pairwise_data = json.load(f)
            
#         df_ab = pd.DataFrame(list(pairwise_data.values()))
        
#         fig = plt.figure(figsize=(12, 5))
#         gs = GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig)
        
#         # -- Panel A: Donut Chart --
#         ax1 = fig.add_subplot(gs[0, 0])
#         win_counts = df_ab['winner'].value_counts()
#         for cat in ["ADAPTIVE", "RAW", "TIE"]:
#             if cat not in win_counts: win_counts[cat] = 0
            
#         sizes = [win_counts['ADAPTIVE'], win_counts['RAW'], win_counts['TIE']]
#         labels = ['ADAPTIVE Wins', 'RAW Wins', 'Ties']
        
#         ax1.pie(sizes, labels=labels, colors=[COLORS['ADAPTIVE'], COLORS['RAW'], COLORS['TIE']], 
#                 autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
#         centre_circle = plt.Circle((0,0), 0.70, fc='white')
#         ax1.add_artist(centre_circle)
#         ax1.set_title("A. Head-to-Head Win Rate", fontweight='bold')

#         # -- Panel B: Diverging Margin Histogram --
#         ax2 = fig.add_subplot(gs[0, 1])
        
#         # Calculate signed margins (- for RAW, + for ADAPTIVE)
#         df_ab['plot_margin'] = df_ab.apply(
#             lambda row: row['margin'] if row['winner'] == 'ADAPTIVE' else (-row['margin'] if row['winner'] == 'RAW' else 0), 
#             axis=1
#         )
        
#         # Plot Histograms for each side
#         bins = np.arange(-5.5, 6.5, 1)
        
#         # RAW Wins (Negative Side)
#         raw_margins = df_ab[df_ab['winner'] == 'RAW']['plot_margin']
#         ax2.hist(raw_margins, bins=bins, color=COLORS['RAW'], edgecolor='black', alpha=0.85, label='RAW Win Margin')
        
#         # ADAPTIVE Wins (Positive Side)
#         adaptive_margins = df_ab[df_ab['winner'] == 'ADAPTIVE']['plot_margin']
#         ax2.hist(adaptive_margins, bins=bins, color=COLORS['ADAPTIVE'], edgecolor='black', alpha=0.85, label='ADAPTIVE Win Margin')
        
#         # TIES (Zero)
#         tie_margins = df_ab[df_ab['winner'] == 'TIE']['plot_margin']
#         if not tie_margins.empty:
#             ax2.hist(tie_margins, bins=bins, color=COLORS['TIE'], edgecolor='black', alpha=0.85, label='Ties')

#         # Formatting
#         ax2.axvline(0, color='black', linewidth=1.5, linestyle='--')
#         ax2.set_xticks([-5, -3, -1, 0, 1, 3, 5])
#         ax2.set_xticklabels(['5 (Crushing)', '3 (Clear)', '1 (Marginal)', 'Tie', '1 (Marginal)', '3 (Clear)', '5 (Crushing)'])
#         ax2.set_title('B. Distribution of Victory Margins', fontweight='bold')
#         ax2.set_xlabel('← RAW Wins                    ADAPTIVE Wins →', fontweight='bold')
#         ax2.set_ylabel('Number of Queries', fontweight='bold')
        
#         # Optional: add a grid on the y-axis to make it easier to read counts
#         ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
#         ax2.xaxis.grid(False)

#         plt.suptitle("Blinded LLM-as-a-Judge Pairwise Evaluation", fontweight='bold', fontsize=14, y=1.05)
#         plt.tight_layout()
#         plt.savefig(VIZ_DIR / "PUBLISH_Fig2_HeadToHead.png", dpi=300, bbox_inches='tight')
#         print("✅ Generated: PUBLISH_Fig2_HeadToHead.png")

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
import matplotlib.patches as patches

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
PAIRWISE_FILE = ROOT_DIR / "pairwise_eval_results.json"

COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C", "TIE": "#95a5a6"}

def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists(): return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

def categorize_query(query):
    """Categorizes queries into groups based on keywords for Panel C."""
    q = query.lower()
    if "leiden" in q or "pca and umap" in q or "alphafold 3" in q or "nextflow pipeline" in q:
        return "Perfect Hit"
    elif "experts" in q or "researchers" in q or "expert" in q:
        return "Knowledge Transfer"
    elif "needleman" in q or "kallisto" in q or "file format" in q or "ampa" in q or "tiering" in q:
        return "Context Overload"
    elif "harmony" in q or "rosetta" in q or "magic" in q or "quantum" in q or "maxquant" in q:
        return "Missing Data (GIGO)"
    elif "simultaneously" in q or "do the thing" in q or "take the outputs" in q or "multiplayer" in q:
        return "Adversarial"
    else:
        return "Baselines"

def main():
    print("🎨 Generating Phase 1: Macro Figures (Final Publication)...")
    
    runs = load_telemetry()
    plot_data = []
    for run in runs:
        mode = run.get("ablation_mode", "").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        gts = t.get("final_judge_score", -1)
        tokens = sum(e.get("context_token_count", 0) for e in t.get("communication_edges", []))
        
        if gts >= 0:
            plot_data.append({"Mode": mode, "Tokens Passed": tokens, "GTS": min(gts, 10.0)}) # Clamp to 10.0

    if plot_data:
        df_pareto = pd.DataFrame(plot_data)
        
        # =========================================================================
        # MASTER FIGURE 1: Enhanced Pareto JointPlot (Cost vs Quality)
        # =========================================================================
        g = plt_sns.JointGrid(data=df_pareto, x="Tokens Passed", y="GTS", hue="Mode", 
                              hue_order=["NONE", "RAW", "ADAPTIVE"], height=7, ratio=4, palette=COLORS)
        
        g.plot_joint(plt_sns.scatterplot, s=100, alpha=0.7, edgecolor="black")
        g.plot_marginals(plt_sns.kdeplot, fill=True, alpha=0.5, linewidth=2, common_norm=False)
        
        means_df = df_pareto.groupby('Mode').mean().reset_index()
        g.ax_joint.scatter(means_df['Tokens Passed'], means_df['GTS'], 
                           marker='X', s=300, c=[COLORS[m] for m in means_df['Mode']], 
                           edgecolor='black', zorder=10)
        
        means_sorted = means_df.sort_values(by="Tokens Passed")
        g.ax_joint.plot(means_sorted['Tokens Passed'], means_sorted['GTS'], color='gray', linestyle='--', alpha=0.6, zorder=5)

        # Highlight Efficient Frontier
        rect = patches.Rectangle((-100, 7.5), 1600, 3.0, linewidth=2, edgecolor='green', facecolor='green', alpha=0.1, zorder=1)
        g.ax_joint.add_patch(rect)
        g.ax_joint.text(750, 9.5, 'Efficient Frontier', color='darkgreen', fontsize=12, fontweight='bold', ha='center', va='center')

        g.ax_joint.set_xlabel('Token Overhead per Query (Edge Communications)', fontweight='bold')
        g.ax_joint.set_ylabel('Global Task Score (0-10)', fontweight='bold')
        g.ax_joint.set_ylim(-0.5, 10.5) # Clamp Visuals
        g.figure.suptitle("The Pareto Shift: Cost Reduction via Context Compression", fontweight='bold', y=1.02)
        
        plt.savefig(VIZ_DIR / "PUBLISH_Fig1_Pareto_Jointplot.png", dpi=300, bbox_inches='tight')
        print("✅ Generated: PUBLISH_Fig1_Pareto_Jointplot.png")

    # =========================================================================
    # MASTER FIGURE 2: Head-to-Head A/B Evaluation (1x3 Subplot)
    # =========================================================================
    if PAIRWISE_FILE.exists():
        with open(PAIRWISE_FILE, "r") as f:
            pairwise_data = json.load(f)
            
        df_ab = pd.DataFrame(list(pairwise_data.values()))
        df_ab['query_group'] = df_ab['query'].apply(categorize_query)
        
        fig = plt.figure(figsize=(18, 5.5))
        gs = GridSpec(1, 3, width_ratios=[1, 1.2, 1.2], figure=fig)
        
        # -- Panel A: Donut Chart --
        ax1 = fig.add_subplot(gs[0, 0])
        win_counts = df_ab['winner'].value_counts()
        for cat in ["ADAPTIVE", "RAW", "TIE"]:
            if cat not in win_counts: win_counts[cat] = 0
            
        sizes = [win_counts['ADAPTIVE'], win_counts['RAW'], win_counts['TIE']]
        labels = ['ADAPTIVE Wins', 'RAW Wins', 'Ties']
        
        ax1.pie(sizes, labels=labels, colors=[COLORS['ADAPTIVE'], COLORS['RAW'], COLORS['TIE']], 
                autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        ax1.add_artist(plt.Circle((0,0), 0.70, fc='white'))
        ax1.set_title("A. Global Head-to-Head Win Rate", fontweight='bold')

        # -- Panel B: Diverging Margin Histogram --
        ax2 = fig.add_subplot(gs[0, 1])
        df_ab['plot_margin'] = df_ab.apply(lambda row: row['margin'] if row['winner'] == 'ADAPTIVE' else (-row['margin'] if row['winner'] == 'RAW' else 0), axis=1)
        bins = np.arange(-5.5, 6.5, 1)
        
        ax2.hist(df_ab[df_ab['winner'] == 'RAW']['plot_margin'], bins=bins, color=COLORS['RAW'], edgecolor='black', alpha=0.85, label='RAW Win Margin')
        ax2.hist(df_ab[df_ab['winner'] == 'ADAPTIVE']['plot_margin'], bins=bins, color=COLORS['ADAPTIVE'], edgecolor='black', alpha=0.85, label='ADAPTIVE Win Margin')
        if not df_ab[df_ab['winner'] == 'TIE'].empty:
            ax2.hist(df_ab[df_ab['winner'] == 'TIE']['plot_margin'], bins=bins, color=COLORS['TIE'], edgecolor='black', alpha=0.85)

        ax2.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax2.set_xticks([-5, -3, -1, 0, 1, 3, 5])
        ax2.set_xticklabels(['5 (Crush)', '3 (Clear)', '1 (Marg)', 'Tie', '1 (Marg)', '3 (Clear)', '5 (Crush)'])
        ax2.set_title('B. Distribution of Victory Margins', fontweight='bold')
        ax2.set_xlabel('← RAW Wins                    ADAPTIVE Wins →', fontweight='bold')
        ax2.set_ylabel('Number of Queries', fontweight='bold')
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)

        # -- Panel C: Win Rate by Query Group --
        ax3 = fig.add_subplot(gs[0, 2])
        group_wins = df_ab.groupby(['query_group', 'winner']).size().unstack(fill_value=0)
        
        # Calculate Win Percentages
        for col in ['ADAPTIVE', 'RAW', 'TIE']:
            if col not in group_wins: group_wins[col] = 0
        group_totals = group_wins.sum(axis=1)
        group_wins_pct = group_wins.div(group_totals, axis=0) * 100

        # Custom sort logic
        sort_order = ["Perfect Hit", "Knowledge Transfer", "Context Overload", "Adversarial", "Missing Data (GIGO)", "Baselines"]
        existing_groups = [g for g in sort_order if g in group_wins_pct.index]
        group_wins_pct = group_wins_pct.loc[existing_groups]

        group_wins_pct[['RAW', 'ADAPTIVE']].plot(kind='bar', color=[COLORS['RAW'], COLORS['ADAPTIVE']], edgecolor='black', ax=ax3)
        ax3.set_title('C. Win Rate by Query Complexity Group', fontweight='bold')
        ax3.set_ylabel('Win Rate (%)', fontweight='bold')
        ax3.set_xlabel('Query Classification', fontweight='bold')
        ax3.set_ylim(0, 105)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.legend(title="Winner", loc='upper left')

        plt.suptitle("Blinded LLM-as-a-Judge Pairwise Evaluation (RAW vs ADAPTIVE)", fontweight='bold', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "PUBLISH_Fig2_HeadToHead.png", dpi=300, bbox_inches='tight')
        print("✅ Generated: PUBLISH_Fig2_HeadToHead.png")

if __name__ == "__main__":
    main()