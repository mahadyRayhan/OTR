import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VALIDATION_DIR = ROOT_DIR / "sweet_spot_validation"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}
AGENT_COLORS = {"RAG": "#9b59b6", "SCHOLAR": "#3498db", "CODE": "#e74c3c", "CLOUD": "#2ecc71"}

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

def extract_novel_metrics(runs, val_results):
    # 1. Failure Taxonomy Data
    failure_data = []
    
    # 2. CRS vs CE Data (Adaptive mode only)
    ce_crs_data = []
    query_groups = defaultdict(lambda: defaultdict(dict))

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        gts = t.get("final_judge_score", 0)
        total_tokens = sum(e.get("context_token_count", 0) for e in t.get("communication_edges", []))
        
        query = run.get("query", "Unknown")
        crs_list = [(edge.get("context_relevance") or 0) * 100 for edge in t.get("communication_edges", [])]
        mean_crs = np.mean(crs_list) if crs_list else 0
        
        query_groups[query][mode] = {"gts": gts, "tokens": total_tokens, "crs": mean_crs}

        # Extract Failures
        for node in t.get("got_dag_nodes", []):
            if (node.get("quality_score") or 1.0) < 0.75:
                fail_mode = node.get("failure_mode", "unknown")
                # Clean up the enum string if necessary
                if fail_mode and isinstance(fail_mode, str):
                    fail_mode = fail_mode.replace("FailureMode.", "").replace("_", " ").title()
                failure_data.append({
                    "Agent": node.get("agent", "UNKNOWN").upper(),
                    "Failure Type": fail_mode
                })

    # Calculate CE for CRS Scatter
    for query, modes in query_groups.items():
        if not all(m in modes for m in ["NONE", "RAW", "ADAPTIVE"]): continue
        gts_none = modes["NONE"]["gts"]
        gts_adaptive = modes["ADAPTIVE"]["gts"]
        tokens_adaptive = modes["ADAPTIVE"]["tokens"]
        crs_adaptive = modes["ADAPTIVE"]["crs"]
        
        delta_q = gts_adaptive - gts_none
        ce = delta_q / (tokens_adaptive / 1000) if tokens_adaptive > 0 else 0
        
        # Only plot if we actually passed tokens and have a CRS
        if tokens_adaptive > 0:
            ce_crs_data.append({"CRS (%)": crs_adaptive, "CE (ΔQ/1k tokens)": ce})

    # 3. Agent Budget Sensitivity Data
    budget_data = []
    for r in val_results:
        agent = r.get("to_agent", "UNKNOWN").upper()
        nqs = r["nqs"] * 10 if r["nqs"] <= 1.0 else r["nqs"]
        budget_data.append({
            "Agent": agent,
            "Budget": r["budget_used"],
            "NQS": nqs
        })

    return pd.DataFrame(failure_data), pd.DataFrame(ce_crs_data), pd.DataFrame(budget_data)

def main():
    print("🎨 Generating Phase 4: Novel Mechanics Figures...")
    
    runs = load_telemetry()
    val_results = load_validation_results()
    
    df_fail, df_ce_crs, df_budget = extract_novel_metrics(runs, val_results)

    # =========================================================================
    # MASTER FIGURE 7: The Reviewer Requested Proofs (1x3 Grid)
    # =========================================================================
    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], figure=fig)
    plt_sns.set_theme(style="whitegrid")

    # ── Panel A: CRS vs CE (The Mechanistic Proof of GIGO) ──
    ax1 = fig.add_subplot(gs[0, 0])
    if not df_ce_crs.empty:
        # Add jitter to see overlapping points
        x_jitter = df_ce_crs["CRS (%)"] + np.random.uniform(-1, 1, size=len(df_ce_crs))
        y_jitter = df_ce_crs["CE (ΔQ/1k tokens)"] + np.random.uniform(-0.5, 0.5, size=len(df_ce_crs))
        
        plt_sns.regplot(x=x_jitter, y=y_jitter, scatter_kws={'s':60, 'alpha':0.6, 'edgecolor':'black'}, 
                        line_kws={'color':'red', 'linestyle':'--'}, ax=ax1, color=COLORS["ADAPTIVE"])
        
        ax1.axhline(0, color='black', linewidth=1.2)
        ax1.axvline(15, color='gray', linestyle=':', label='Theoretical CRS Gate (15%)')
        ax1.set_title("A. Context Relevance vs. Comm. Efficiency", fontweight='bold')
        ax1.set_xlabel("Context Relevance Score (CRS %)")
        ax1.set_ylabel("Communication Efficiency (CE)")
        ax1.legend(loc='lower right')
    else:
        ax1.text(0.5, 0.5, "Insufficient overlapping Data", ha='center', va='center')

    # ── Panel B: Failure Taxonomy Distribution ──
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_fail.empty:
        # Cross-tabulate and normalize to percentages
        fail_counts = pd.crosstab(df_fail['Agent'], df_fail['Failure Type'])
        fail_pct = fail_counts.div(fail_counts.sum(axis=1), axis=0) * 100
        
        # Ensure agents are in a logical order
        agents_present = [a for a in ["CLOUD", "CODE", "SCHOLAR", "RAG"] if a in fail_pct.index]
        fail_pct = fail_pct.loc[agents_present]
        
        fail_pct.plot(kind='bar', stacked=True, colormap='Spectral', edgecolor='black', ax=ax2)
        
        ax2.set_title("B. DAG Failure Taxonomy by Agent Type", fontweight='bold')
        ax2.set_ylabel("Percentage of Total Failures (%)")
        ax2.set_xlabel("Agent Type")
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='x', rotation=0)
        ax2.legend(title='Failure Classification', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No Failed Nodes Logged", ha='center', va='center')

    # ── Panel C: Agent-Specific Budget Sensitivity ──
    ax3 = fig.add_subplot(gs[0, 2])
    if not df_budget.empty:
        # Group by Agent and Budget to get Mean NQS
        budget_means = df_budget.groupby(["Agent", "Budget"])["NQS"].mean().reset_index()
        
        for agent in budget_means["Agent"].unique():
            agent_data = budget_means[budget_means["Agent"] == agent]
            color = AGENT_COLORS.get(agent, "black")
            ax3.plot(agent_data["Budget"], agent_data["NQS"], marker='o', markersize=8, 
                     linewidth=2.5, label=agent, color=color, alpha=0.85)

        ax3.axhline(7.5, color="grey", linestyle="--", linewidth=1.5, label="Quality Gate (7.5)")
        ax3.set_title("C. Agent Sensitivity to Context Compression", fontweight='bold')
        ax3.set_xlabel("Forced Token Budget (B)")
        ax3.set_ylabel("Mean Node Quality Score (0-10)")
        ax3.set_ylim(-0.5, 10.5)
        ax3.set_xscale('log') # Log scale is better for viewing budgets from 50 to 1200
        ax3.set_xticks([50, 150, 300, 500, 800, 1200])
        ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Remove scientific notation
        
        # Legend
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels, loc='lower left', fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No Validation Budget Data", ha='center', va='center')

    plt.suptitle("Empirical Justification for Dynamic Context Routing Policies", fontweight='bold', fontsize=16, y=1.05)
    plt.tight_layout()
    
    output_path = VIZ_DIR / "PUBLISH_Fig7_Novel_Mechanics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path.name}")

if __name__ == "__main__":
    main()