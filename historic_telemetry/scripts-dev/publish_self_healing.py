# import os
# import json
# import re
# from pathlib import Path
# from collections import defaultdict
# from openai import OpenAI  # Use synchronous OpenAI for simpler script execution
# from dotenv import load_dotenv
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import numpy as np
# from matplotlib.gridspec import GridSpec

# # Load environment variables
# load_dotenv()

# # ── Configuration ────────────────────────────────────────────────────────────
# ROOT_DIR = Path(os.path.dirname(__file__)).parent
# TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
# VIZ_DIR = ROOT_DIR / "visualizations"
# VIZ_DIR.mkdir(exist_ok=True)
# RESULTS_FILE = ROOT_DIR / "reflexion_quality_results.json"

# COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

# def load_telemetry():
#     runs = []
#     if not TELEMETRY_DIR.exists(): return []
#     for path in TELEMETRY_DIR.glob("run_*.json"):
#         with open(path, "r") as f:
#             try: runs.append(json.load(f))
#             except: pass
#     return runs

# def extract_reflection_text(output: str) -> str:
#     """Robust extraction: Looks for XML tags, falls back to markdown headers, or grabs first 500 chars."""
#     if not output: return ""
#     match = re.search(r'<reflection>(.*?)</reflection>', output, re.DOTALL | re.IGNORECASE)
#     if match: return match.group(1).strip()
#     match = re.search(r'(?:###|\*\*|#)?\s*Reflection[:\s]*(.*?)(?:\n\n|\Z)', output, re.DOTALL | re.IGNORECASE)
#     if match: return match.group(1).strip()
#     return output[:500].strip()

# def evaluate_reflection_block_sync(client: OpenAI, critique: str, reflection: str, original_query: str) -> int:
#     """Synchronous LLM judge to grade how accurately the agent analyzed its own mistake."""
#     prompt = f"""
#     You are evaluating the internal logic of a self-correcting AI agent.
#     The agent failed to answer this query: "{original_query}"
#     The System Judge provided this critique: "{critique}"
#     Before retrying, the agent wrote this reflection: "{reflection}"
    
#     Evaluate the reflection block on a scale of 1 to 10 based on:
#     1. Did it accurately identify the specific error mentioned in the critique?
#     2. Did it propose a concrete, logical fix?
    
#     Return ONLY a single integer from 1 to 10.
#     """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         match = re.search(r'\d+', response.choices[0].message.content)
#         return int(match.group()) if match else 0
#     except Exception as e:
#         print(f"⚠️ Judge Error: {e}")
#         return 0

# def process_self_healing_data(runs):
#     system_stats = defaultdict(lambda: {"initial_failures": 0, "successful_recoveries": 0})
#     node_stats = []
#     depth_distribution = defaultdict(lambda: defaultdict(int))
    
#     # Initialize OpenAI Client
#     api_key = os.getenv("OPENAI_API_KEY")
#     client = OpenAI(api_key=api_key) if api_key else None
    
#     # Load cached reflection scores
#     existing_results = {}
#     if RESULTS_FILE.exists():
#         with open(RESULTS_FILE, "r") as f:
#             existing_results = json.load(f)

#     for run in runs:
#         mode = run.get("ablation_mode", "UNKNOWN").upper()
#         if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
#         t = run.get("engine_telemetry", {})
        
#         # ── SYSTEM LEVEL ──
#         retries = t.get("global_retries", 0)
#         final_score = t.get("final_judge_score", 0)
#         if retries > 0:
#             system_stats[mode]["initial_failures"] += 1
#             if final_score >= 8:
#                 system_stats[mode]["successful_recoveries"] += 1

#         # ── NODE LEVEL ──
#         nodes = t.get("got_dag_nodes", [])
#         node_dict = {n["node_id"]: n for n in nodes}
        
#         for node in nodes:
#             if node.get("operator") == "refine":
#                 parent_ids = node.get("parent_ids", [])
#                 if not parent_ids: continue
                
#                 parent = node_dict.get(parent_ids[0])
#                 if not parent: continue
                
#                 parent_score = parent.get("quality_score", 0)
#                 refine_score = node.get("quality_score", 0)
#                 recovered = True if refine_score >= 0.75 else False
                
#                 depth = node.get("depth", 1) - parent.get("depth", 0)
#                 if recovered:
#                     depth_distribution[mode][f"{depth} Retry"] += 1
#                 else:
#                     depth_distribution[mode]["Failed to Recover"] += 1
                
#                 output_text = node.get("output", "")
#                 reflection_text = extract_reflection_text(output_text)
                
#                 reflection_score = 0
#                 if reflection_text:
#                     cache_key = str(hash(reflection_text))
#                     if cache_key in existing_results:
#                         reflection_score = existing_results[cache_key]
#                     elif client:
#                         critique = parent.get("judge_critique", "No critique provided.")
#                         query = parent.get("query", "Unknown query.")
#                         print(f"⚖️ Grading new reflection block...")
#                         reflection_score = evaluate_reflection_block_sync(client, critique, reflection_text, query)
#                         existing_results[cache_key] = reflection_score
                
#                 node_stats.append({
#                     "Mode": mode,
#                     "Pre-Critique (Failed Node)": parent_score,
#                     "Post-Critique (REFINE Node)": refine_score,
#                     "Reflection Score (1-10)": reflection_score
#                 })

#     # Save cache
#     with open(RESULTS_FILE, "w") as f:
#         json.dump(existing_results, f, indent=4)

#     return system_stats, pd.DataFrame(node_stats), depth_distribution

# def main():
#     print("🎨 Generating Publication-Ready Self-Healing Figure...")
    
#     runs = load_telemetry()
#     system_stats, df_nodes, depth_distribution = process_self_healing_data(runs)

#     if df_nodes.empty:
#         print("⚠️ Not enough data to generate Master Figure 5.")
#         return

#     # =========================================================================
#     # MASTER FIGURE 5: Autonomous Self-Healing Dynamics (2x2 Grid)
#     # =========================================================================
#     fig = plt.figure(figsize=(16, 12))
#     gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
#     plt_sns.set_theme(style="whitegrid")

#     # ── Panel A: System-Level Recovery Rate (Bar Chart) ──
#     ax1 = fig.add_subplot(gs[0, 0])
#     sys_plot_data = []
#     for mode in ["NONE", "RAW", "ADAPTIVE"]:
#         stats = system_stats.get(mode, {"initial_failures": 0, "successful_recoveries": 0})
#         failures = stats["initial_failures"]
#         recoveries = stats["successful_recoveries"]
#         rate = (recoveries / failures * 100) if failures > 0 else 0
#         sys_plot_data.append({"Mode": mode, "Recovery Rate (%)": rate})
        
#     df_sys = pd.DataFrame(sys_plot_data)
#     plt_sns.barplot(data=df_sys, x='Mode', y='Recovery Rate (%)', palette=COLORS, edgecolor='black', alpha=0.85, ax=ax1)
    
#     ax1.set_title("A. System-Level Reflexion Recovery Rate", fontweight='bold')
#     ax1.set_ylabel("Queries Recovered to Passing Score (%)")
#     ax1.set_ylim(0, 105)
#     for p in ax1.patches:
#         ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
#                      ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 3), textcoords='offset points')

#     # ── Panel B: Recovery Depth Distribution (Stacked Bar) ──
#     ax2 = fig.add_subplot(gs[0, 1])
#     df_depth = pd.DataFrame(depth_distribution).fillna(0).T
#     expected_cols = ["1 Retry", "2 Retry", "3 Retry", "Failed to Recover"]
#     for col in expected_cols:
#         if col not in df_depth.columns:
#             df_depth[col] = 0
#     df_depth = df_depth[expected_cols]
#     df_depth = df_depth.reindex(["NONE", "RAW", "ADAPTIVE"])

#     df_depth.plot(kind='bar', stacked=True, color=['#2DC653', '#F4A261', '#E07B39', '#D62828'], edgecolor='black', alpha=0.85, ax=ax2)
#     ax2.set_title('B. Recovery Depth Distribution', fontweight='bold')
#     ax2.set_ylabel('Number of Instances')
#     ax2.set_xlabel('')
#     ax2.tick_params(axis='x', rotation=0)
#     ax2.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')

#     # ── Panel C: Node-Level NQS Improvement (Grouped Boxplot) ──
#     ax3 = fig.add_subplot(gs[1, 0])
    
#     # Restructure data for grouped boxplot
#     df_melted = df_nodes.melt(id_vars=['Mode'], value_vars=['Pre-Critique (Failed Node)', 'Post-Critique (REFINE Node)'], 
#                               var_name='Stage', value_name='NQS Score (0.0 to 1.0)')
    
#     plt_sns.boxplot(data=df_melted, x="Stage", y="NQS Score (0.0 to 1.0)", hue="Mode", 
#                     palette=COLORS, linewidth=1.2, fliersize=0, ax=ax3)
    
#     # Overlay swarmplot for data points
#     plt_sns.swarmplot(data=df_melted, x="Stage", y="NQS Score (0.0 to 1.0)", hue="Mode", 
#                       dodge=True, color=".25", alpha=0.6, size=4, ax=ax3, legend=False)

#     ax3.set_title('C. Node-Level NQS Improvement via DAG Refinement', fontweight='bold')
#     ax3.set_ylabel('Node Quality Score (0.0 to 1.0)')
#     ax3.set_xlabel('')
#     ax3.set_ylim(-0.05, 1.05)
#     ax3.axhline(0.75, color='red', linestyle='--', alpha=0.7, label='Passing Threshold (0.75)')
    
#     # Fix legend
#     handles, labels = ax3.get_legend_handles_labels()
#     ax3.legend(handles[:3], labels[:3], title='Ablation Mode', loc='lower right')

#     # ── Panel D: Reflexion Quality Score (Violin Plot) ──
#     ax4 = fig.add_subplot(gs[1, 1])
#     df_rq = df_nodes[df_nodes["Reflection Score (1-10)"] > 0]
    
#     if not df_rq.empty:
#         plt_sns.violinplot(data=df_rq, x="Mode", y="Reflection Score (1-10)", 
#                            order=["NONE", "RAW", "ADAPTIVE"], palette=COLORS, inner="quartile", ax=ax4)
#         plt_sns.swarmplot(data=df_rq, x="Mode", y="Reflection Score (1-10)", 
#                           order=["NONE", "RAW", "ADAPTIVE"], color=".25", alpha=0.5, size=5, ax=ax4)
                          
#         ax4.set_title('D. Reflexion Quality Score (LLM-as-a-Judge)', fontweight='bold')
#         ax4.set_ylabel('Reflection Score (1 to 10)')
#         ax4.set_xlabel('')
#         ax4.set_ylim(0, 11)
#     else:
#         ax4.text(0.5, 0.5, "Error: OPENAI_API_KEY missing or no <reflection> blocks found.", 
#                  ha='center', va='center', fontsize=12, color='red')
#         ax4.set_title('D. Reflexion Quality Score (LLM-as-a-Judge)', fontweight='bold')

#     plt.suptitle("Autonomous Self-Healing Dynamics via Graph-of-Thoughts (GoT)", fontweight='bold', fontsize=16, y=0.95)
    
#     output_path = VIZ_DIR / "PUBLISH_Fig5_Self_Healing.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Generated: {output_path.name}")

# if __name__ == "__main__":
#     main()


import os
import json
import re
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np
from matplotlib.gridspec import GridSpec

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = ROOT_DIR / "reflexion_quality_results.json"

COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists(): return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

def extract_reflection_text(output: str) -> str:
    if not output: return ""
    match = re.search(r'<reflection>(.*?)</reflection>', output, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'(?:###|\*\*|#)?\s*Reflection[:\s]*(.*?)(?:\n\n|\Z)', output, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return output[:500].strip()

def evaluate_reflection_block_sync(client: OpenAI, critique: str, reflection: str, original_query: str) -> int:
    prompt = f"""
    You are evaluating the internal logic of a self-correcting AI agent.
    The agent failed to answer this query: "{original_query}"
    The System Judge provided this critique: "{critique}"
    Before retrying, the agent wrote this reflection: "{reflection}"
    Evaluate the reflection block on a scale of 1 to 10 based on:
    1. Did it accurately identify the specific error mentioned in the critique?
    2. Did it propose a concrete, logical fix?
    Return ONLY a single integer from 1 to 10.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        match = re.search(r'\d+', response.choices[0].message.content)
        return int(match.group()) if match else 0
    except: return 0

def process_self_healing_data(runs):
    system_stats = defaultdict(lambda: {"initial_failures": 0, "successful_recoveries": 0})
    node_stats = []
    depth_distribution = defaultdict(lambda: defaultdict(int))
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None
    
    existing_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f: existing_results = json.load(f)

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        retries = t.get("global_retries", 0)
        final_score = t.get("final_judge_score", 0)
        if retries > 0:
            system_stats[mode]["initial_failures"] += 1
            if final_score >= 8: system_stats[mode]["successful_recoveries"] += 1

        nodes = t.get("got_dag_nodes", [])
        node_dict = {n["node_id"]: n for n in nodes}
        
        for node in nodes:
            if node.get("operator") == "refine":
                parent_ids = node.get("parent_ids", [])
                if not parent_ids: continue
                parent = node_dict.get(parent_ids[0])
                if not parent: continue
                
                parent_score = parent.get("quality_score", 0)
                refine_score = node.get("quality_score", 0)
                recovered = True if refine_score >= 0.75 else False
                
                depth = node.get("depth", 1) - parent.get("depth", 0)
                if recovered: depth_distribution[mode][f"{depth} Retry"] += 1
                else: depth_distribution[mode]["Failed to Recover"] += 1
                
                output_text = node.get("output", "")
                reflection_text = extract_reflection_text(output_text)
                
                reflection_score = 0
                if reflection_text:
                    cache_key = str(hash(reflection_text))
                    if cache_key in existing_results:
                        reflection_score = existing_results[cache_key]
                    elif client:
                        print(f"⚖️ Grading new reflection block...")
                        reflection_score = evaluate_reflection_block_sync(client, parent.get("judge_critique", ""), reflection_text, parent.get("query", ""))
                        existing_results[cache_key] = reflection_score
                
                node_stats.append({"Mode": mode, "Pre-Critique (Failed)": parent_score, "Post-Critique (REFINE)": refine_score, "Reflection Score": reflection_score})

    with open(RESULTS_FILE, "w") as f: json.dump(existing_results, f, indent=4)
    return system_stats, pd.DataFrame(node_stats), depth_distribution

def main():
    print("🎨 Generating Phase 2: Self-Healing Figures (Final Publication)...")
    runs = load_telemetry()
    system_stats, df_nodes, depth_distribution = process_self_healing_data(runs)
    if df_nodes.empty: return

    # =========================================================================
    # MASTER FIGURE 5: Autonomous Self-Healing Dynamics (2x3 Grid)
    # =========================================================================
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    plt_sns.set_theme(style="whitegrid")

    # ── Panel A: System-Level Recovery Rate ──
    ax1 = fig.add_subplot(gs[0, 0])
    sys_plot_data = [{"Mode": m, "Recovery Rate (%)": (system_stats[m]["successful_recoveries"] / system_stats[m]["initial_failures"] * 100) if system_stats[m]["initial_failures"] > 0 else 0} for m in ["NONE", "RAW", "ADAPTIVE"]]
    sns_ax1 = plt_sns.barplot(data=pd.DataFrame(sys_plot_data), x='Mode', y='Recovery Rate (%)', palette=COLORS, edgecolor='black', ax=ax1)
    ax1.set_title("A. System Recovery Rate\n(Among Failed Queries Only)", fontweight='bold')
    ax1.set_ylim(0, 105); ax1.set_xlabel("")
    for p in sns_ax1.patches: ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ── Panel B: Recovery Depth ──
    ax2 = fig.add_subplot(gs[0, 1:3])
    df_depth = pd.DataFrame(depth_distribution).fillna(0).T
    expected_cols = ["1 Retry", "2 Retry", "3 Retry", "Failed to Recover"]
    for col in expected_cols:
        if col not in df_depth.columns: df_depth[col] = 0
    df_depth = df_depth[expected_cols].reindex(["NONE", "RAW", "ADAPTIVE"])
    df_depth.plot(kind='bar', stacked=True, color=['#2DC653', '#F4A261', '#E07B39', '#D62828'], edgecolor='black', alpha=0.85, ax=ax2)
    ax2.set_title('B. Recovery Depth Distribution', fontweight='bold')
    ax2.set_ylabel('Number of Instances'); ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=0); ax2.legend(title='Outcome', loc='upper right')

    # ── Panel C: Node-Level NQS Boxplot ──
    ax3 = fig.add_subplot(gs[1, 0])
    df_melted = df_nodes.melt(id_vars=['Mode'], value_vars=['Pre-Critique (Failed)', 'Post-Critique (REFINE)'], var_name='Stage', value_name='NQS Score (0-1)')
    plt_sns.boxplot(data=df_melted, x="Stage", y="NQS Score (0-1)", hue="Mode", palette=COLORS, linewidth=1.2, fliersize=0, ax=ax3)
    ax3.set_title('C. Node NQS Improvement', fontweight='bold')
    ax3.set_ylim(-0.05, 1.05); ax3.set_xlabel(''); ax3.set_ylabel('Node Quality Score')
    ax3.axhline(0.75, color='red', linestyle='--', alpha=0.7)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[:3], labels[:3], title='Mode', loc='lower right')

    # ── Panel D: Reflexion Quality ──
    ax4 = fig.add_subplot(gs[1, 1])
    df_rq = df_nodes[df_nodes["Reflection Score"] > 0]
    plt_sns.violinplot(data=df_rq, x="Mode", y="Reflection Score", order=["NONE", "RAW", "ADAPTIVE"], palette=COLORS, inner="quartile", ax=ax4)
    ax4.set_title('D. Reflexion Diagnostic Accuracy', fontweight='bold')
    ax4.set_ylim(0, 11); ax4.set_xlabel(''); ax4.set_ylabel('LLM Judge Score (1-10)')

    # ── Panel E: Reflexion vs NQS Scatter (NEW!) ──
    ax5 = fig.add_subplot(gs[1, 2])
    plt_sns.scatterplot(data=df_rq, x="Reflection Score", y="Post-Critique (REFINE)", hue="Mode", palette=COLORS, s=60, alpha=0.7, edgecolor="black", ax=ax5)
    plt_sns.regplot(data=df_rq, x="Reflection Score", y="Post-Critique (REFINE)", scatter=False, color="black", line_kws={"linestyle":"--"}, ax=ax5)
    ax5.set_title('E. Reflection Quality Predicts Recovery', fontweight='bold')
    ax5.set_xlabel('Reflection Quality Score (1-10)')
    ax5.set_ylabel('Recovered NQS (0-1)')
    ax5.set_ylim(-0.05, 1.05)
    ax5.axhline(0.75, color='red', linestyle='--', alpha=0.7)
    ax5.get_legend().remove()

    plt.suptitle("Autonomous Self-Healing Dynamics via Graph-of-Thoughts (GoT)", fontweight='bold', fontsize=16, y=0.95)
    output_path = VIZ_DIR / "PUBLISH_Fig5_Self_Healing.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path.name}")

if __name__ == "__main__":
    main()