import os
import json
import asyncio
import re
from pathlib import Path
from collections import defaultdict
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

# Load environment variables
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
RESULTS_FILE = ROOT_DIR / "reflexion_quality_results.json"

COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

# ── 1. Data Loader ───────────────────────────────────────────────────────────
def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists():
        print(f"❌ Error: {TELEMETRY_DIR} not found.")
        return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

# ── 2. LLM Judge & Text Extraction ───────────────────────────────────────────
async def evaluate_reflection_block(client: AsyncOpenAI, critique: str, reflection: str, original_query: str) -> int:
    """Uses GPT-4o-mini to grade how accurately the agent analyzed its own mistake."""
    prompt = f"""
    You are evaluating the internal logic of a self-correcting AI agent.
    
    The agent failed to answer this query: "{original_query}"
    The System Judge provided this critique: "{critique}"
    
    Before retrying, the agent wrote this reflection to analyze its mistake:
    "{reflection}"
    
    Evaluate the reflection block on a scale of 1 to 10 based on:
    1. Did it accurately identify the specific error mentioned in the critique?
    2. Did it propose a concrete, logical fix for the next attempt?
    
    Return ONLY a single integer from 1 to 10. No other text.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        # Extract the first number found in the response
        match = re.search(r'\d+', response.choices[0].message.content)
        return int(match.group()) if match else 0
    except Exception as e:
        print(f"⚠️ Judge Error: {e}")
        return 0

def extract_reflection_text(output: str) -> str:
    """Robust extraction: Looks for XML tags, falls back to markdown headers, or just grabs the first 500 chars."""
    if not output: return ""
    
    # 1. Try strict XML tags
    match = re.search(r'<reflection>(.*?)</reflection>', output, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    
    # 2. Try markdown headers (e.g., ### Reflection)
    match = re.search(r'(?:###|\*\*|#)?\s*Reflection[:\s]*(.*?)(?:\n\n|\Z)', output, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    
    # 3. Fallback: The LLM probably just started writing its reflection at the top. Grab the first 500 characters.
    return output[:500].strip()

# ── 3. Main Metrics Extraction & Execution ───────────────────────────────────
async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ADVANCED REFLEXION & RECOVERY DYNAMICS ANALYSIS      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        print("❌ OPENAI_API_KEY missing. Cannot evaluate Reflexion blocks.")
        return

    runs = load_telemetry()
    if not runs: return

    # Caches LLM calls
    existing_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            existing_results = json.load(f)

    # Data structures for all 4 graphs
    system_stats = defaultdict(lambda: {"initial_failures": 0, "successful_recoveries": 0})
    node_stats = []
    depth_distribution = defaultdict(lambda: defaultdict(int)) # Mode -> Depth -> Count
    
    print("📦 Processing nodes and evaluating <reflection> blocks...")

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        t = run.get("engine_telemetry", {})
        
        # ── A. SYSTEM-LEVEL REFLEXION (Global Diagnostician) ──
        retries = t.get("global_retries", 0)
        final_score = t.get("final_judge_score", 0)
        
        if retries > 0:
            system_stats[mode]["initial_failures"] += 1
            if final_score >= 8:
                system_stats[mode]["successful_recoveries"] += 1

        # ── B. NODE-LEVEL REFLEXION (GoT DAG 'REFINE' Operator) ──
        nodes = t.get("got_dag_nodes", [])
        node_dict = {n["node_id"]: n for n in nodes}
        
        for node in nodes:
            if node.get("operator") == "refine":
                parent_ids = node.get("parent_ids", [])
                if not parent_ids: continue
                
                parent = node_dict.get(parent_ids[0])
                if not parent: continue
                
                # 1. Recovery Metrics (NQS Deltas)
                parent_score = parent.get("quality_score", 0)
                refine_score = node.get("quality_score", 0)
                recovered = True if refine_score >= 0.75 else False
                
                # 2. Recovery Depth
                depth = node.get("depth", 1) - parent.get("depth", 0)
                if recovered:
                    depth_distribution[mode][f"{depth} Retry"] += 1
                else:
                    depth_distribution[mode]["Failed to Recover"] += 1
                
                # 3. Reflexion Quality Evaluation
                output_text = node.get("output", "")
                reflection_text = extract_reflection_text(output_text)
                
                reflection_score = 0
                if reflection_text:
                    cache_key = str(hash(reflection_text))
                    if cache_key in existing_results:
                        reflection_score = existing_results[cache_key]
                    else:
                        critique = parent.get("judge_critique", "No critique provided.")
                        query = parent.get("query", "Unknown query.")
                        reflection_score = await evaluate_reflection_block(client, critique, reflection_text, query)
                        existing_results[cache_key] = reflection_score
                
                node_stats.append({
                    "Mode": mode,
                    "Agent": node.get("agent", "UNKNOWN").upper(),
                    "Parent NQS": parent_score,
                    "Refined NQS": refine_score,
                    "Delta NQS": refine_score - parent_score,
                    "Recovered": recovered,
                    "Reflection Score (1-10)": reflection_score
                })

    with open(RESULTS_FILE, "w") as f:
        json.dump(existing_results, f, indent=4)

    df_nodes = pd.DataFrame(node_stats)
    
    if df_nodes.empty:
        print("⚠️ No REFINE nodes found in telemetry.")
        return

    # --- TERMINAL PRINTOUT ---
    print("\n🔬 SYSTEM-LEVEL RECOVERY RATE (Global Diagnostician)")
    for mode in ["NONE", "RAW", "ADAPTIVE"]:
        stats = system_stats.get(mode, {"initial_failures": 0, "successful_recoveries": 0})
        failures = stats["initial_failures"]
        recoveries = stats["successful_recoveries"]
        rate = (recoveries / failures * 100) if failures > 0 else 0
        print(f"  ► {mode:<8} : {recoveries}/{failures} recoveries ({rate:.1f}%)")

    print("\n🔬 NODE-LEVEL REFLEXION METRICS (GoT DAG REFINE Operator)")
    for mode in ["NONE", "RAW", "ADAPTIVE"]:
        df_mode = df_nodes[df_nodes["Mode"] == mode]
        if df_mode.empty: continue
        avg_delta = df_mode["Delta NQS"].mean()
        recovery_rate = df_mode["Recovered"].mean() * 100
        print(f"  ► {mode:<8} : Avg NQS Boost = {avg_delta:+.3f} | Recovery Rate = {recovery_rate:.1f}%")

    print("\n🔬 RECOVERY DEPTH DISTRIBUTION")
    for mode in ["NONE", "RAW", "ADAPTIVE"]:
        print(f"  ► {mode}: {dict(depth_distribution[mode])}")

    print("\n🔬 MEAN REFLEXION QUALITY SCORE (1-10)")
    rq_stats = df_nodes[df_nodes["Reflection Score (1-10)"] > 0].groupby("Mode")["Reflection Score (1-10)"].mean()
    print(rq_stats.round(2).to_string())

    print("\n📊 Generating All Reflexion Graphs...")

    # ── Graph 1: System-Level Recovery Rate (Bar Chart) ──
    sys_plot_data =[]
    for mode in ["NONE", "RAW", "ADAPTIVE"]:
        stats = system_stats.get(mode, {"initial_failures": 0, "successful_recoveries": 0})
        failures = stats["initial_failures"]
        recoveries = stats["successful_recoveries"]
        rate = (recoveries / failures * 100) if failures > 0 else 0
        sys_plot_data.append({"Mode": mode, "Recovery Rate (%)": rate})
        
    df_sys = pd.DataFrame(sys_plot_data)
    
    plt.figure(figsize=(7, 5))
    plt_sns.set_theme(style="whitegrid")
    ax = plt_sns.barplot(data=df_sys, x='Mode', y='Recovery Rate (%)', palette=COLORS, edgecolor='black', alpha=0.85)
    
    plt.title('System-Level Reflexion Recovery Rate\n(Turning Failing Queries into Passing Queries)', fontweight='bold')
    plt.ylabel('Recovery Rate (%)')
    plt.ylim(0, 105)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 3),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig09_system_recovery_rate.png")

    # ── Graph 2: Node-Level NQS Delta (Slopegraph) ──
    plt.figure(figsize=(8, 6))
    for _, row in df_nodes.iterrows():
        color = COLORS.get(row['Mode'], 'gray')
        jitter = np.random.uniform(-0.1, 0.1)
        plt.plot([1 + jitter, 2 + jitter], [row['Parent NQS'], row['Refined NQS']], 
                 marker='o', color=color, alpha=0.5, linewidth=2, markersize=8)

    plt.xticks([1, 2],['Pre-Critique (Failed Node)', 'Post-Critique (REFINE Node)'], fontweight='bold')
    plt.ylabel('Node Quality Score (0.0 to 1.0)')
    plt.title('Node-Level Reflexion: NQS Improvement via DAG Refinement', fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.axhline(0.75, color='red', linestyle='--', alpha=0.7, label='Passing Threshold (0.75)')
    
    from matplotlib.patches import Patch
    legend_elements =[Patch(facecolor=COLORS[m], label=m, alpha=0.7) for m in ["NONE", "RAW", "ADAPTIVE"]]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Passing Threshold'))
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig10_node_nqs_improvement.png")

    # ── Graph 3: Recovery Depth Distribution (Stacked Bar) ──
    df_depth = pd.DataFrame(depth_distribution).fillna(0).T
    expected_cols = ["1 Retry", "2 Retry", "3 Retry", "Failed to Recover"]
    for col in expected_cols:
        if col not in df_depth.columns:
            df_depth[col] = 0
    df_depth = df_depth[expected_cols]

    plt.figure(figsize=(9, 6))
    df_depth.plot(kind='bar', stacked=True, color=['#2DC653', '#F4A261', '#E07B39', '#D62828'], edgecolor='black', alpha=0.85, ax=plt.gca())
    
    plt.title('Recovery Depth Distribution\n(How many retries to achieve NQS ≥ 0.75?)', fontweight='bold')
    plt.ylabel('Number of Instances')
    plt.xlabel('Ablation Mode')
    plt.xticks(rotation=0)
    plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig23_recovery_depth_distribution.png")

    # ── Graph 4: Reflexion Quality Score (Violin Plot) ──
    df_rq = df_nodes[df_nodes["Reflection Score (1-10)"] > 0]
    
    if not df_rq.empty:
        plt.figure(figsize=(8, 6))
        ax2 = plt_sns.violinplot(data=df_rq, x="Mode", y="Reflection Score (1-10)", 
                                 order=["NONE", "RAW", "ADAPTIVE"], palette=COLORS, inner="quartile")
        plt_sns.swarmplot(data=df_rq, x="Mode", y="Reflection Score (1-10)", 
                          order=["NONE", "RAW", "ADAPTIVE"], color=".25", alpha=0.5, size=6)
                          
        plt.title('Reflexion Quality Score (LLM-as-a-Judge)\nHow accurately did the agent diagnose its own mistake?', fontweight='bold')
        plt.ylabel('Reflection Quality Score (1 to 10)')
        plt.ylim(0, 11)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "fig24_reflexion_quality_score.png")
        print("✅ Generated all 4 Reflexion Graphs successfully!")
    else:
        print("⚠️ No valid reflection text found to generate Figure 24.")

if __name__ == "__main__":
    asyncio.run(main())


# ╔══════════════════════════════════════════════════════════╗
# ║     ADVANCED REFLEXION & RECOVERY DYNAMICS ANALYSIS      ║
# ╚══════════════════════════════════════════════════════════╝
# 📦 Processing nodes and evaluating <reflection> blocks...

# 🔬 SYSTEM-LEVEL RECOVERY RATE (Global Diagnostician)
#   ► NONE     : 1/2 recoveries (50.0%)
#   ► RAW      : 2/3 recoveries (66.7%)
#   ► ADAPTIVE : 1/4 recoveries (25.0%)

# 🔬 NODE-LEVEL REFLEXION METRICS (GoT DAG REFINE Operator)
#   ► NONE     : Avg NQS Boost = +0.900 | Recovery Rate = 100.0%
#   ► RAW      : Avg NQS Boost = +0.778 | Recovery Rate = 100.0%
#   ► ADAPTIVE : Avg NQS Boost = +0.774 | Recovery Rate = 100.0%

# 🔬 RECOVERY DEPTH DISTRIBUTION
#   ► NONE: {'1 Retry': 4}
#   ► RAW: {'1 Retry': 6}
#   ► ADAPTIVE: {'1 Retry': 7}

# 🔬 MEAN REFLEXION QUALITY SCORE (1-10)
# Series([], )