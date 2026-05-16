# import os
# import json
# import asyncio
# import re
# from pathlib import Path
# from collections import defaultdict
# from openai import AsyncOpenAI
# from dotenv import load_dotenv
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import numpy as np

# # Load environment variables
# load_dotenv()

# ROOT_DIR = Path(os.path.dirname(__file__)).parent
# TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
# RESULTS_FILE = ROOT_DIR / "llm_judge_qualitative_results.json"
# VIZ_DIR = ROOT_DIR / "visualizations"
# VIZ_DIR.mkdir(exist_ok=True)

# # ── 1. Data Loader ──────────────────────────────────────────────────────────
# def load_telemetry():
#     runs = []
#     if not TELEMETRY_DIR.exists():
#         print(f"❌ Error: {TELEMETRY_DIR} not found.")
#         return []
#     for path in TELEMETRY_DIR.glob("run_*.json"):
#         with open(path, "r") as f:
#             try:
#                 runs.append(json.load(f))
#             except:
#                 pass
#     return runs

# # ── 2. LLM Judge Setup ──────────────────────────────────────────────────────
# async def evaluate_final_answer(client: AsyncOpenAI, query: str, answer: str) -> dict:
#     prompt = f"""
#     You are an expert scientific AI evaluator.
#     Please evaluate the following AI-generated answer to a bioinformatics/computational biology query.
    
#     User Query: "{query}"
    
#     AI Answer:
#     "{answer}"
    
#     Evaluate the answer on the following 3 criteria (score 1 to 10):
#     1. Helpfulness: Does it directly solve the user's overarching problem?
#     2. Informativeness: Is the technical depth sufficient, accurate, and educational?
#     3. Actionability: If code or commands are provided, are they ready to use? If experts are asked for, are real names provided?
    
#     Return ONLY a valid JSON object in this exact format:
#     {{
#         "helpfulness": 8,
#         "informativeness": 7,
#         "actionability": 9,
#         "critique": "A brief 1-sentence explanation of the scores."
#     }}
#     """
#     try:
#         response = await client.chat.completions.create(
#             model="gpt-5.4", # Cost effective but highly accurate for scoring
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         text = response.choices[0].message.content.strip()
#         text = re.sub(r'^```json|```$', '', text, flags=re.MULTILINE).strip()
#         return json.loads(text)
#     except Exception as e:
#         print(f"⚠️ Judge Error: {e}")
#         return {"helpfulness": 0, "informativeness": 0, "actionability": 0, "critique": "Error"}

# # ── 3. Main Evaluation Loop ─────────────────────────────────────────────────
# async def main():
#     print("╔══════════════════════════════════════════════════════════╗")
#     print("║        STARTING OFFLINE LLM-AS-A-JUDGE EVALUATION        ║")
#     print("╚══════════════════════════════════════════════════════════╝")
    
#     client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     if not client.api_key:
#         print("❌ OPENAI_API_KEY missing. Please add it to your .env file.")
#         return

#     runs = load_telemetry()
#     if not runs:
#         return
        
#     print(f"📦 Loaded {len(runs)} runs for evaluation...")

#     # Group runs to avoid redundant API calls if we already judged this dataset
#     existing_results = {}
#     if RESULTS_FILE.exists():
#         with open(RESULTS_FILE, "r") as f:
#             existing_results = json.load(f)

#     evaluation_data = []
#     agent_scores = defaultdict(lambda: defaultdict(list)) # agent -> mode -> [scores]

#     for run in runs:
#         run_id = run.get("run_id")
#         mode = run.get("ablation_mode", "unknown").upper()
#         query = run.get("query", "Unknown Query")
#         t = run.get("engine_telemetry", {})
#         final_answer = t.get("final_answer", "")
        
#         # --- AGENT LEVEL AGGREGATION ---
#         # Aggregate the live NQS (Node Quality Score) for each agent type
#         for node in t.get("got_dag_nodes", []):
#             agent_type = node.get("agent", "unknown").upper()
#             score = node.get("quality_score")
#             if score is not None:
#                 # Convert 0.0-1.0 score to 0-10 scale for consistency
#                 agent_scores[agent_type][mode].append(score * 10)

#         # --- SYSTEM LEVEL EVALUATION ---
#         if run_id in existing_results:
#             print(f"⏭️  Skipping {run_id} (Already Judged)")
#             evaluation_data.append(existing_results[run_id])
#             continue
            
#         if not final_answer:
#             continue

#         print(f"⚖️  Judging Run {run_id} ({mode})...")
#         scores = await evaluate_final_answer(client, query, final_answer)
        
#         record = {
#             "run_id": run_id,
#             "mode": mode,
#             "query": query,
#             "helpfulness": scores.get("helpfulness", 0),
#             "informativeness": scores.get("informativeness", 0),
#             "actionability": scores.get("actionability", 0),
#             "critique": scores.get("critique", "")
#         }
#         evaluation_data.append(record)
#         existing_results[run_id] = record

#     # Save results to avoid re-running API calls
#     with open(RESULTS_FILE, "w") as f:
#         json.dump(existing_results, f, indent=4)
        
#     print(f"\n💾 Saved qualitative evaluations to {RESULTS_FILE.name}")

#     # ── 4. Graph Generation ─────────────────────────────────────────────────
#     print("📊 Generating Qualitative Graphs...")
    
#     # Graph A: System Level Qualitative Dimensions
#     df = pd.DataFrame(evaluation_data)
#     # Filter out 'NONE' mode if we just want to compare RAW vs ADAPTIVE
#     df = df[df['mode'].isin(['RAW', 'ADAPTIVE'])]
    
#     if not df.empty:
#         df_melted = df.melt(id_vars=['mode'], value_vars=['helpfulness', 'informativeness', 'actionability'], 
#                             var_name='Metric', value_name='Score')
        
#         plt.figure(figsize=(9, 6))
#         plt_sns.set_theme(style="whitegrid")
#         ax = plt_sns.barplot(data=df_melted, x='Metric', y='Score', hue='mode', 
#                              palette={"RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}, edgecolor='black', alpha=0.85)
        
#         plt.title('Qualitative Performance by Ablation Mode (LLM-as-a-Judge)')
#         plt.ylabel('Mean Score (0-10)')
#         plt.xlabel('')
#         plt.ylim(0, 10.5)
#         plt.legend(title='Mode')
        
#         # Add labels on top of bars
#         for p in ax.patches:
#             if p.get_height() > 0:
#                 ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
#                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', xytext=(0, 3),
#                             textcoords='offset points')
                
#         plt.tight_layout()
#         plt.savefig(VIZ_DIR / "fig5_qualitative_dimensions.png")
#         print("✅ Generated: fig5_qualitative_dimensions.png")

#     # Graph B: Agent Level Performance
#     agent_plot_data = []
#     valid_agents = ["RAG", "CODE", "CLOUD", "SCHOLAR"]
    
#     for agent in valid_agents:
#         for mode in ["RAW", "ADAPTIVE"]:
#             scores = agent_scores.get(agent, {}).get(mode, [])
#             if scores:
#                 agent_plot_data.append({
#                     "Agent": agent,
#                     "Mode": mode,
#                     "Mean NQS": np.mean(scores)
#                 })
                
#     if agent_plot_data:
#         df_agent = pd.DataFrame(agent_plot_data)
        
#         plt.figure(figsize=(9, 6))
#         ax2 = plt_sns.barplot(data=df_agent, x='Agent', y='Mean NQS', hue='Mode', 
#                               palette={"RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}, edgecolor='black', alpha=0.85)
        
#         plt.title('Individual Agent Performance: Mean Node Quality Score (NQS)')
#         plt.ylabel('Mean NQS (Scaled 0-10)')
#         plt.xlabel('Agent Type')
#         plt.ylim(0, 10.5)
#         plt.legend(title='Mode')
        
#         for p in ax2.patches:
#             if p.get_height() > 0:
#                 ax2.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', xytext=(0, 3),
#                              textcoords='offset points')
                             
#         plt.tight_layout()
#         plt.savefig(VIZ_DIR / "fig6_agent_performance.png")
#         print("✅ Generated: fig6_agent_performance.png")

# if __name__ == "__main__":
#     asyncio.run(main())


import os
import json
import asyncio
import re
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

load_dotenv()

ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
RESULTS_FILE = ROOT_DIR / "llm_judge_qualitative_results.json"
VIZ_DIR = ROOT_DIR / "visualizations" / "final"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"NONE": "#6C757D", "RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists(): return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

async def evaluate_final_answer(client: AsyncOpenAI, query: str, answer: str) -> dict:
    prompt = f"""
    Evaluate the following AI-generated answer to a computational biology query.
    User Query: "{query}"
    AI Answer: "{answer}"
    Evaluate the answer on the following 3 criteria (score 1 to 10):
    1. Helpfulness: Does it directly solve the user's overarching problem?
    2. Informativeness: Is the technical depth sufficient and accurate?
    3. Actionability: If code/commands/experts are provided, are they ready to use?
    Return ONLY a valid JSON object:
    {{"helpfulness": 8, "informativeness": 7, "actionability": 9}}
    """
    try:
        response = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        text = re.sub(r'^```json|```$', '', response.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
        return json.loads(text)
    except: return {"helpfulness": 0, "informativeness": 0, "actionability": 0}

async def main():
    print("🎨 Generating Phase 3: Qualitative Dimensions (Offline LLM Judge)...")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        print("❌ OPENAI_API_KEY missing. Cannot run Qualitative Evaluation.")
        return

    runs = load_telemetry()
    if not runs: return

    existing_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f: existing_results = json.load(f)

    evaluation_data = []

    for run in runs:
        run_id = run.get("run_id")
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["NONE", "RAW", "ADAPTIVE"]: continue
        
        query = run.get("query", "Unknown Query")
        final_answer = run.get("engine_telemetry", {}).get("final_answer", "")
        if not final_answer: continue

        if run_id in existing_results:
            evaluation_data.append(existing_results[run_id])
            continue
            
        print(f"⚖️  Judging Run {run_id} ({mode})...")
        scores = await evaluate_final_answer(client, query, final_answer)
        
        record = {
            "run_id": run_id, "mode": mode, "query": query,
            "helpfulness": scores.get("helpfulness", 0),
            "informativeness": scores.get("informativeness", 0),
            "actionability": scores.get("actionability", 0)
        }
        evaluation_data.append(record)
        existing_results[run_id] = record

    with open(RESULTS_FILE, "w") as f: json.dump(existing_results, f, indent=4)

    df = pd.DataFrame(evaluation_data)
    if df.empty: return

    df_melted = df.melt(id_vars=['mode'], value_vars=['helpfulness', 'informativeness', 'actionability'], var_name='Metric', value_name='Score')

    # =========================================================================
    # PUBLISH FIGURE: Qualitative Performance
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt_sns.set_theme(style="whitegrid")
    ax = plt_sns.barplot(data=df_melted, x='Metric', y='Score', hue='mode', hue_order=["NONE", "RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', alpha=0.85)
    
    plt.title('Qualitative System Performance by Ablation Mode (LLM-as-a-Judge)', fontweight='bold')
    plt.ylabel('Mean Score (0-10)')
    plt.xlabel('')
    plt.ylim(0, 10.5)
    plt.legend(title='Ablation Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', xytext=(0, 3), textcoords='offset points')
            
    plt.tight_layout()
    output_path = VIZ_DIR / "PUBLISH_Fig8_Qualitative_Dimensions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_path.name}")

if __name__ == "__main__":
    asyncio.run(main())