import os
import json
import asyncio
import re
import random
from pathlib import Path
from collections import defaultdict
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("⚠️ scipy not installed. Run `pip install scipy` for formal statistical p-values.")
    wilcoxon = None

# Load environment variables
load_dotenv()

ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
RESULTS_FILE = ROOT_DIR / "pairwise_eval_results.json"
VIZ_DIR = ROOT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# ── 1. Data Loader & Pair Matcher ────────────────────────────────────────────
def load_and_pair_telemetry():
    """Loads telemetry and groups runs by Query so we can compare RAW vs ADAPTIVE."""
    runs = []
    if not TELEMETRY_DIR.exists():
        print(f"❌ Error: {TELEMETRY_DIR} not found.")
        return {}
        
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try:
                runs.append(json.load(f))
            except:
                pass

    # Group by query
    pairs = defaultdict(dict)
    for run in runs:
        mode = run.get("ablation_mode", "").upper()
        query = run.get("query", "Unknown Query")
        t = run.get("engine_telemetry", {})
        
        if mode in ["RAW", "ADAPTIVE"] and t.get("final_answer"):
            pairs[query][mode] = {
                "run_id": run.get("run_id"),
                "answer": t.get("final_answer"),
                "gts": t.get("final_judge_score", 0)
            }
            
    valid_pairs = {q: data for q, data in pairs.items() if "RAW" in data and "ADAPTIVE" in data}
    print(f"📦 Found {len(valid_pairs)} valid query pairs for A/B testing.")
    return valid_pairs

# ── 2. LLM Pairwise Judge ───────────────────────────────────────────────────
async def evaluate_pair(client: AsyncOpenAI, query: str, answer_raw: str, answer_adaptive: str) -> dict:
    """Performs a blinded, randomized A/B comparison using an LLM Judge."""
    
    is_raw_a = random.choice([True, False])
    
    if is_raw_a:
        ans_a, ans_b = answer_raw, answer_adaptive
    else:
        ans_a, ans_b = answer_adaptive, answer_raw

    prompt = f"""
    You are an expert scientific AI evaluator.
    Please compare two AI-generated answers to the following bioinformatics query.
    
    User Query: "{query}"
    
    [ANSWER A]
    {ans_a}
    
    [ANSWER B]
    {ans_b}
    
    Evaluate which answer is better based on accuracy, completeness, and lack of hallucinations.
    Determine the winner ("A", "B", or "TIE") and the margin of victory on a scale of 1 to 5.
    Margin Scale:
    1 = Marginally better
    3 = Clearly better
    5 = Crushing victory (the other answer is fundamentally broken or hallucinated)
    If it's a TIE, margin MUST be 0.
    
    Return ONLY a valid JSON object in this exact format:
    {{
        "winner": "A", 
        "margin": 3,
        "reason": "A brief 1-sentence explanation of why the winner was chosen."
    }}
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r'^```json|```$', '', text, flags=re.MULTILINE).strip()
        result = json.loads(text)
        
        llm_winner = result.get("winner", "TIE").upper()
        if llm_winner == "TIE":
            actual_winner = "TIE"
        elif (llm_winner == "A" and is_raw_a) or (llm_winner == "B" and not is_raw_a):
            actual_winner = "RAW"
        else:
            actual_winner = "ADAPTIVE"
            
        return {
            "winner": actual_winner,
            "margin": result.get("margin", 0) if actual_winner != "TIE" else 0,
            "reason": result.get("reason", "No reason provided.")
        }
    except Exception as e:
        print(f"⚠️ Judge Error: {e}")
        return {"winner": "TIE", "margin": 0, "reason": "Error parsing LLM judge."}

# ── 3. Main Execution & Stats ───────────────────────────────────────────────
async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        STARTING PAIRWISE A/B ABLATION EVALUATION         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        print("❌ OPENAI_API_KEY missing. Please add it to your .env file.")
        return

    pairs = load_and_pair_telemetry()
    if not pairs: return

    existing_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            existing_results = json.load(f)

    evaluation_data = []
    gts_raw_list = []
    gts_adaptive_list = []

    for idx, (query, data) in enumerate(pairs.items()):
        gts_raw_list.append(data["RAW"]["gts"])
        gts_adaptive_list.append(data["ADAPTIVE"]["gts"])
        
        cache_key = str(hash(query))
        if cache_key in existing_results:
            print(f"⏭️  Skipping Pair {idx+1} (Already Judged)")
            record = existing_results[cache_key]
            record["query_short"] = f"Q{idx+1}: {query[:30]}..."
            evaluation_data.append(record)
            continue
            
        print(f"⚖️  Judging Pair {idx+1}/{len(pairs)}...")
        eval_res = await evaluate_pair(client, query, data["RAW"]["answer"], data["ADAPTIVE"]["answer"])
        
        record = {
            "query": query,
            "query_short": f"Q{idx+1}: {query[:30]}...",
            "winner": eval_res["winner"],
            "margin": eval_res["margin"],
            "reason": eval_res["reason"],
            "gts_raw": data["RAW"]["gts"],
            "gts_adaptive": data["ADAPTIVE"]["gts"]
        }
        evaluation_data.append(record)
        existing_results[cache_key] = record

    with open(RESULTS_FILE, "w") as f:
        json.dump(existing_results, f, indent=4)

    df = pd.DataFrame(evaluation_data)

    # ── 4. Formal Statistical Math & Printouts ──────────────────────────────
    print("\n" + "="*60)
    print("🔬 FORMAL STATISTICAL ANALYSIS & MARGIN BREAKDOWN")
    print("="*60)
    
    # Calculate exact average margins
    raw_wins = df[df['winner'] == 'RAW']
    adaptive_wins = df[df['winner'] == 'ADAPTIVE']
    
    avg_raw_margin = raw_wins['margin'].mean() if not raw_wins.empty else 0
    avg_adaptive_margin = adaptive_wins['margin'].mean() if not adaptive_wins.empty else 0
    
    print(f"🏆 RAW Mode Total Wins      : {len(raw_wins)}")
    if len(raw_wins) > 0:
        print(f"   ↳ Average Win Margin   : {avg_raw_margin:.2f} / 5.0")
        
    print(f"🏆 ADAPTIVE Mode Total Wins : {len(adaptive_wins)}")
    if len(adaptive_wins) > 0:
        print(f"   ↳ Average Win Margin   : {avg_adaptive_margin:.2f} / 5.0")

    print(f"🤝 Total Ties               : {len(df[df['winner'] == 'TIE'])}")
    
    print("-" * 60)
    
    if wilcoxon and len(gts_raw_list) >= 3:
        # Check if all differences are zero (Wilcoxon throws an error if data is perfectly identical)
        differences = [a - b for a, b in zip(gts_raw_list, gts_adaptive_list)]
        if all(d == 0 for d in differences):
            print("   ⚠️ Wilcoxon Test Skipped: RAW and ADAPTIVE produced identical GTS scores for all queries.")
        else:
            stat, p_value = wilcoxon(gts_raw_list, gts_adaptive_list)
            print(f"   ► Wilcoxon Signed-Rank Test p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("   ✅ The difference in performance is STATISTICALLY SIGNIFICANT (p < 0.05).")
            else:
                print("   ⚠️ The difference is NOT yet statistically significant (p >= 0.05). More queries are needed.")
    else:
        print("   ⚠️ Not enough data points to run Wilcoxon test (need >= 3), or scipy not installed.")
    print("="*60 + "\n")

    # ── 5. Graph Generation ─────────────────────────────────────────────────
    print("📊 Generating Pairwise Graphs...")
    
    win_counts = df['winner'].value_counts()
    for category in ["ADAPTIVE", "RAW", "TIE"]:
        if category not in win_counts: win_counts[category] = 0
        
    labels = ['ADAPTIVE Wins', 'RAW Wins', 'Ties']
    sizes = [win_counts['ADAPTIVE'], win_counts['RAW'], win_counts['TIE']]
    colors = ['#E1812C', '#4C72B0', '#95a5a6']
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    plt.title('Head-to-Head Win Rate (Blinded LLM A/B Test)', fontweight='bold')
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig7_win_rate_donut.png")
    print("✅ Generated: fig7_win_rate_donut.png")

    # Diverging Bar Chart
    df['plot_margin'] = df.apply(lambda row: row['margin'] if row['winner'] == 'ADAPTIVE' else (-row['margin'] if row['winner'] == 'RAW' else 0), axis=1)
    df_sorted = df.sort_values(by='plot_margin', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt_sns.set_theme(style="whitegrid")
    colors_mapped = ['#E1812C' if x > 0 else ('#4C72B0' if x < 0 else '#95a5a6') for x in df_sorted['plot_margin']]
    
    ax = plt_sns.barplot(x='plot_margin', y='query_short', data=df_sorted, palette=colors_mapped, edgecolor='black')
    
    plt.axvline(0, color='black', linewidth=1.2)
    plt.title('Pairwise Margin of Victory (Scale: 1 to 5)', fontweight='bold')
    plt.xlabel('← RAW Wins (Margin)           ADAPTIVE Wins (Margin) →')
    plt.ylabel('User Query')
    plt.xlim(-5.5, 5.5)
    
    for p in ax.patches:
        width = p.get_width()
        if width == 0: continue
        x_offset = 0.2 if width > 0 else -0.2
        ha = 'left' if width > 0 else 'right'
        ax.text(width + x_offset, p.get_y() + p.get_height() / 2, f'{abs(width):.0f}', 
                ha=ha, va='center', fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig8_diverging_margins.png")
    print("✅ Generated: fig8_diverging_margins.png")

if __name__ == "__main__":
    asyncio.run(main())

# ============================================================
# 🔬 FORMAL STATISTICAL ANALYSIS & MARGIN BREAKDOWN
# ============================================================
# 🏆 RAW Mode Total Wins      : 2
#    ↳ Average Win Margin   : 3.00 / 5.0
# 🏆 ADAPTIVE Mode Total Wins : 2
#    ↳ Average Win Margin   : 3.00 / 5.0
# 🤝 Total Ties               : 0
# ------------------------------------------------------------