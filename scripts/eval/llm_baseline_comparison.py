"""
llm_baseline_comparison.py
==========================
Compares adaptive multi-agent answers against vanilla Gemini and GPT-5 baselines.

For each benchmark query:
  1. Loads the best adaptive-mode answer already saved in raw_telemetry/
  2. Calls bare Gemini (no RAG, no agents) with the same question
  3. Calls bare GPT-5 (no RAG, no agents) with the same question
  4. Scores all three answers on 6 dimensions using GPT as judge
  5. Saves a per-query comparison JSON + summary bar chart

Hypothesis: adaptive > generic LLM because it is grounded in real papers (RAG),
uses a structured decomposer, and draws on agent-specialised knowledge.

Usage:
    python scripts/eval/llm_baseline_comparison.py
    python scripts/eval/llm_baseline_comparison.py --dir raw_telemetry --queries 5
    python scripts/eval/llm_baseline_comparison.py --skip-gpt   # Gemini vs adaptive only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import google.generativeai as genai
from openai import AsyncOpenAI

# ── Config ────────────────────────────────────────────────────────────────────
from config import Config

GEMINI_MODEL   = Config.GENERATION_MODEL          # same model as the pipeline
GPT_MODEL      = "gpt-5"                          # strong GPT baseline
JUDGE_MODEL    = Config.JUDGE_MODEL

DIMS = ["Correctness", "Completeness", "Grounding", "Specificity", "Code_Quality", "Coherence"]

COLORS = {
    "adaptive": "#2ecc71",
    "gemini":   "#3498db",
    "gpt":      "#e67e22",
}

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator of bioinformatics AI systems.

QUESTION: "{query}"

ANSWER TO EVALUATE:
{answer}

Score this answer on each dimension from 0–10. Return ONLY valid JSON, no prose:

{{
  "Correctness":   <0-10>,
  "Completeness":  <0-10>,
  "Grounding":     <0-10>,
  "Specificity":   <0-10>,
  "Code_Quality":  <0-10>,
  "Coherence":     <0-10>,
  "reasoning":     "<one sentence>"
}}

Scoring guide:
  Correctness  — factual accuracy, no hallucinations
  Completeness — covers all parts of the question
  Grounding    — cites specific algorithms, papers, tools, or data (not generic claims)
  Specificity  — concrete parameter values, exact tool names, real examples
  Code_Quality — runnable code with proper structure (10 if no code needed, 0 if broken)
  Coherence    — logical flow and clarity"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_adaptive_answers(tel_dir: Path, max_queries: int = 0) -> dict[str, str]:
    """Load best adaptive answer per unique query from telemetry files."""
    best: dict[str, dict] = {}
    for f in sorted(tel_dir.glob("run_*.json")):
        try:
            d  = json.loads(f.read_text())
            et = d.get("engine_telemetry", {})
            if d.get("ablation_mode") != "adaptive":
                continue
            q   = (et.get("user_query") or d.get("query", "")).strip()
            ans = (et.get("final_answer") or "").strip()
            if not q or not ans:
                continue
            score = et.get("final_judge_score") or 0
            if q not in best or score > best[q]["score"]:
                best[q] = {"answer": ans, "score": score}
        except Exception:
            pass

    items = list(best.items())
    if max_queries > 0:
        items = items[:max_queries]
    return {q: v["answer"] for q, v in items}


async def call_gemini(query: str, model) -> str:
    try:
        resp = await asyncio.to_thread(model.generate_content, query)
        return resp.text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"


async def call_gpt(query: str, client: AsyncOpenAI) -> str:
    try:
        resp = await client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "developer", "content": "You are a helpful expert bioinformatics assistant."},
                {"role": "user",      "content": query},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ✘ GPT call error: {e}")
        return f"[GPT error: {e}]"


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a string, tolerating surrounding prose."""
    import re
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


async def judge_answer(query: str, answer: str, client: AsyncOpenAI) -> dict:
    prompt = JUDGE_PROMPT.format(query=query[:300], answer=answer[:3000])
    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "developer", "content": "You are a strict JSON-only evaluator. Output only valid JSON."},
                {"role": "user",      "content": prompt},
            ],
        )
        raw  = resp.choices[0].message.content.strip()
        data = _extract_json(raw)
        if not data:
            raise ValueError(f"No JSON found in judge response: {raw[:200]}")
        scores = {d: float(data.get(d, 0)) for d in DIMS}
        scores["mean"] = round(sum(scores[d] for d in DIMS) / len(DIMS), 2)
        scores["reasoning"] = data.get("reasoning", "")
        return scores
    except Exception as e:
        print(f"   ✘ Judge error: {e}")
        return {d: 0.0 for d in DIMS} | {"mean": 0.0, "reasoning": f"Judge error: {e}"}


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args):
    tel_dir = Path(args.dir) if args.dir else ROOT / "raw_telemetry"
    out_dir = ROOT / "results" / "llm_baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clients
    api_key = os.getenv("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_client = AsyncOpenAI(api_key=openai_key)

    # Load adaptive answers
    print(f"Loading adaptive answers from {tel_dir}...")
    adaptive_answers = load_adaptive_answers(tel_dir, max_queries=args.queries)
    print(f"  {len(adaptive_answers)} unique queries loaded.\n")

    if not adaptive_answers:
        print("No adaptive answers found. Run the ablation benchmark first.")
        sys.exit(1)

    all_results = []
    sys_scores: dict[str, list] = defaultdict(list)  # system → list of mean scores

    for i, (query, adaptive_ans) in enumerate(adaptive_answers.items(), 1):
        print(f"── Query {i}/{len(adaptive_answers)} ──────────────────────────────")
        print(f"   {query[:90]}{'...' if len(query) > 90 else ''}")

        # Step 1: Get baseline answers
        print("   Calling Gemini baseline...")
        t0 = time.monotonic()
        gemini_ans = await call_gemini(query, gemini_model)
        gemini_latency = time.monotonic() - t0

        gpt_ans = ""
        gpt_latency = 0.0
        if not args.skip_gpt:
            print("   Calling GPT baseline...")
            t0 = time.monotonic()
            gpt_ans = await call_gpt(query, openai_client)
            gpt_latency = time.monotonic() - t0

        # Step 2: Judge all answers concurrently
        print("   Judging all answers...")
        judge_tasks = [
            judge_answer(query, adaptive_ans,  openai_client),
            judge_answer(query, gemini_ans,    openai_client),
        ]
        if not args.skip_gpt and gpt_ans:
            judge_tasks.append(judge_answer(query, gpt_ans, openai_client))

        judge_results = await asyncio.gather(*judge_tasks)

        adaptive_scores = judge_results[0]
        gemini_scores   = judge_results[1]
        gpt_scores      = judge_results[2] if (not args.skip_gpt and gpt_ans) else None

        # Print summary
        print(f"   Adaptive  mean={adaptive_scores['mean']:.2f}  | {adaptive_scores['reasoning'][:80]}")
        print(f"   Gemini    mean={gemini_scores['mean']:.2f}  | {gemini_scores['reasoning'][:80]}")
        if gpt_scores:
            print(f"   GPT-5     mean={gpt_scores['mean']:.2f}  | {gpt_scores['reasoning'][:80]}")

        # Accumulate
        sys_scores["adaptive"].append(adaptive_scores["mean"])
        sys_scores["gemini"].append(gemini_scores["mean"])
        if gpt_scores:
            sys_scores["gpt"].append(gpt_scores["mean"])

        record = {
            "query_id": f"Q{i:02d}",
            "query": query,
            "adaptive": {
                "answer": adaptive_ans[:500],
                "scores": adaptive_scores,
            },
            "gemini": {
                "answer": gemini_ans[:500],
                "latency_s": round(gemini_latency, 2),
                "scores": gemini_scores,
            },
        }
        if gpt_scores:
            record["gpt"] = {
                "answer": gpt_ans[:500],
                "latency_s": round(gpt_latency, 2),
                "scores": gpt_scores,
            }
        all_results.append(record)

        await asyncio.sleep(1)  # avoid rate limits

    # Save raw results
    out_json = out_dir / "comparison_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved detailed results → {out_json}")

    # Print summary table
    print("\n" + "─" * 55)
    print(f"  {'System':<12} {'N':>4}  {'Mean Score':>11}  {'Std':>7}")
    print("─" * 55)
    systems = ["adaptive", "gemini"] + (["gpt"] if not args.skip_gpt else [])
    dim_means: dict[str, dict] = {}

    for sys_name in systems:
        scores = sys_scores[sys_name]
        if scores:
            dim_means[sys_name] = {}
            for dim in DIMS:
                dim_means[sys_name][dim] = np.mean([r[sys_name]["scores"][dim] for r in all_results if sys_name in r])
            print(f"  {sys_name:<12} {len(scores):>4}  {np.mean(scores):>9.2f}/10  {np.std(scores):>7.2f}")
    print("─" * 55)

    # Per-dimension breakdown
    print("\nPer-dimension mean scores:")
    print(f"  {'Dimension':<14}", end="")
    for s in systems:
        print(f"  {s:>10}", end="")
    print()
    for dim in DIMS:
        print(f"  {dim:<14}", end="")
        for s in systems:
            val = dim_means.get(s, {}).get(dim, 0)
            print(f"  {val:>10.2f}", end="")
        print()

    # Chart
    _make_chart(sys_scores, dim_means, systems, out_dir / "comparison_chart.png")
    print(f"\nChart saved → {out_dir / 'comparison_chart.png'}")


def _make_chart(sys_scores: dict, dim_means: dict, systems: list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # Left: overall mean scores
    ax = axes[0]
    means = [np.mean(sys_scores[s]) for s in systems]
    stds  = [np.std(sys_scores[s])  for s in systems]
    colors = [COLORS[s] for s in systems]
    x = np.arange(len(systems))
    bars = ax.bar(x, means, 0.5, color=colors, edgecolor="white",
                  yerr=stds, capsize=5, error_kw={"linewidth": 1, "color": "#555"}, zorder=3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _display = {"adaptive": "Adaptive", "gemini": "Gemini", "gpt": "GPT-5"}
    ax.set_xticks(x)
    ax.set_xticklabels([_display.get(s, s.capitalize()) for s in systems], fontsize=11)
    ax.set_ylabel("Mean Judge Score (0–10)", fontsize=9)
    ax.set_title("Overall Answer Quality", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: per-dimension radar-style grouped bar
    ax2 = axes[1]
    n_dims = len(DIMS)
    x2 = np.arange(n_dims)
    width = 0.25
    offsets = np.linspace(-(len(systems)-1)*width/2, (len(systems)-1)*width/2, len(systems))
    _display = {"adaptive": "Adaptive", "gemini": "Gemini", "gpt": "GPT-5"}
    for sys_name, offset in zip(systems, offsets):
        vals = [dim_means.get(sys_name, {}).get(d, 0) for d in DIMS]
        ax2.bar(x2 + offset, vals, width, label=_display.get(sys_name, sys_name.capitalize()),
                color=COLORS[sys_name], edgecolor="white", zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([d.replace("_", "\n") for d in DIMS], fontsize=8)
    ax2.set_ylabel("Score (0–10)", fontsize=9)
    ax2.set_title("Per-Dimension Breakdown", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.yaxis.grid(True, color="#eee", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    n_str = f"N={len(sys_scores.get('adaptive', []))} queries"
    fig.suptitle(f"Adaptive Multi-Agent vs Vanilla LLM Baselines  ({n_str})",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",      default="",  help="Telemetry dir (default: raw_telemetry/)")
    p.add_argument("--queries",  type=int, default=0, help="Limit to first N queries (0=all)")
    p.add_argument("--skip-gpt", action="store_true", help="Skip GPT baseline (Gemini only)")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
