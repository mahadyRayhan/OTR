"""
run_profiler_ablation.py
========================
Evaluate the impact of the user profiler on Adaptive pipeline quality.

Strategy
--------
  WITH profiler  : Reuse existing raw_telemetry/ adaptive runs (all have
                   use_profiler=True).  No re-running needed — 0 LLM calls.
  WITHOUT profiler: Run the 30 best-queries through OrchestratorEngine
                   with config.Config.USE_PROFILER = False.

The 30 queries are those where Adaptive beats BOTH Gemini and GPT on GS_Score
in the full 50-query benchmark (see best_questions.txt for details and scores).

Total new LLM calls: ~30 (without-profiler answers) + 60 (judge) = ~90.

Output
------
  results/profiler_ablation/
    profiler_ablation_results.json  — per-query scores for both conditions
    profiler_ablation_chart.png     — GS_Score bars + per-dimension breakdown

Usage
-----
    python scripts/run_profiler_ablation.py
    python scripts/run_profiler_ablation.py --queries 5   # quick test
    python scripts/run_profiler_ablation.py --skip-run    # judge only (reuse saved no-profiler answers)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI
from config import Config

# ── Paths ─────────────────────────────────────────────────────────────────────
TEL_DIR     = ROOT / "raw_telemetry"
RESULTS_DIR = ROOT / "results" / "profiler_ablation"
ENGINE_TEL  = ROOT / "experiment_telemetry.json"
NO_PROF_DIR = ROOT / "raw_telemetry_noprofiler"

JUDGE_MODEL = Config.JUDGE_MODEL

DIMS = ["Correctness", "Completeness", "Grounding", "Specificity", "Code_Quality", "Coherence"]
GS_WEIGHTS = {
    "Grounding": 0.35, "Specificity": 0.35, "Correctness": 0.15,
    "Completeness": 0.10, "Code_Quality": 0.03, "Coherence": 0.02,
}
COLORS = {"with_profiler": "#2ecc71", "without_profiler": "#e74c3c"}
LABELS = {"with_profiler": "Adaptive + Profiler", "without_profiler": "Adaptive (no profiler)"}

# ── Load the 30 best queries from the shared JSON file ────────────────────────
# Edit best_questions.json at the project root to change which queries are used.
_BEST_Q_PATH = ROOT.parent / "best_questions.json"
if not _BEST_Q_PATH.exists():
    _BEST_Q_PATH = ROOT / "best_questions.json"

def _load_best_queries(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return [entry["query"] for entry in data]

BEST_QUERIES    = _load_best_queries(_BEST_Q_PATH)
BENCHMARK_QUERIES = BEST_QUERIES  # alias used throughout the script
# BENCHMARK_QUERIES = BEST_QUERIES[:10]  # limit to top 10 for profiler ablation


# ── Load existing "with profiler" answers from raw_telemetry ──────────────────

def load_with_profiler_answers(queries: list[str]) -> dict[str, str]:
    """
    Scan raw_telemetry/ for adaptive runs with use_profiler=True.
    For each query return the latest answer (by timestamp).
    """
    best: dict[str, tuple[str, str]] = {}  # query_prefix -> (timestamp, answer)

    for fn in TEL_DIR.glob("run_*.json"):
        try:
            data = json.loads(fn.read_text())
        except Exception:
            continue
        if data.get("ablation_mode") != "adaptive":
            continue
        et = data.get("engine_telemetry", {})
        if not et.get("use_profiler"):
            continue
        q = (et.get("user_query") or data.get("query", "")).strip()
        ans = et.get("final_answer", "").strip()
        ts = data.get("timestamp", "")
        if not (q and ans):
            continue
        key = q[:120]
        if key not in best or ts > best[key][0]:
            best[key] = (ts, ans)

    result: dict[str, str] = {}
    for q in queries:
        key = q[:120]
        if key in best:
            result[q] = best[key][1]
        else:
            result[q] = ""
    return result


# ── Run without-profiler answers ──────────────────────────────────────────────

def _read_engine_telemetry() -> dict:
    if not ENGINE_TEL.exists():
        return {}
    try:
        data = json.loads(ENGINE_TEL.read_text())
        if isinstance(data, list) and data:
            return data[-1]
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_noprofiler_telemetry(query: str, telemetry: dict, wall_ms: int) -> str:
    NO_PROF_DIR.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    record = {
        "run_id":            run_id,
        "timestamp":         datetime.utcnow().isoformat(),
        "ablation_mode":     "adaptive",
        "profiler_ablation": True,
        "use_profiler":      False,
        "query":             query,
        "wall_latency_ms":   wall_ms,
        "engine_telemetry":  telemetry,
    }
    (NO_PROF_DIR / f"run_{run_id}.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    return run_id


def load_noprofiler_checkpoint(queries: list[str]) -> dict[str, str]:
    """Load already-completed without-profiler answers from NO_PROF_DIR."""
    best: dict[str, tuple[str, str]] = {}
    if not NO_PROF_DIR.exists():
        return {}
    for fn in NO_PROF_DIR.glob("run_*.json"):
        try:
            data = json.loads(fn.read_text())
        except Exception:
            continue
        et = data.get("engine_telemetry", {})
        q = (et.get("user_query") or data.get("query", "")).strip()
        ans = et.get("final_answer", "").strip()
        ts = data.get("timestamp", "")
        if not (q and ans):
            continue
        key = q[:120]
        if key not in best or ts > best[key][0]:
            best[key] = (ts, ans)

    return {q: best.get(q[:120], ("", ""))[1] for q in queries}


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator of bioinformatics AI systems. Many questions ask the system to retrieve information from a specific knowledge base (KB) of papers. Reward answers that cite paper-specific content; penalize answers that give generic textbook descriptions.

QUESTION: "{query}"

ANSWER TO EVALUATE:
{answer}

Score this answer on each dimension from 0-10. Return ONLY valid JSON, no prose:

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
  Correctness  — factual accuracy with no hallucinations; verify specific claims (parameter names, flag values, formulas)
  Completeness — covers all parts of the multi-part question; partial answers lose points proportionally
  Grounding    — how well the answer is anchored to paper/KB content:
                   0-3: generic claims only ("Leiden uses community detection"), no paper references
                   4-6: names specific tools/parameters ("leidenalg with CPM resolution parameter")
                   7-9: cites paper-specific content with exact values, algorithm steps, or named methods
                  10: directly quotes or precisely paraphrases paper-exact content traceable to a named KB source
  Specificity  — concreteness of technical detail:
                   0-3: vague descriptions ("performs clustering", "computes similarities")
                   4-6: tool-specific but generic ("uses sc.tl.leiden() with default parameters")
                   7-9: exact parameter values, algorithm steps, or library calls that reflect actual paper methods
                  10: paper-exact formulas, specific numerical defaults, or pseudocode traceable to a named KB source
  Code_Quality — runnable code with correct API calls, proper parameters, and logical structure (10 if no code needed, 0 if broken)
  Coherence    — logical flow, clear structure, and well-organized multi-part answer"""


def _extract_json(raw: str) -> dict:
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
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
            raise ValueError(f"No JSON in judge response: {raw[:200]}")
        scores = {d: float(data.get(d, 0)) for d in DIMS}
        scores["mean"]      = round(sum(scores[d] for d in DIMS) / len(DIMS), 2)
        scores["gs_score"]  = round(sum(scores[d] * GS_WEIGHTS[d] for d in DIMS), 2)
        scores["reasoning"] = data.get("reasoning", "")
        return scores
    except Exception as e:
        print(f"   ✘ Judge error: {e}")
        return {d: 0.0 for d in DIMS} | {"mean": 0.0, "gs_score": 0.0, "reasoning": f"Judge error: {e}"}


# ── Chart ─────────────────────────────────────────────────────────────────────

def _make_chart(systems: list, sys_gs: dict, sys_scores: dict, dim_means: dict,
                n_queries: int, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    gs_means = [np.mean(sys_gs[s]) for s in systems]
    gs_stds  = [np.std(sys_gs[s])  for s in systems]
    x = np.arange(len(systems))
    bars = ax.bar(x, gs_means, 0.5, color=[COLORS[s] for s in systems],
                  edgecolor="white", yerr=gs_stds, capsize=5,
                  error_kw={"linewidth": 1, "color": "#555"}, zorder=3)
    for bar, v in zip(bars, gs_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in systems], fontsize=9)
    ax.set_ylabel("GS_Score (0–10)\n(0.35×Grounding + 0.35×Specificity + …)", fontsize=8)
    ax.set_title("Profiler Impact: Primary Metric GS_Score", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    if "with_profiler" in systems and "without_profiler" in systems:
        wp  = np.mean(sys_gs["with_profiler"])
        nop = np.mean(sys_gs["without_profiler"])
        delta = (wp - nop) / nop * 100 if nop else 0
        idx = systems.index("with_profiler")
        ax.annotate(f"{delta:+.1f}%\nvs no profiler",
                    xy=(x[idx], gs_means[idx] + gs_stds[idx] + 0.4),
                    ha="center", fontsize=8, fontweight="bold",
                    color="#2ecc71" if delta >= 0 else "#e74c3c")

    ax2 = axes[1]
    x2 = np.arange(len(DIMS))
    width = 0.35
    offsets = [-width / 2, width / 2]
    for sys_name, offset in zip(systems, offsets):
        vals = [dim_means.get(sys_name, {}).get(d, 0) for d in DIMS]
        ax2.bar(x2 + offset, vals, width, label=LABELS[sys_name],
                color=COLORS[sys_name], edgecolor="white", zorder=3)
    for xi in [DIMS.index("Grounding"), DIMS.index("Specificity")]:
        ax2.axvspan(xi - 0.45, xi + 0.45, color="#fffde7", zorder=0, alpha=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([d.replace("_", "\n") for d in DIMS], fontsize=8)
    ax2.set_ylabel("Score (0–10)", fontsize=9)
    ax2.set_title("Per-Dimension Breakdown  (highlighted = weighted dims)", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.yaxis.grid(True, color="#eee", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Profiler Ablation  |  N={n_queries} queries  |  Adaptive pipeline  |  Primary: GS_Score",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queries = BENCHMARK_QUERIES[:args.queries] if args.queries > 0 else BENCHMARK_QUERIES

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set — needed for judge calls.")
        sys.exit(1)
    client = AsyncOpenAI(api_key=openai_key)

    # ── Step 1: load "with profiler" answers (0 new LLM calls) ──────────────
    print("⏳ Loading 'with profiler' answers from raw_telemetry/...")
    with_answers = load_with_profiler_answers(queries)
    missing_with = [q for q in queries if not with_answers.get(q)]
    if missing_with:
        print(f"  ⚠  {len(missing_with)} queries have no 'with profiler' answer in raw_telemetry/")
    else:
        print(f"  ✔ All {len(queries)} 'with profiler' answers loaded.")

    # ── Step 2: run "without profiler" answers ───────────────────────────────
    no_answers: dict[str, str] = {}

    if args.skip_run:
        print("\n⏭  --skip-run: loading without-profiler answers from checkpoint...")
        no_answers = load_noprofiler_checkpoint(queries)
        missing_no = [q for q in queries if not no_answers.get(q)]
        if missing_no:
            print(f"  ⚠  {len(missing_no)} queries have no checkpoint — run without --skip-run first.")
    else:
        checkpoint = load_noprofiler_checkpoint(queries)
        already_done = {q for q in queries if checkpoint.get(q)}
        if already_done:
            print(f"\n  ✔ {len(already_done)} without-profiler answers loaded from checkpoint.")
        no_answers.update(checkpoint)

        remaining = [q for q in queries if not no_answers.get(q)]
        if remaining:
            # Disable profiler before engine is imported/initialised
            Config.USE_PROFILER = False
            print(f"\n⏳ Loading OrchestratorEngine (USE_PROFILER=False)...")
            from orchestrator.engine import OrchestratorEngine
            engine = OrchestratorEngine()
            if ENGINE_TEL.exists():
                ENGINE_TEL.unlink()

            print(f"\n  Running {len(remaining)} queries without profiler...\n")
            for i, query in enumerate(remaining, 1):
                qi = queries.index(query) + 1
                print(f"══ Query {qi}/{len(queries)} (run {i}/{len(remaining)}) {'═'*35}")
                print(f"   {query[:100]}{'...' if len(query) > 100 else ''}")
                t0 = time.time()
                try:
                    result = await engine.run(query, ablation_mode="adaptive", start_time=t0)
                    wall_ms = int((time.time() - t0) * 1000)
                    ans = result.get("final_answer", "").strip()
                    tel = _read_engine_telemetry()
                    run_id = _save_noprofiler_telemetry(query, tel, wall_ms)
                    no_answers[query] = ans
                    print(f"  ✔ Done ({wall_ms}ms) → saved run_{run_id}.json")
                except Exception as e:
                    print(f"  ✘ Error: {e}")
                    no_answers[query] = f"[Engine error: {e}]"
                    await asyncio.sleep(5)
                    continue
                await asyncio.sleep(2)

    # ── Step 3: judge both conditions ────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  JUDGING: with profiler vs without profiler")
    print(f"  Queries: {len(queries)}")
    print(f"{'═'*62}\n")

    systems = ["with_profiler", "without_profiler"]
    sys_scores: dict[str, list] = defaultdict(list)
    sys_gs:     dict[str, list] = defaultdict(list)
    all_results: list[dict] = []

    checkpoint_path = RESULTS_DIR / "profiler_ablation_results.json"

    for q_idx, query in enumerate(queries, 1):
        print(f"\n── Q{q_idx:02d}/{len(queries)}: {query[:80]}...")

        ans_with = with_answers.get(query, "")
        ans_no   = no_answers.get(query, "")

        if not ans_with:
            print(f"  ⚠  Skipping — no 'with profiler' answer.")
            continue
        if not ans_no:
            print(f"  ⚠  Skipping — no 'without profiler' answer.")
            continue

        judge_tasks = [
            judge_answer(query, ans_with, client),
            judge_answer(query, ans_no,   client),
        ]
        scored = await asyncio.gather(*judge_tasks)
        scores_with, scores_no = scored

        sys_scores["with_profiler"].append(scores_with["mean"])
        sys_gs["with_profiler"].append(scores_with["gs_score"])
        sys_scores["without_profiler"].append(scores_no["mean"])
        sys_gs["without_profiler"].append(scores_no["gs_score"])

        record = {
            "query_id": f"Q{q_idx:02d}",
            "query": query,
            "with_profiler": {
                "answer": ans_with[:500],
                "scores": scores_with,
            },
            "without_profiler": {
                "answer": ans_no[:500],
                "scores": scores_no,
            },
            "gs_delta": round(scores_with["gs_score"] - scores_no["gs_score"], 3),
        }
        all_results.append(record)

        winner = "with" if scores_with["gs_score"] >= scores_no["gs_score"] else "without"
        print(f"  with={scores_with['gs_score']:.2f}  no={scores_no['gs_score']:.2f}  "
              f"Δ={record['gs_delta']:+.2f}  [{winner} wins]")

        checkpoint_path.write_text(json.dumps(all_results, indent=2))
        await asyncio.sleep(1)

    # ── Step 4: summary ──────────────────────────────────────────────────────
    n = len(all_results)
    print(f"\n{'═'*72}")
    print(f"  PROFILER ABLATION RESULTS  |  N={n} queries")
    print(f"{'═'*72}")
    print(f"  {'Condition':<24} {'N':>4}  {'Mean (equal)':>13}  {'GS_Score (wt)':>14}  {'Std':>6}")
    print(f"  {'-'*68}")
    for s in systems:
        gs, sc = sys_gs[s], sys_scores[s]
        if gs:
            print(f"  {LABELS[s]:<24} {len(gs):>4}  {np.mean(sc):>11.2f}/10  "
                  f"{np.mean(gs):>12.2f}/10  {np.std(gs):>6.2f}")

    if sys_gs["with_profiler"] and sys_gs["without_profiler"]:
        delta = np.mean(sys_gs["with_profiler"]) - np.mean(sys_gs["without_profiler"])
        pct   = delta / np.mean(sys_gs["without_profiler"]) * 100
        print(f"\n  Profiler GS_Score lift: {delta:+.3f} pts  ({pct:+.1f}%)")

        wins_with = sum(1 for r in all_results if r["gs_delta"] > 0)
        wins_no   = sum(1 for r in all_results if r["gs_delta"] < 0)
        ties      = sum(1 for r in all_results if r["gs_delta"] == 0)
        print(f"  Per-query wins — with: {wins_with}  without: {wins_no}  ties: {ties}")

    print(f"\n  Per-dimension means:")
    print(f"  {'Dimension':<14}  {'With profiler':>14}  {'No profiler':>12}  {'Delta':>8}")
    print(f"  {'-'*54}")
    dim_means: dict[str, dict] = {}
    for s in systems:
        if not all_results:
            dim_means[s] = {d: 0.0 for d in DIMS}
            continue
        dim_means[s] = {
            d: np.mean([r[s]["scores"][d] for r in all_results if s in r])
            for d in DIMS
        }
    for d in DIMS:
        wp  = dim_means["with_profiler"].get(d, 0)
        nop = dim_means["without_profiler"].get(d, 0)
        print(f"  {d:<14}  {wp:>14.2f}  {nop:>12.2f}  {wp-nop:>+8.2f}")

    checkpoint_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nDetailed results → {checkpoint_path}")

    if all_results:
        _make_chart(
            systems, sys_gs, sys_scores, dim_means, n,
            RESULTS_DIR / "profiler_ablation_chart.png",
        )


def main():
    p = argparse.ArgumentParser(description="Profiler ablation: Adaptive WITH vs WITHOUT profiler")
    p.add_argument("--queries",  type=int, default=0,
                   help="Run only first N queries (0 = all 50)")
    p.add_argument("--skip-run", action="store_true",
                   help="Skip running without-profiler answers — judge only using saved results")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
