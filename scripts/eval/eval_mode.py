"""
eval_mode.py — Comprehensive per-mode evaluation over raw telemetry
====================================================================
Loads all telemetry files for a given ablation mode, runs LLM-as-judge
on every final answer, and computes all available metrics.

Judge dimensions (match the GS_Score formula used throughout the system):
    Grounding     weight 0.35  — answer grounded in real sources / papers / KB
    Specificity   weight 0.35  — specific details, not generic platitudes
    Correctness   weight 0.15  — factually / technically correct
    Completeness  weight 0.10  — all parts of the question addressed
    Code_Quality  weight 0.03  — valid, runnable code where applicable
    Coherence     weight 0.02  — logically structured, easy to follow

GS_Score = weighted sum of the six dimensions (0–10 scale).

Telemetry folder searched:
    raw_telemetry/  (default — override with --folder)

Usage:
    python scripts/eval/eval_mode.py --mode react_cot_no_context
    python scripts/eval/eval_mode.py --mode adaptive --out results/eval_adaptive.json
    python scripts/eval/eval_mode.py --mode raw --folder raw_telemetry_best
    python scripts/eval/eval_mode.py --mode raw --concurrency 5
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import Config

# ---------------------------------------------------------------------------
# GS_Score weights (must match the rest of the system)
# ---------------------------------------------------------------------------
GS_WEIGHTS = {
    "grounding":    0.35,
    "specificity":  0.35,
    "correctness":  0.15,
    "completeness": 0.10,
    "code_quality": 0.03,
    "coherence":    0.02,
}

DEFAULT_TELEMETRY_FOLDER = ROOT / "raw_telemetry"

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an expert bioinformatics scientist and software engineer evaluating a multi-agent AI system response.

The system was asked:
"{query}"

The system returned:
<response>
{response}
</response>

Score the response on each dimension from 0 to 10:

1. **Grounding** (0-10): Is the answer grounded in real sources, papers, algorithms, or KB content — or does it rely on generic LLM knowledge / hallucinations?
   - 9-10: Specific paper titles, tool names, parameters, or code from real sources
   - 5-8: Mostly correct with minor unsupported claims
   - 0-4: Mostly generic or hallucinated

2. **Specificity** (0-10): Does the answer give concrete, specific details relevant to the exact question asked?
   - 9-10: Precise values, equations, API calls, or step-by-step procedures
   - 5-8: Partially specific but with some vague sections
   - 0-4: Generic, could apply to any question in the domain

3. **Correctness** (0-10): Are the scientific, biological, or technical claims factually correct?
   - 9-10: No errors detected
   - 5-8: Mostly correct with minor inaccuracies
   - 0-4: Notable factual errors or contradictions

4. **Completeness** (0-10): Does the answer address ALL parts of the question?
   - 9-10: All aspects covered
   - 5-8: Most aspects covered, one part missing
   - 0-4: Significant parts of the question ignored

5. **Code_Quality** (0-10): If the response contains code, is it syntactically valid, imports correct libraries, and follows good practice? Score 5 if no code was requested or expected.
   - 9-10: Correct, runnable, well-structured code
   - 5-8: Mostly correct with minor issues
   - 0-4: Broken, hallucinated functions, or missing imports
   - 5: No code in response (neutral)

6. **Coherence** (0-10): Is the response logically structured and easy to follow?
   - 9-10: Clear structure, logical flow, no contradictions
   - 5-8: Mostly clear with minor structural issues
   - 0-4: Disorganised or self-contradictory

Return ONLY this JSON (no other text):
{{
  "grounding":    <0-10>,
  "specificity":  <0-10>,
  "correctness":  <0-10>,
  "completeness": <0-10>,
  "code_quality": <0-10>,
  "coherence":    <0-10>,
  "rationale":    "<one sentence summary of main strengths and weaknesses>"
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_gs_score(scores: dict) -> float:
    return round(sum(GS_WEIGHTS[k] * scores[k] for k in GS_WEIGHTS), 3)


def find_telemetry(mode: str, folder: Path) -> list[Path]:
    files = []
    if not folder.exists():
        return files
    for f in sorted(folder.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("ablation_mode") == mode:
                files.append(f)
        except Exception:
            pass
    return files


def extract_metrics(d: dict) -> dict:
    """Pull every numeric metric out of a telemetry file."""
    et = d.get("engine_telemetry", {})
    nodes = et.get("got_dag_nodes", [])
    edges = et.get("communication_edges", [])
    tt    = et.get("total_tokens", {})
    ot    = et.get("orchestrator_tokens", {})

    # ── Node-level aggregates ────────────────────────────────────────────────
    quality_scores  = [n["quality_score"] for n in nodes if n.get("quality_score") is not None]
    node_latencies  = [n["elapsed_s"]     for n in nodes if n.get("elapsed_s")     is not None]
    context_tokens  = [n["context_tokens"]for n in nodes if n.get("context_tokens") is not None]
    tokens_in_list  = [n.get("tokens_in",  0) for n in nodes]
    tokens_out_list = [n.get("tokens_out", 0) for n in nodes]
    cugcr_nodes     = sum(1 for n in nodes if n.get("cugcr_triggered"))
    failed_nodes    = sum(1 for n in nodes if n.get("failure_mode") and n["failure_mode"] not in (None, "none"))
    paper_counts    = [n["paper_count"] for n in nodes if n.get("paper_count") is not None]

    failure_mode_counts: dict[str, int] = {}
    for n in nodes:
        fm = n.get("failure_mode")
        if fm and fm != "none":
            failure_mode_counts[fm] = failure_mode_counts.get(fm, 0) + 1

    # ── Edge-level aggregates ────────────────────────────────────────────────
    ctx_tok_list  = [e.get("context_token_count", 0) for e in edges]
    util_list     = [e.get("utilization_score", None) for e in edges if e.get("utilization_score") is not None]
    rel_list      = [e.get("context_relevance",  None) for e in edges if e.get("context_relevance")  is not None]

    def mean(lst):  return round(sum(lst) / len(lst), 4) if lst else None
    def total(lst): return sum(lst) if lst else 0

    return {
        # identity
        "run_id":            d.get("run_id"),
        "query_id":          d.get("query_id"),
        "ablation_mode":     d.get("ablation_mode"),
        "dag_locked":        d.get("dag_locked"),
        # latency
        "wall_latency_ms":   d.get("wall_latency_ms"),
        "latency_s":         et.get("latency_seconds"),
        "profiler_lat_ms":   et.get("profiler_latency_ms"),
        "decomp_lat_ms":     et.get("decomposer_latency_ms"),
        # tokens — orchestrator
        "orch_tokens_in":    ot.get("input", 0),
        "orch_tokens_out":   ot.get("output", 0),
        # tokens — agents (populated only for runs after token tracking was added)
        "agent_tokens_in":   tt.get("agents_input",  0),
        "agent_tokens_out":  tt.get("agents_output", 0),
        "grand_total_tokens":tt.get("grand_total",   ot.get("input", 0) + ot.get("output", 0)),
        # DAG / GoT
        "n_nodes":           len(nodes),
        "n_edges":           len(edges),
        "n_cugcr_triggered": cugcr_nodes,
        "cugcr_run":         et.get("cugcr_triggered", False),
        "n_failed_nodes":    failed_nodes,
        "failure_modes":     failure_mode_counts,
        "mean_node_quality": mean(quality_scores),
        "mean_node_lat_s":   mean(node_latencies),
        "mean_context_tokens": mean(context_tokens),
        "total_context_tokens": total(context_tokens),
        "node_tokens_in":    total(tokens_in_list),
        "node_tokens_out":   total(tokens_out_list),
        "mean_paper_count":  mean(paper_counts),
        # communication
        "mean_ctx_token_count":  mean(ctx_tok_list),
        "mean_utilization_score":mean(util_list),
        "mean_context_relevance":mean(rel_list),
        # self-healing
        "first_pass_success":et.get("first_pass_success"),
        "global_retries":    et.get("global_retries", 0),
        # profiler
        "profiler_used":     et.get("use_profiler"),
        "decomposer":        et.get("decomposer_used"),
    }


# ---------------------------------------------------------------------------
# LLM judge (async, rate-limited)
# ---------------------------------------------------------------------------

async def judge_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    run_id: str,
    query: str,
    answer: str,
) -> dict:
    prompt = JUDGE_PROMPT.format(query=query, response=answer[:5000])
    async with sem:
        try:
            result = await client.chat.completions.create(
                model=Config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = result.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            scores = json.loads(text)
            scores["gs_score"] = compute_gs_score(scores)
            return {"run_id": run_id, "judge": scores, "error": None}
        except Exception as exc:
            fallback = {k: 0 for k in GS_WEIGHTS}
            fallback["gs_score"] = 0.0
            fallback["rationale"] = f"Judge error: {exc}"
            return {"run_id": run_id, "judge": fallback, "error": str(exc)}


async def run_all_judges(runs: list[dict], concurrency: int) -> dict[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    client = AsyncOpenAI(api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    tasks = [
        judge_one(sem, client, r["run_id"], r["query"], r["final_answer"])
        for r in runs
    ]
    print(f"  Running LLM-as-judge on {len(tasks)} answers "
          f"(model={Config.JUDGE_MODEL}, concurrency={concurrency})...")
    t0 = time.monotonic()
    results = await asyncio.gather(*tasks)
    print(f"  Judge complete in {time.monotonic()-t0:.1f}s")
    return {r["run_id"]: r for r in results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], mode: str) -> None:
    print(f"\n{'═'*110}")
    print(f"  MODE: {mode}   |   {len(rows)} runs")
    print(f"{'═'*110}")

    # Header
    hdr = (
        f"  {'ID':<8} {'QID':<6} "
        f"{'GS':>5} {'Grnd':>5} {'Spec':>5} {'Corr':>5} {'Comp':>5} {'Code':>5} {'Cohr':>5} "
        f"{'Lat(s)':>7} {'OTok':>6} {'ATok':>6} "
        f"{'Nodes':>6} {'NQS':>5} {'Edges':>6} {'Ret':>4} {'1stP':>5}"
    )
    print(hdr)
    print(f"  {'─'*8} {'─'*6} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*7} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*5} {'─'*6} {'─'*4} {'─'*5}")

    for r in rows:
        j  = r.get("judge", {})
        m  = r.get("metrics", {})
        gs = j.get("gs_score", 0)
        ot = (m.get("orch_tokens_in", 0) or 0) + (m.get("orch_tokens_out", 0) or 0)
        at = (m.get("agent_tokens_in", 0) or 0) + (m.get("agent_tokens_out", 0) or 0)
        nq = m.get("mean_node_quality")
        fp = "✓" if m.get("first_pass_success") else ("✗" if m.get("first_pass_success") is False else "—")
        print(
            f"  {m.get('run_id','?')[:8]:<8} {m.get('query_id','?'):<6} "
            f"{gs:>5.2f} {j.get('grounding',0):>5} {j.get('specificity',0):>5} "
            f"{j.get('correctness',0):>5} {j.get('completeness',0):>5} "
            f"{j.get('code_quality',0):>5} {j.get('coherence',0):>5} "
            f"{(m.get('latency_s') or 0):>7.1f} {ot:>6} {at:>6} "
            f"{(m.get('n_nodes') or 0):>6} "
            f"{f'{nq:.2f}' if nq is not None else '  — ':>5} "
            f"{(m.get('n_edges') or 0):>6} "
            f"{(m.get('global_retries') or 0):>4} {fp:>5}"
        )

    # Summary
    gs_scores    = [r["judge"]["gs_score"] for r in rows]
    lat_vals     = [r["metrics"].get("latency_s") or 0 for r in rows]
    orch_toks    = [(r["metrics"].get("orch_tokens_in", 0) or 0) +
                    (r["metrics"].get("orch_tokens_out", 0) or 0) for r in rows]
    agent_toks   = [(r["metrics"].get("agent_tokens_in", 0) or 0) +
                    (r["metrics"].get("agent_tokens_out", 0) or 0) for r in rows]
    retries      = [r["metrics"].get("global_retries", 0) or 0 for r in rows]
    first_pass   = [r["metrics"].get("first_pass_success") for r in rows]
    fp_rate      = sum(1 for x in first_pass if x is True) / len(rows) * 100

    def mean(lst): return sum(lst) / len(lst) if lst else 0

    dim_means = {k: mean([r["judge"].get(k, 0) for r in rows]) for k in GS_WEIGHTS}

    print(f"\n{'─'*110}")
    print(f"  SUMMARY")
    print(f"  {'Mean GS_Score':.<35} {mean(gs_scores):.3f} / 10")
    print(f"  {'Mean Grounding':.<35} {dim_means['grounding']:.2f}")
    print(f"  {'Mean Specificity':.<35} {dim_means['specificity']:.2f}")
    print(f"  {'Mean Correctness':.<35} {dim_means['correctness']:.2f}")
    print(f"  {'Mean Completeness':.<35} {dim_means['completeness']:.2f}")
    print(f"  {'Mean Code_Quality':.<35} {dim_means['code_quality']:.2f}")
    print(f"  {'Mean Coherence':.<35} {dim_means['coherence']:.2f}")
    print(f"  {'─'*50}")
    print(f"  {'Mean Latency (s)':.<35} {mean(lat_vals):.2f}")
    print(f"  {'Mean Orch Tokens (in+out)':.<35} {mean(orch_toks):.0f}")
    print(f"  {'Mean Agent Tokens (in+out)':.<35} {mean(agent_toks):.0f}  {'(0 = runs predating token tracking)' if mean(agent_toks)==0 else ''}")
    print(f"  {'Total Orch Tokens':.<35} {sum(orch_toks)}")
    print(f"  {'Total Agent Tokens':.<35} {sum(agent_toks)}")
    print(f"  {'Mean Global Retries':.<35} {mean(retries):.2f}")
    print(f"  {'First-Pass Success Rate':.<35} {fp_rate:.0f}%")
    print(f"{'═'*110}\n")


async def main(mode: str, out_path: Path, concurrency: int, skip_judge: bool, folder: Path) -> None:
    print(f"\n── Mode Evaluation: {mode} {'─'*50}")
    print(f"  Telemetry folder: {folder}")

    # 1. Find files
    files = find_telemetry(mode, folder)
    if not files:
        print(f"  No telemetry found for mode='{mode}' in: {folder}")
        sys.exit(1)
    print(f"  Found {len(files)} telemetry files.")

    # 2. Load
    runs = []
    for f in files:
        d = json.loads(f.read_text())
        et = d.get("engine_telemetry", {})
        runs.append({
            "run_id":       d.get("run_id", f.stem),
            "query_id":     d.get("query_id", "?"),
            "query":        et.get("user_query", d.get("query", "")),
            "final_answer": et.get("final_answer", ""),
            "metrics":      extract_metrics(d),
        })

    # 3. Judge
    judge_map: dict[str, dict] = {}
    if skip_judge:
        print("  --skip-judge set: using zero scores.")
        for r in runs:
            judge_map[r["run_id"]] = {
                "run_id": r["run_id"],
                "judge": {k: 0 for k in list(GS_WEIGHTS) + ["gs_score", "rationale"]},
                "error": "skipped",
            }
    else:
        judge_map = await run_all_judges(runs, concurrency)

    # 4. Merge
    rows = []
    for r in runs:
        jdata = judge_map.get(r["run_id"], {})
        rows.append({
            "run_id":       r["run_id"],
            "query_id":     r["query_id"],
            "query":        r["query"],
            "final_answer": r["final_answer"],
            "metrics":      r["metrics"],
            "judge":        jdata.get("judge", {}),
            "judge_error":  jdata.get("error"),
        })

    # Sort by query_id for readability
    rows.sort(key=lambda x: x["query_id"])

    # 5. Print table
    print_table(rows, mode)

    # 6. Build summary for JSON
    gs_scores = [r["judge"].get("gs_score", 0) for r in rows]
    dim_means = {k: sum(r["judge"].get(k, 0) for r in rows) / len(rows) for k in GS_WEIGHTS}
    lat_vals  = [r["metrics"].get("latency_s") or 0 for r in rows]
    orch_toks = [(r["metrics"].get("orch_tokens_in", 0) or 0) +
                 (r["metrics"].get("orch_tokens_out", 0) or 0) for r in rows]
    agent_toks= [(r["metrics"].get("agent_tokens_in", 0) or 0) +
                 (r["metrics"].get("agent_tokens_out", 0) or 0) for r in rows]

    out_data = {
        "mode":              mode,
        "n_runs":            len(rows),
        "judge_model":       Config.JUDGE_MODEL,
        "summary": {
            "mean_gs_score":      round(sum(gs_scores) / len(gs_scores), 3),
            "dim_means":          {k: round(v, 3) for k, v in dim_means.items()},
            "mean_latency_s":     round(sum(lat_vals) / len(lat_vals), 2),
            "total_orch_tokens":  sum(orch_toks),
            "mean_orch_tokens":   round(sum(orch_toks) / len(orch_toks)),
            "total_agent_tokens": sum(agent_toks),
            "mean_agent_tokens":  round(sum(agent_toks) / len(agent_toks)),
            "first_pass_rate_pct": round(
                sum(1 for r in rows if r["metrics"].get("first_pass_success") is True)
                / len(rows) * 100, 1),
            "mean_global_retries": round(
                sum(r["metrics"].get("global_retries", 0) or 0 for r in rows) / len(rows), 2),
        },
        "runs": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"  → Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        required=True,
                        help="Ablation mode to evaluate, e.g. react_cot_no_context, adaptive, raw")
    parser.add_argument("--out",         type=Path, default=None,
                        help="Output JSON path (default: results/eval_<mode>.json)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel LLM judge calls (default 5)")
    parser.add_argument("--skip-judge",  action="store_true",
                        help="Skip LLM judge, only compute telemetry metrics")
    parser.add_argument("--folder",      type=Path, default=None,
                        help="Telemetry folder to search (default: raw_telemetry/)")
    args = parser.parse_args()

    folder = args.folder or DEFAULT_TELEMETRY_FOLDER
    out    = args.out or (ROOT / "results" / f"eval_{args.mode}.json")
    asyncio.run(main(args.mode, out, args.concurrency, args.skip_judge, folder))
