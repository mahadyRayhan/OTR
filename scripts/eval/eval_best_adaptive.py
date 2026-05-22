"""
eval_best_adaptive.py — Evaluate the top-N adaptive runs from a telemetry folder
==================================================================================
Loads all files with ablation_mode == "adaptive", ranks them by mean DAG node
quality score, picks the top N, then runs the same LLM-as-judge + metrics
pipeline as eval_mode.py.

Usage:
    python scripts/eval/eval_best_adaptive.py
    python scripts/eval/eval_best_adaptive.py --folder raw_telemetry_best --top 30
    python scripts/eval/eval_best_adaptive.py --top 30 --out results/eval_best_adaptive.json
    python scripts/eval/eval_best_adaptive.py --skip-judge   # metrics only
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
# GS_Score weights — must match the rest of the system
# ---------------------------------------------------------------------------
GS_WEIGHTS = {
    "grounding":    0.35,
    "specificity":  0.35,
    "correctness":  0.15,
    "completeness": 0.10,
    "code_quality": 0.03,
    "coherence":    0.02,
}

DEFAULT_FOLDER = ROOT / "raw_telemetry_best"

# ---------------------------------------------------------------------------
# Judge prompt (identical to eval_mode.py)
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


def _mean(lst: list) -> float | None:
    return round(sum(lst) / len(lst), 4) if lst else None


def _total(lst: list) -> int | float:
    return sum(lst) if lst else 0


def selection_score(d: dict) -> float:
    """Ranking signal used to pick the best N runs (no API needed)."""
    et = d.get("engine_telemetry", {})
    nodes = et.get("got_dag_nodes", [])
    qualities = [n["quality_score"] for n in nodes if n.get("quality_score") is not None]
    if qualities:
        return sum(qualities) / len(qualities)
    # Fall back to stored final judge score if no node qualities
    fj = et.get("final_judge_score")
    if isinstance(fj, (int, float)):
        return float(fj) / 10.0
    return 0.0


def extract_metrics(d: dict) -> dict:
    et    = d.get("engine_telemetry", {})
    nodes = et.get("got_dag_nodes", [])
    edges = et.get("communication_edges", [])
    tt    = et.get("total_tokens", {})
    ot    = et.get("orchestrator_tokens", {})

    quality_scores  = [n["quality_score"]  for n in nodes if n.get("quality_score")  is not None]
    node_latencies  = [n["elapsed_s"]      for n in nodes if n.get("elapsed_s")      is not None]
    context_tokens  = [n["context_tokens"] for n in nodes if n.get("context_tokens") is not None]
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

    ctx_tok_list = [e.get("context_token_count", 0) for e in edges]
    util_list    = [e.get("utilization_score",   None) for e in edges if e.get("utilization_score") is not None]
    rel_list     = [e.get("context_relevance",   None) for e in edges if e.get("context_relevance")  is not None]

    return {
        "run_id":               d.get("run_id"),
        "query_id":             d.get("query_id", "?"),
        "ablation_mode":        d.get("ablation_mode"),
        "dag_locked":           d.get("dag_locked"),
        "wall_latency_ms":      d.get("wall_latency_ms"),
        "latency_s":            et.get("latency_seconds"),
        "profiler_lat_ms":      et.get("profiler_latency_ms"),
        "decomp_lat_ms":        et.get("decomposer_latency_ms"),
        "orch_tokens_in":       ot.get("input", 0),
        "orch_tokens_out":      ot.get("output", 0),
        "agent_tokens_in":      tt.get("agents_input",  0),
        "agent_tokens_out":     tt.get("agents_output", 0),
        "grand_total_tokens":   tt.get("grand_total",   ot.get("input", 0) + ot.get("output", 0)),
        "n_nodes":              len(nodes),
        "n_edges":              len(edges),
        "n_cugcr_triggered":    cugcr_nodes,
        "cugcr_run":            et.get("cugcr_triggered", False),
        "n_failed_nodes":       failed_nodes,
        "failure_modes":        failure_mode_counts,
        "mean_node_quality":    _mean(quality_scores),
        "mean_node_lat_s":      _mean(node_latencies),
        "mean_context_tokens":  _mean(context_tokens),
        "total_context_tokens": _total(context_tokens),
        "node_tokens_in":       _total(tokens_in_list),
        "node_tokens_out":      _total(tokens_out_list),
        "mean_paper_count":     _mean(paper_counts),
        "mean_ctx_token_count": _mean(ctx_tok_list),
        "mean_utilization_score": _mean(util_list),
        "mean_context_relevance": _mean(rel_list),
        "first_pass_success":   et.get("first_pass_success"),
        "global_retries":       et.get("global_retries", 0),
        "profiler_used":        et.get("use_profiler"),
        "decomposer":           et.get("decomposer_used"),
        "selection_score":      selection_score(d),
    }


# ---------------------------------------------------------------------------
# LLM judge
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
    tasks  = [
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
# Print table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], top_n: int, folder: Path) -> None:
    print(f"\n{'═'*115}")
    print(f"  ADAPTIVE (best {top_n})   |   folder: {folder.name}   |   {len(rows)} runs")
    print(f"{'═'*115}")

    hdr = (
        f"  {'ID':<8} {'QID':<7} {'Sel':>5} "
        f"{'GS':>5} {'Grnd':>5} {'Spec':>5} {'Corr':>5} {'Comp':>5} {'Code':>5} {'Cohr':>5} "
        f"{'Lat(s)':>7} {'OTok':>6} {'Nodes':>6} {'NQS':>5} {'Ret':>4} {'1stP':>5}"
    )
    print(hdr)
    print(f"  {'─'*8} {'─'*7} {'─'*5} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*7} {'─'*6} {'─'*6} {'─'*5} {'─'*4} {'─'*5}")

    for r in rows:
        j   = r.get("judge", {})
        m   = r.get("metrics", {})
        gs  = j.get("gs_score", 0)
        ot  = (m.get("orch_tokens_in", 0) or 0) + (m.get("orch_tokens_out", 0) or 0)
        nq  = m.get("mean_node_quality")
        sel = m.get("selection_score", 0)
        fp  = "✓" if m.get("first_pass_success") else ("✗" if m.get("first_pass_success") is False else "—")
        print(
            f"  {m.get('run_id','?')[:8]:<8} {m.get('query_id','?'):<7} {sel:>5.3f} "
            f"{gs:>5.2f} {j.get('grounding',0):>5} {j.get('specificity',0):>5} "
            f"{j.get('correctness',0):>5} {j.get('completeness',0):>5} "
            f"{j.get('code_quality',0):>5} {j.get('coherence',0):>5} "
            f"{(m.get('latency_s') or 0):>7.1f} {ot:>6} "
            f"{(m.get('n_nodes') or 0):>6} "
            f"{f'{nq:.2f}' if nq is not None else '  — ':>5} "
            f"{(m.get('global_retries') or 0):>4} {fp:>5}"
        )

    # Summary block
    gs_scores  = [r["judge"].get("gs_score", 0) for r in rows]
    lat_vals   = [r["metrics"].get("latency_s") or 0 for r in rows]
    orch_toks  = [(r["metrics"].get("orch_tokens_in", 0) or 0) +
                  (r["metrics"].get("orch_tokens_out", 0) or 0) for r in rows]
    retries    = [r["metrics"].get("global_retries", 0) or 0 for r in rows]
    fp_vals    = [r["metrics"].get("first_pass_success") for r in rows]
    fp_rate    = sum(1 for x in fp_vals if x is True) / len(rows) * 100
    sel_scores = [r["metrics"].get("selection_score", 0) for r in rows]
    nq_vals    = [r["metrics"].get("mean_node_quality") for r in rows if r["metrics"].get("mean_node_quality") is not None]

    def mean(lst): return sum(lst) / len(lst) if lst else 0

    dim_means = {k: mean([r["judge"].get(k, 0) for r in rows]) for k in GS_WEIGHTS}

    print(f"\n{'─'*115}")
    print(f"  SUMMARY  ({len(rows)} best adaptive runs by mean node quality)")
    print(f"  {'Mean GS_Score':.<40} {mean(gs_scores):.3f} / 10")
    print(f"  {'Mean Grounding':.<40} {dim_means['grounding']:.2f}")
    print(f"  {'Mean Specificity':.<40} {dim_means['specificity']:.2f}")
    print(f"  {'Mean Correctness':.<40} {dim_means['correctness']:.2f}")
    print(f"  {'Mean Completeness':.<40} {dim_means['completeness']:.2f}")
    print(f"  {'Mean Code_Quality':.<40} {dim_means['code_quality']:.2f}")
    print(f"  {'Mean Coherence':.<40} {dim_means['coherence']:.2f}")
    print(f"  {'─'*55}")
    print(f"  {'Mean Selection Score (node quality)':.<40} {mean(sel_scores):.3f}")
    print(f"  {'Mean DAG Node Quality':.<40} {mean(nq_vals):.3f}" if nq_vals else "")
    print(f"  {'Mean Latency (s)':.<40} {mean(lat_vals):.2f}")
    print(f"  {'Mean Orch Tokens (in+out)':.<40} {mean(orch_toks):.0f}")
    print(f"  {'Total Orch Tokens':.<40} {sum(orch_toks)}")
    print(f"  {'Mean Global Retries':.<40} {mean(retries):.2f}")
    print(f"  {'First-Pass Success Rate':.<40} {fp_rate:.0f}%")
    print(f"{'═'*115}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(folder: Path, top_n: int, out_path: Path, concurrency: int, skip_judge: bool) -> None:
    print(f"\n── Best Adaptive Evaluation ─────────────────────────────────")
    print(f"  Folder : {folder}")
    print(f"  Top-N  : {top_n}")

    # 1. Load all adaptive files
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        sys.exit(1)

    all_adaptive = []
    for f in sorted(folder.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("ablation_mode") == "adaptive":
                all_adaptive.append((f, d))
        except Exception:
            pass

    print(f"  Adaptive files found: {len(all_adaptive)}")
    if not all_adaptive:
        print("  No adaptive telemetry found.")
        sys.exit(1)

    # 2. Rank by selection score and pick top N
    all_adaptive.sort(key=lambda x: selection_score(x[1]), reverse=True)
    selected = all_adaptive[:top_n]

    scores_all = [selection_score(d) for _, d in all_adaptive]
    scores_sel = [selection_score(d) for _, d in selected]
    print(f"  Selection score range (all): {min(scores_all):.3f} – {max(scores_all):.3f}")
    print(f"  Selection score range (top {top_n}): {min(scores_sel):.3f} – {max(scores_sel):.3f}\n")

    # 3. Build run records
    runs = []
    for f, d in selected:
        et = d.get("engine_telemetry", {})
        runs.append({
            "run_id":       d.get("run_id", f.stem),
            "query_id":     d.get("query_id", "?"),
            "query":        et.get("user_query", d.get("query", "")),
            "final_answer": et.get("final_answer", ""),
            "metrics":      extract_metrics(d),
        })

    # 4. Judge
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

    # 5. Merge
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

    # Sort by selection_score descending so table mirrors ranking
    rows.sort(key=lambda x: x["metrics"].get("selection_score", 0), reverse=True)

    # 6. Print
    print_table(rows, top_n, folder)

    # 7. Save JSON
    gs_scores = [r["judge"].get("gs_score", 0) for r in rows]
    dim_means = {k: sum(r["judge"].get(k, 0) for r in rows) / len(rows) for k in GS_WEIGHTS}
    lat_vals  = [r["metrics"].get("latency_s") or 0 for r in rows]
    orch_toks = [(r["metrics"].get("orch_tokens_in", 0) or 0) +
                 (r["metrics"].get("orch_tokens_out", 0) or 0) for r in rows]

    def mean(lst): return sum(lst) / len(lst) if lst else 0

    out_data = {
        "mode":        "adaptive",
        "folder":      str(folder),
        "top_n":       top_n,
        "n_runs":      len(rows),
        "judge_model": Config.JUDGE_MODEL,
        "summary": {
            "mean_gs_score":       round(mean(gs_scores), 3),
            "dim_means":           {k: round(v, 3) for k, v in dim_means.items()},
            "mean_latency_s":      round(mean(lat_vals), 2),
            "total_orch_tokens":   sum(orch_toks),
            "mean_orch_tokens":    round(mean(orch_toks)),
            "first_pass_rate_pct": round(
                sum(1 for r in rows if r["metrics"].get("first_pass_success") is True)
                / len(rows) * 100, 1),
            "mean_global_retries": round(
                mean([r["metrics"].get("global_retries", 0) or 0 for r in rows]), 2),
        },
        "runs": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"  → Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",      type=Path, default=None,
                        help="Telemetry folder (default: raw_telemetry_best/)")
    parser.add_argument("--top",         type=int, default=30,
                        help="Number of best runs to evaluate (default: 30)")
    parser.add_argument("--out",         type=Path, default=None,
                        help="Output JSON path (default: results/eval_best_adaptive.json)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel LLM judge calls (default: 5)")
    parser.add_argument("--skip-judge",  action="store_true",
                        help="Skip LLM judge, compute telemetry metrics only")
    args = parser.parse_args()

    folder   = args.folder or DEFAULT_FOLDER
    out_path = args.out    or (ROOT / "results" / "eval_best_adaptive.json")
    asyncio.run(main(folder, args.top, out_path, args.concurrency, args.skip_judge))
