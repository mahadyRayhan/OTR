"""
eval_code_agent.py — Standalone evaluation for the Code Agent
==============================================================
Criteria:
  - LLM-as-Judge (0-10): correctness, provenance adherence, runability
  - Ground-truth keyword check: expected scanpy functions/imports present
  - Latency (wall seconds per query)

Ground truth is derived from the knowledge base notebooks:
  code_agent/knowledge_base/*.ipynb
Each test query has expected_keywords — specific functions/API calls that
the KB contains and the agent SHOULD produce.

Usage:
    python scripts/eval/eval_code_agent.py
    python scripts/eval/eval_code_agent.py --queries 5 --out results/code_eval.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from code_agent.agent import call_code_agent, parse_provenance_ratio

# ---------------------------------------------------------------------------
# Ground-truth test suite
# Each query has: expected_keywords (functions/imports from the actual KB)
# These come directly from inspecting the knowledge_base *.ipynb notebooks.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "id": "CQ01",
        "query": "Write a Python script to perform basic single-cell RNA-seq clustering using Scanpy. Load an h5ad file, normalize, find highly variable genes, run PCA, compute neighbors, and run leiden clustering.",
        "expected_keywords": ["sc.pp.normalize_total", "sc.pp.highly_variable_genes", "sc.tl.pca", "sc.pp.neighbors", "sc.tl.leiden"],
        "category": "clustering_pipeline",
    },
    {
        "id": "CQ02",
        "query": "Generate a UMAP visualization for single-cell data after running PCA and neighbor computation. Include code to plot the UMAP colored by leiden cluster labels.",
        "expected_keywords": ["sc.tl.umap", "sc.pl.umap"],
        "category": "visualization",
    },
    {
        "id": "CQ03",
        "query": "Write code to preprocess UMI count data using analytic Pearson residuals with Scanpy.",
        "expected_keywords": ["sc.experimental.pp.highly_variable_genes", "sc.experimental.pp.normalize_pearson_residuals"],
        "category": "preprocessing_pearson",
    },
    {
        "id": "CQ04",
        "query": "Write a script to perform trajectory inference for hematopoiesis data using PAGA and diffusion pseudotime in Scanpy.",
        "expected_keywords": ["sc.tl.paga", "sc.tl.dpt", "sc.pp.neighbors"],
        "category": "trajectory",
    },
    {
        "id": "CQ05",
        "query": "Show how to integrate multiple single-cell datasets using Scanpy's ingest method to transfer labels from a reference dataset.",
        "expected_keywords": ["sc.tl.ingest"],
        "category": "integration",
    },
    {
        "id": "CQ06",
        "query": "Write scanpy code to compute and plot a dot plot showing marker gene expression per cluster.",
        "expected_keywords": ["sc.pl.dotplot"],
        "category": "visualization",
    },
    {
        "id": "CQ07",
        "query": "Generate code to filter cells and genes, compute QC metrics, and remove low-quality cells from a single-cell dataset.",
        "expected_keywords": ["sc.pp.filter_cells", "sc.pp.filter_genes", "sc.pp.calculate_qc_metrics"],
        "category": "qc_filtering",
    },
    {
        "id": "CQ08",
        "query": "Write a full scanpy pipeline: load data, QC, normalize, HVG selection, scale, PCA, neighbors, UMAP, and leiden clustering.",
        "expected_keywords": ["sc.pp.normalize_total", "sc.pp.highly_variable_genes", "sc.tl.pca", "sc.pp.neighbors", "sc.tl.umap", "sc.tl.leiden"],
        "category": "full_pipeline",
    },
    {
        "id": "CQ09",
        "query": "Show how to find marker genes for each cluster using rank_genes_groups and visualize the results.",
        "expected_keywords": ["sc.tl.rank_genes_groups", "sc.pl.rank_genes_groups"],
        "category": "marker_genes",
    },
    {
        "id": "CQ10",
        "query": "Write code to log-transform single-cell data and scale it to unit variance using Scanpy.",
        "expected_keywords": ["sc.pp.log1p", "sc.pp.scale"],
        "category": "normalization",
    },
]

JUDGE_PROMPT = """You are an expert bioinformatics code reviewer evaluating a Code Agent response.

The agent was asked:
"{query}"

The agent returned:
<response>
{response}
</response>

Score the response 0–10 using ONLY these criteria:
1. **Correctness** (0-4): Is the code syntactically valid Python? Does it use the right scanpy/anndata API calls for the task?
2. **Provenance** (0-3): Does the code reference real scanpy functions? Is it grounded in actual library usage (not hallucinated methods)?
3. **Completeness** (0-3): Does the code cover the full pipeline requested, with necessary imports and a logical flow?

Scoring notes:
- A well-structured script with correct scanpy calls scores 8–10.
- A script missing key steps or using wrong function names scores 4–6.
- A script with hallucinated functions (e.g. sc.pp.nonexistent) scores 0–3.
- An error message or empty response scores 0.

Return ONLY this JSON (no other text):
{{
  "correctness": <0-4>,
  "provenance": <0-3>,
  "completeness": <0-3>,
  "total": <0-10>,
  "rationale": "<one sentence>"
}}
"""


def get_judge_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return AsyncOpenAI(api_key=api_key)


async def judge_response(client: AsyncOpenAI, query: str, response: str) -> dict:
    prompt = JUDGE_PROMPT.format(query=query, response=response[:4000])
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
        return json.loads(text)
    except Exception as exc:
        return {"correctness": 0, "provenance": 0, "completeness": 0, "total": 0, "rationale": f"Judge error: {exc}"}


def keyword_check(response: str, keywords: list[str]) -> dict:
    """Check how many expected ground-truth keywords appear in the response."""
    hits = [kw for kw in keywords if kw in response]
    return {
        "hit": len(hits),
        "total": len(keywords),
        "rate": len(hits) / len(keywords) if keywords else 1.0,
        "missing": [kw for kw in keywords if kw not in response],
    }


async def run_single(query_item: dict, client: AsyncOpenAI) -> dict:
    qid      = query_item["id"]
    query    = query_item["query"]
    keywords = query_item["expected_keywords"]

    print(f"  [{qid}] {query[:65]}...")
    t0 = time.monotonic()
    try:
        response = await call_code_agent(user_input=query)
        error = None
    except Exception as exc:
        response = f"[ERROR] {exc}"
        error = str(exc)
    latency = time.monotonic() - t0

    scores = await judge_response(client, query, response)
    kw     = keyword_check(response, keywords)
    passed = scores.get("total", 0) >= 7

    # Extract code block for provenance analysis
    code_block = ""
    if "```python" in response:
        code_block = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code_block = response.split("```")[1].split("```")[0]
    prov = parse_provenance_ratio(code_block) if code_block else {"ai_pct": 0.0, "kb_pct": 0.0, "ai_lines": 0, "kb_lines": 0, "total_code_lines": 0, "untagged_lines": 0}

    print(f"    score={scores['total']}/10  kw={kw['hit']}/{kw['total']}  AI={prov['ai_pct']}%  latency={latency:.1f}s  {'PASS' if passed else 'FAIL'}")
    if kw["missing"]:
        print(f"    missing keywords: {kw['missing']}")
    print(f"    rationale: {scores.get('rationale','')}")

    return {
        "query_id":     qid,
        "category":     query_item["category"],
        "query":        query,
        "latency_s":    round(latency, 2),
        "response_len": len(response),
        "judge":        scores,
        "keywords":     kw,
        "provenance":   prov,
        "passed":       passed,
        "error":        error,
    }


async def main(n_queries: int, out_path: Path):
    print("\n── Code Agent Evaluation ────────────────────────────────────")

    client = get_judge_client()
    queries = TEST_QUERIES[:n_queries]

    results = []
    for q in queries:
        r = await run_single(q, client)
        results.append(r)

    # Summary
    pass_rate    = sum(r["passed"] for r in results) / len(results) * 100
    mean_score   = sum(r["judge"]["total"] for r in results) / len(results)
    mean_lat     = sum(r["latency_s"] for r in results) / len(results)
    mean_kw_hit  = sum(r["keywords"]["rate"] for r in results) / len(results) * 100
    mean_ai_pct  = sum(r["provenance"]["ai_pct"] for r in results) / len(results)
    mean_kb_pct  = sum(r["provenance"]["kb_pct"] for r in results) / len(results)

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Queries run        : {len(results)}")
    print(f"  Pass rate          : {pass_rate:.0f}%  (threshold ≥ 7/10)")
    print(f"  Mean judge score   : {mean_score:.2f} / 10")
    print(f"  Mean keyword hit   : {mean_kw_hit:.0f}%  (ground-truth functions)")
    print(f"  Mean AI-generated  : {mean_ai_pct:.1f}%  of code lines")
    print(f"  Mean KB-sourced    : {mean_kb_pct:.1f}%  of code lines")
    print(f"  Mean latency       : {mean_lat:.1f} s")

    print(f"\n  {'ID':<6} {'Score':>6} {'KW%':>5} {'AI%':>5} {'KB%':>5} {'Lat(s)':>7} {'Pass':>5}  Category")
    print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*5}  {'─'*20}")
    for r in results:
        kw_pct = r["keywords"]["rate"] * 100
        ai_pct = r["provenance"]["ai_pct"]
        kb_pct = r["provenance"]["kb_pct"]
        print(f"  {r['query_id']:<6} {r['judge']['total']:>6} {kw_pct:>5.0f}% {ai_pct:>4.0f}% {kb_pct:>4.0f}% {r['latency_s']:>7.1f} {'✓' if r['passed'] else '✗':>5}  {r['category']}")

    # Save
    out = {
        "agent": "code",
        "n_queries": len(results),
        "pass_rate_pct": round(pass_rate, 1),
        "mean_judge_score": round(mean_score, 2),
        "mean_keyword_hit_pct": round(mean_kw_hit, 1),
        "mean_ai_generated_pct": round(mean_ai_pct, 1),
        "mean_kb_sourced_pct": round(mean_kb_pct, 1),
        "mean_latency_s": round(mean_lat, 2),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  → {out_path}")
    print("Code eval done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "eval_code.json")
    args = parser.parse_args()
    asyncio.run(main(args.queries, args.out))

# ── Summary ──────────────────────────────────────────────────
#   Queries run        : 10
#   Pass rate          : 100%  (threshold ≥ 7/10)
#   Mean judge score   : 9.40 / 10
#   Mean keyword hit   : 95%  (ground-truth functions)
#   Mean AI-generated  : 43.1%  of code lines
#   Mean KB-sourced    : 56.9%  of code lines
#   Mean latency       : 13.5 s

#   ID      Score   KW%   AI%   KB%  Lat(s)  Pass  Category
#   ────── ────── ───── ───── ───── ─────── ─────  ────────────────────
#   CQ01       10   100%   92%    8%    24.0     ✓  clustering_pipeline
#   CQ02       10   100%   44%   56%    10.1     ✓  visualization
#   CQ03       10    50%   29%   71%    15.5     ✓  preprocessing_pearson
#   CQ04        9   100%   57%   43%     9.7     ✓  trajectory
#   CQ05        8   100%   58%   42%    13.3     ✓  integration
#   CQ06       10   100%   23%   77%    11.7     ✓  visualization
#   CQ07       10   100%   38%   62%     8.6     ✓  qc_filtering
#   CQ08       10   100%   62%   38%    12.7     ✓  full_pipeline
#   CQ09        8   100%   29%   71%    13.9     ✓  marker_genes
#   CQ10        9   100%    0%  100%    15.9     ✓  normalization