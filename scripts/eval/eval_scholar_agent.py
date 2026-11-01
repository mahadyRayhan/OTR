"""
eval_scholar_agent.py — Standalone evaluation for the Scholar Agent
====================================================================
Criteria:
  - LLM-as-Judge (0-10): relevance, completeness, result quality
  - Latency (wall seconds per query)

No ground truth required — scholar searches live API (Semantic Scholar).

Usage:
    python scripts/eval/eval_scholar_agent.py
    python scripts/eval/eval_scholar_agent.py --queries 5 --out results/scholar_eval.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from scholar_agent.agent import call_scholar_agent

# ---------------------------------------------------------------------------
# Test queries — mix of specific and broad, expert-search and collaboration
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # G1: Specific bioinformatics topics
    {"id": "SQ01", "query": "Find experts in single-cell RNA sequencing trajectory analysis", "is_collab": False, "limit": 3},
    {"id": "SQ02", "query": "Who are the leading researchers in spatial transcriptomics?", "is_collab": False, "limit": 3},
    {"id": "SQ03", "query": "Find experts in chromatin accessibility and ATAC-seq data analysis", "is_collab": False, "limit": 3},
    # G2: Multi-omics / integration
    {"id": "SQ04", "query": "Find experts working on multi-omics data integration methods", "is_collab": False, "limit": 3},
    {"id": "SQ05", "query": "Who publishes on integrating proteomics and transcriptomics?", "is_collab": False, "limit": 3},
    # G3: Computational methods
    {"id": "SQ06", "query": "Find researchers developing deep learning models for genomics", "is_collab": False, "limit": 3},
    {"id": "SQ07", "query": "Who works on graph neural networks for protein structure prediction?", "is_collab": False, "limit": 3},
    # G4: Collaboration mode
    {"id": "SQ08", "query": "I need a collaborator working on cancer single-cell sequencing and tumor microenvironment", "is_collab": True, "limit": 2},
    {"id": "SQ09", "query": "Find a collaborator in CRISPR screen analysis and functional genomics", "is_collab": True, "limit": 2},
    # G5: Broad / stress
    {"id": "SQ10", "query": "Find experts in bioinformatics tools for Oxford Nanopore long-read sequencing", "is_collab": False, "limit": 3},
]

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a Scholar Agent response.

The agent was given this query:
"{query}"

The agent returned:
<response>
{response}
</response>

Score the response on a scale of 0–10 using ONLY these criteria:
1. **Relevance** (0-4): Are the returned experts/papers directly relevant to the query topic?
2. **Completeness** (0-3): Does the response include key details (names, institutions, papers, citation counts)?
3. **Usefulness** (0-3): Would a researcher actually find this response helpful for their needs?

Important notes:
- A response with real expert names and paper titles should score higher even without perfect detail.
- A "[SHORTFALL]" marker is acceptable if the API returned fewer results than requested — do not penalize heavily.
- An empty response or pure error message scores 0.
- A mock/placeholder response (no API key) scores 0.

Return your evaluation in this exact JSON format:
{{
  "relevance": <0-4>,
  "completeness": <0-3>,
  "usefulness": <0-3>,
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
    prompt = JUDGE_PROMPT.format(query=query, response=response[:3000])
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
        return {"relevance": 0, "completeness": 0, "usefulness": 0, "total": 0, "rationale": f"Judge error: {exc}"}


async def run_single(query_item: dict, client: AsyncOpenAI) -> dict:
    qid   = query_item["id"]
    query = query_item["query"]
    is_collab = query_item.get("is_collab", False)
    limit = query_item.get("limit", 3)

    print(f"  [{qid}] {query[:60]}...")
    t0 = time.monotonic()
    try:
        response, _ = await call_scholar_agent(
            user_input=query,
            is_collaboration=is_collab,
            limit=limit,
        )
        error = None
    except Exception as exc:
        response = f"[ERROR] {exc}"
        error = str(exc)
    latency = time.monotonic() - t0

    scores = await judge_response(client, query, response)
    passed = scores.get("total", 0) >= 7
    cache_hit = "[CACHE-SUPPLEMENTED" in response

    print(f"    score={scores['total']}/10  latency={latency:.1f}s  cache={'yes' if cache_hit else 'no'}  {'PASS' if passed else 'FAIL'}")
    print(f"    rationale: {scores.get('rationale', '')}")

    return {
        "query_id":     qid,
        "query":        query,
        "is_collab":    is_collab,
        "latency_s":    round(latency, 2),
        "response_len": len(response),
        "judge":        scores,
        "cache_hit":    cache_hit,
        "passed":       passed,
        "error":        error,
    }


async def main(n_queries: int, out_path: Path):
    print("\n── Scholar Agent Evaluation ─────────────────────────────────")

    client = get_judge_client()
    queries = TEST_QUERIES[:n_queries]

    results = []
    for q in queries:
        r = await run_single(q, client)
        results.append(r)

    # Summary
    pass_rate   = sum(r["passed"] for r in results) / len(results) * 100
    mean_score  = sum(r["judge"]["total"] for r in results) / len(results)
    mean_lat    = sum(r["latency_s"] for r in results) / len(results)
    cache_pct   = sum(r["cache_hit"] for r in results) / len(results) * 100

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Queries run   : {len(results)}")
    print(f"  Pass rate     : {pass_rate:.0f}%  (threshold ≥ 7/10)")
    print(f"  Mean score    : {mean_score:.2f} / 10")
    print(f"  Cache used    : {cache_pct:.0f}%  of queries needed cache fallback")
    print(f"  Mean latency  : {mean_lat:.1f} s")

    # Per-query table
    print(f"\n  {'ID':<6} {'Score':>6} {'Cache':>6} {'Lat(s)':>7} {'Pass':>5}  Rationale")
    print(f"  {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*5}  {'─'*40}")
    for r in results:
        cache_flag = "yes" if r["cache_hit"] else "no"
        print(f"  {r['query_id']:<6} {r['judge']['total']:>6} {cache_flag:>6} {r['latency_s']:>7.1f} {'✓' if r['passed'] else '✗':>5}  {r['judge'].get('rationale','')[:50]}")

    # Save
    out = {
        "agent": "scholar",
        "n_queries": len(results),
        "pass_rate_pct": round(pass_rate, 1),
        "mean_judge_score": round(mean_score, 2),
        "cache_fallback_pct": round(cache_pct, 1),
        "mean_latency_s": round(mean_lat, 2),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  → {out_path}")
    print("Scholar eval done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES),
                        help="Number of test queries to run (default: all)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "eval_scholar.json")
    args = parser.parse_args()
    asyncio.run(main(args.queries, args.out))


# ── Summary ──────────────────────────────────────────────────
#   Queries run   : 10
#   Pass rate     : 60%  (threshold ≥ 7/10)
#   Mean score    : 6.40 / 10
#   Cache used    : 100%  of queries needed cache fallback
#   Mean latency  : 9.1 s

#   ID      Score  Cache  Lat(s)  Pass  Rationale
#   ────── ────── ────── ─────── ─────  ────────────────────────────────────────
#   SQ01        7    yes    17.3     ✓  The response lists real researchers and highly rel
#   SQ02        5    yes    13.3     ✗  The response includes real-looking names, papers, 
#   SQ03        8    yes     6.7     ✓  The response identifies highly relevant ATAC-seq/c
#   SQ04        5    yes     9.4     ✗  The response includes real names and papers relate
#   SQ05        5    yes    10.2     ✗  The response includes real researchers, paper titl
#   SQ06        7    yes     7.7     ✓  The response lists real-seeming researchers and cl
#   SQ07        5    yes     5.6     ✗  The response includes real names and papers relate
#   SQ08        8    yes     5.1     ✓  The response lists real researchers and highly rel
#   SQ09        7    yes     8.2     ✓  The response names real researchers and relevant C
#   SQ10        7    yes     7.2     ✓  The response names real Oxford Nanopore-associate

# -------------------- Latest Evaluation (2024-06-10) --------------------
#  Summary ──────────────────────────────────────────────────
#   Queries run   : 10
#   Pass rate     : 50%  (threshold ≥ 7/10)
#   Mean score    : 6.00 / 10
#   Cache used    : 100%  of queries needed cache fallback
#   Mean latency  : 3.8 s

#   ID      Score  Cache  Lat(s)  Pass  Rationale
#   ────── ────── ────── ─────── ─────  ────────────────────────────────────────
#   SQ01        7    yes    18.2     ✓  The response names real and mostly relevant single
#   SQ02        4    yes     2.3     ✗  The response includes some genuinely relevant spat
#   SQ03        7    yes     1.6     ✓  The response identifies genuinely relevant ATAC-se
#   SQ04        5    yes     2.1     ✗  The response includes real names and relevant pape
#   SQ05        5    yes     1.8     ✗  The response includes real researchers and papers 
#   SQ06        5    yes     1.8     ✗  The response is partly on-topic and includes names
#   SQ07        5    yes     2.2     ✗  The response includes real researchers, paper titl
#   SQ08        8    yes     2.1     ✓  The response names real researchers and highly rel
#   SQ09        7    yes     3.0     ✓  The response provides real researchers and relevan
#   SQ10        7    yes     3.2     ✓  The response names real and mostly relevant Oxford

# -------------- Latest Evaluation (2024-06-10) --------------------
# ── Summary ──────────────────────────────────────────────────
#   Queries run   : 10
#   Pass rate     : 60%  (threshold ≥ 7/10)
#   Mean score    : 6.20 / 10
#   Cache used    : 100%  of queries needed cache fallback
#   Mean latency  : 3.4 s

#   ID      Score  Cache  Lat(s)  Pass  Rationale
#   ────── ────── ────── ─────── ─────  ────────────────────────────────────────
#   SQ01        7    yes    17.4     ✓  The response includes real researchers, papers, ci
#   SQ02        5    yes     1.9     ✗  The response includes some genuinely relevant spat
#   SQ03        7    yes     1.8     ✓  The response identifies highly relevant ATAC-seq/c
#   SQ04        5    yes     1.8     ✗  The response includes real names and papers clearl
#   SQ05        5    yes     1.6     ✗  The response includes real researchers, paper titl
#   SQ06        7    yes     1.6     ✓  The response identifies plausible researchers and 
#   SQ07        4    yes     2.6     ✗  The response includes real names, papers, and cita
#   SQ08        8    yes     1.9     ✓  The response identifies real, highly relevant expe
#   SQ09        7    yes     1.9     ✓  The response provides real researchers and relevan
#   SQ10        7    yes     1.6     ✓  The response names real and mostly relevant Oxford