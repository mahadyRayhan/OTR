"""
eval_cloud_agent.py — Standalone evaluation for the Cloud Agent
================================================================
Criteria:
  - LLM-as-Judge (0-10): architecture quality, cost awareness, IaC correctness
  - Ground-truth keyword check: expected services/tools from the KB markdown docs
  - Latency (wall seconds per query)

Ground truth derives from cloud_agent/knowledge_base/*.md:
  - genomics_hpc_requirements.md
  - single_cell_hpc_requirements.md
  - machine_learning_llm_hpc_requirements.md
  - cloud_cost_optimization_strategies.md
  - os_and_container_standards.md
  - ai_structural_biology_requirements.md

Usage:
    python scripts/eval/eval_cloud_agent.py
    python scripts/eval/eval_cloud_agent.py --queries 5 --out results/cloud_eval.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from cloud_agent.agent import call_cloud_agent

# ---------------------------------------------------------------------------
# Ground-truth test suite
# expected_keywords come from the actual KB markdown documents.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # ── Genomics HPC ──────────────────────────────────────────────────────
    {
        "id": "CCQ01",
        "query": "Design a cloud HPC environment for running whole-genome sequencing alignment with BWA-MEM2 and variant calling with GATK HaplotypeCaller. Include compute, storage, and cost estimates.",
        "expected_keywords": ["spot", "S3", "EBS", "Docker", "Singularity"],
        "category": "genomics_hpc",
    },
    {
        "id": "CCQ02",
        "query": "What AWS instance type and storage should I use for running STAR aligner on bulk RNA-seq data for 50 samples?",
        "expected_keywords": ["r5", "S3", "EBS"],
        "category": "genomics_hpc",
    },
    # ── Single-cell ───────────────────────────────────────────────────────
    {
        "id": "CCQ03",
        "query": "Design an AWS architecture to run Cell Ranger for single-cell RNA-seq preprocessing on 20 10x Genomics samples. Include parallelization strategy.",
        "expected_keywords": ["spot", "S3", "paralleliz"],
        "category": "single_cell",
    },
    {
        "id": "CCQ04",
        "query": "I want to run Seurat or Scanpy downstream analysis on a cluster. Recommend an EC2 instance type and storage setup for 500k cells.",
        "expected_keywords": ["r5", "S3"],
        "category": "single_cell",
    },
    # ── ML / LLM HPC ─────────────────────────────────────────────────────
    {
        "id": "CCQ05",
        "query": "Provide IaC (Terraform or CloudFormation) to spin up a GPU cluster on AWS for training a protein language model.",
        "expected_keywords": ["p3", "p4", "FSx", "EFA"],
        "category": "ml_gpu",
    },
    {
        "id": "CCQ06",
        "query": "Design a cost-efficient cloud architecture for batch inference using a large language model for genomics report generation.",
        "expected_keywords": ["spot", "g4dn", "SageMaker"],
        "category": "ml_gpu",
    },
    # ── Cost optimization ─────────────────────────────────────────────────
    {
        "id": "CCQ07",
        "query": "What strategies should I use to cut AWS costs for a bioinformatics pipeline that runs overnight batch jobs daily?",
        "expected_keywords": ["spot", "Reserved", "S3 Intelligent"],
        "category": "cost_optimization",
    },
    {
        "id": "CCQ08",
        "query": "Design a lifecycle policy for bioinformatics data on S3 to minimize storage costs for raw FASTQ, processed BAM, and final results.",
        "expected_keywords": ["S3 Intelligent", "Glacier", "lifecycle"],
        "category": "cost_optimization",
    },
    # ── Containers / OS ───────────────────────────────────────────────────
    {
        "id": "CCQ09",
        "query": "What container and OS standards should I follow for deploying bioinformatics workflows on AWS to ensure reproducibility?",
        "expected_keywords": ["Docker", "Singularity", "Ubuntu"],
        "category": "containers",
    },
    # ── AI structural biology ─────────────────────────────────────────────
    {
        "id": "CCQ10",
        "query": "What GPU type and memory should I provision on AWS to run AlphaFold2 for a batch of 500 protein structure predictions?",
        "expected_keywords": ["p4d", "A100", "FSx"],
        "category": "ai_structural_biology",
    },
]

JUDGE_PROMPT = """You are an expert cloud architect evaluating a Cloud Agent response for bioinformatics workloads.

The agent was asked:
"{query}"

The agent returned:
<response>
{response}
</response>

Score the response 0–10 using ONLY these criteria:
1. **Architecture Quality** (0-4): Is the proposed architecture suitable for the workload? Are instance types, storage, and networking choices reasonable?
2. **Cost Awareness** (0-3): Does the response address cost (spot instances, reserved, storage tiers, or at minimum acknowledge trade-offs)?
3. **Actionability** (0-3): Is the response specific enough to implement? Does it include concrete service names, instance types, or IaC snippets?

Scoring notes:
- A specific, well-reasoned architecture with concrete AWS/GCP resources scores 8–10.
- A generic answer without specific instance types or services scores 4–6.
- An empty response, error, or pure disclaimer scores 0–2.
- An "AI EXTRAPOLATION WARNING" is acceptable and does NOT reduce score — it shows honesty.

Return ONLY this JSON (no other text):
{{
  "architecture_quality": <0-4>,
  "cost_awareness": <0-3>,
  "actionability": <0-3>,
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
        return {"architecture_quality": 0, "cost_awareness": 0, "actionability": 0, "total": 0, "rationale": f"Judge error: {exc}"}


def keyword_check(response: str, keywords: list[str]) -> dict:
    hits = [kw for kw in keywords if kw.lower() in response.lower()]
    return {
        "hit": len(hits),
        "total": len(keywords),
        "rate": len(hits) / len(keywords) if keywords else 1.0,
        "missing": [kw for kw in keywords if kw.lower() not in response.lower()],
    }


def extrapolation_warning_present(response: str) -> bool:
    return "AI EXTRAPOLATION WARNING" in response or "EXTRAPOLATION" in response.upper()


async def run_single(query_item: dict, client: AsyncOpenAI) -> dict:
    qid      = query_item["id"]
    query    = query_item["query"]
    keywords = query_item["expected_keywords"]

    print(f"  [{qid}] {query[:65]}...")
    t0 = time.monotonic()
    try:
        response = await call_cloud_agent(user_input=query)
        error = None
    except Exception as exc:
        response = f"[ERROR] {exc}"
        error = str(exc)
    latency = time.monotonic() - t0

    scores    = await judge_response(client, query, response)
    kw        = keyword_check(response, keywords)
    has_warn  = extrapolation_warning_present(response)
    passed    = scores.get("total", 0) >= 7

    print(f"    score={scores['total']}/10  kw={kw['hit']}/{kw['total']}  latency={latency:.1f}s  {'PASS' if passed else 'FAIL'}")
    if has_warn:
        print(f"    ⚠ Extrapolation warning present (tool not in KB)")
    if kw["missing"]:
        print(f"    missing keywords: {kw['missing']}")
    print(f"    rationale: {scores.get('rationale','')}")

    return {
        "query_id":            qid,
        "category":            query_item["category"],
        "query":               query,
        "latency_s":           round(latency, 2),
        "response_len":        len(response),
        "judge":               scores,
        "keywords":            kw,
        "extrapolation_warn":  has_warn,
        "passed":              passed,
        "error":               error,
    }


async def main(n_queries: int, out_path: Path):
    print("\n── Cloud Agent Evaluation ───────────────────────────────────")

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
    mean_kw_hit = sum(r["keywords"]["rate"] for r in results) / len(results) * 100
    extrap_pct  = sum(r["extrapolation_warn"] for r in results) / len(results) * 100

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Queries run          : {len(results)}")
    print(f"  Pass rate            : {pass_rate:.0f}%  (threshold ≥ 7/10)")
    print(f"  Mean judge score     : {mean_score:.2f} / 10")
    print(f"  Mean keyword hit     : {mean_kw_hit:.0f}%  (ground-truth KB terms)")
    print(f"  Extrapolation warns  : {extrap_pct:.0f}%  (tool not in KB)")
    print(f"  Mean latency         : {mean_lat:.1f} s")

    print(f"\n  {'ID':<7} {'Score':>6} {'KW%':>5} {'Lat(s)':>7} {'Warn':>5} {'Pass':>5}  Category")
    print(f"  {'─'*7} {'─'*6} {'─'*5} {'─'*7} {'─'*5} {'─'*5}  {'─'*22}")
    for r in results:
        kw_pct = r["keywords"]["rate"] * 100
        w = "⚠" if r["extrapolation_warn"] else " "
        print(f"  {r['query_id']:<7} {r['judge']['total']:>6} {kw_pct:>5.0f}% {r['latency_s']:>7.1f} {w:>5} {'✓' if r['passed'] else '✗':>5}  {r['category']}")

    # Save
    out = {
        "agent": "cloud",
        "n_queries": len(results),
        "pass_rate_pct": round(pass_rate, 1),
        "mean_judge_score": round(mean_score, 2),
        "mean_keyword_hit_pct": round(mean_kw_hit, 1),
        "mean_latency_s": round(mean_lat, 2),
        "extrapolation_warn_pct": round(extrap_pct, 1),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  → {out_path}")
    print("Cloud eval done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "eval_cloud.json")
    args = parser.parse_args()
    asyncio.run(main(args.queries, args.out))

# ── Summary ──────────────────────────────────────────────────
#   Queries run          : 10
#   Pass rate            : 100%  (threshold ≥ 7/10)
#   Mean judge score     : 8.90 / 10
#   Mean keyword hit     : 65%  (ground-truth KB terms)
#   Extrapolation warns  : 40%  (tool not in KB)
#   Mean latency         : 13.2 s

#   ID       Score   KW%  Lat(s)  Warn  Pass  Category
#   ─────── ────── ───── ─────── ───── ─────  ──────────────────────
#   CCQ01        9    80%    22.2     ⚠     ✓  genomics_hpc
#   CCQ02       10    67%    10.0           ✓  genomics_hpc
#   CCQ03       10   100%    14.3           ✓  single_cell
#   CCQ04       10    50%    11.1           ✓  single_cell
#   CCQ05        8    50%    14.8     ⚠     ✓  ml_gpu
#   CCQ06        7    33%    13.7     ⚠     ✓  ml_gpu
#   CCQ07        8    33%    13.6           ✓  cost_optimization
#   CCQ08        9   100%    10.7           ✓  cost_optimization
#   CCQ09        8   100%     9.6           ✓  containers
#   CCQ10       10    33%    11.8     ⚠     ✓  ai_structural_biology