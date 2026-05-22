"""
eval_rag_agent.py — Standalone evaluation for the RAG Agent
============================================================
Criteria:
  - LLM-as-Judge (0-10): factual accuracy, literature grounding, completeness
  - Ground-truth keyword check: domain terms that should appear in a well-grounded answer
  - Retrieval check: paper_count > 0, no STRUCTURAL_FAIL marker
  - Latency (wall seconds per query)

Test queries span the topics in the RAG knowledge base:
  - Leiden / community detection
  - CRISPR base editing (ABE, CBE)
  - DESeq2 / bulk RNA-seq
  - Single-cell RNA-seq (Scanpy / Seurat / UMAP)
  - WNN multimodal integration
  - STAR splice-aware alignment

Usage:
    python scripts/eval/eval_rag_agent.py
    python scripts/eval/eval_rag_agent.py --queries 5 --out results/eval_rag.json
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
from rag_agent.agent import call_rag_agent
from rag_agent.knowledge_base import KnowledgeBase

# ---------------------------------------------------------------------------
# Ground-truth test suite
# expected_keywords are domain terms that must appear in a grounded answer.
# They are drawn from the actual topics in the RAG KB papers.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # ── Leiden / community detection ──────────────────────────────────────
    {
        "id": "RQ01",
        "query": "Explain how the Leiden algorithm guarantees well-connected communities compared to Louvain.",
        "expected_keywords": ["refinement", "Constant Potts Model", "modularity", "resolution"],
        "category": "community_detection",
    },
    {
        "id": "RQ02",
        "query": "What is the Constant Potts Model and why does it avoid the resolution limit in community detection?",
        "expected_keywords": ["CPM", "resolution", "partition", "gamma"],
        "category": "community_detection",
    },
    # ── CRISPR base editing ───────────────────────────────────────────────
    {
        "id": "RQ03",
        "query": "How does the ABE7.10 base editor achieve adenine-to-inosine deamination without double-strand breaks?",
        "expected_keywords": ["adenine", "deaminase", "TadA", "inosine", "Cas9"],
        "category": "crispr_base_editing",
    },
    {
        "id": "RQ04",
        "query": "Explain bystander editing in CRISPR base editing and how it affects specificity.",
        "expected_keywords": ["bystander", "editing window", "cytosine", "off-target"],
        "category": "crispr_base_editing",
    },
    {
        "id": "RQ05",
        "query": "What are the differences between CBE and ABE base editors in terms of mechanism and application?",
        "expected_keywords": ["cytosine", "adenine", "C-to-T", "A-to-G", "deaminase"],
        "category": "crispr_base_editing",
    },
    # ── DESeq2 / bulk RNA-seq ─────────────────────────────────────────────
    {
        "id": "RQ06",
        "query": "Explain the negative binomial statistical model used by DESeq2 for differential expression analysis.",
        "expected_keywords": ["negative binomial", "dispersion", "size factor", "Wald test"],
        "category": "deseq2",
    },
    {
        "id": "RQ07",
        "query": "Why does DESeq2 use shrinkage estimation for log fold change, and what problem does it solve?",
        "expected_keywords": ["shrinkage", "log fold change", "low count", "variance"],
        "category": "deseq2",
    },
    # ── Single-cell / Scanpy ──────────────────────────────────────────────
    {
        "id": "RQ08",
        "query": "Describe the Weighted Nearest Neighbor method for multimodal single-cell integration.",
        "expected_keywords": ["WNN", "modality", "kNN", "weight", "CITE-seq"],
        "category": "single_cell",
    },
    {
        "id": "RQ09",
        "query": "How does UMAP preserve local and global structure compared to t-SNE for single-cell data?",
        "expected_keywords": ["UMAP", "t-SNE", "manifold", "neighbor", "embedding"],
        "category": "single_cell",
    },
    # ── STAR alignment ────────────────────────────────────────────────────
    {
        "id": "RQ10",
        "query": "How does STAR handle chimeric and splice-aware alignment for RNA-seq reads?",
        "expected_keywords": ["STAR", "splice junction", "chimeric", "genome index", "seed"],
        "category": "star_alignment",
    },
]

JUDGE_PROMPT = """You are an expert bioinformatics scientist evaluating a RAG (Retrieval-Augmented Generation) agent response.

The agent was asked:
"{query}"

The agent returned:
<response>
{response}
</response>

Score the response 0–10 using ONLY these three criteria:

1. **Factual Accuracy** (0-4): Are the biological and computational claims scientifically correct? Penalise wrong mechanisms, wrong gene names, or contradictions.
2. **Literature Grounding** (0-3): Does the answer reference specific methods, tools, papers, or findings rather than generic LLM knowledge? A response that cites specific algorithms, parameter names, or paper titles scores higher.
3. **Completeness** (0-3): Does the answer fully address the specific aspect asked about? A partial answer that ignores a key part of the question scores lower.

Scoring notes:
- A precise, mechanism-level answer grounded in specific literature scores 8–10.
- A correct but superficial answer without specifics scores 5–7.
- A generic or partially wrong answer scores 3–5.
- An error marker (STRUCTURAL_FAIL) or empty answer scores 0–2.

Return ONLY this JSON (no other text):
{{
  "factual_accuracy": <0-4>,
  "literature_grounding": <0-3>,
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
        return {
            "factual_accuracy": 0, "literature_grounding": 0,
            "completeness": 0, "total": 0, "rationale": f"Judge error: {exc}",
        }


def keyword_check(response: str, keywords: list[str]) -> dict:
    hits = [kw for kw in keywords if kw.lower() in response.lower()]
    return {
        "hit":     len(hits),
        "total":   len(keywords),
        "rate":    len(hits) / len(keywords) if keywords else 1.0,
        "missing": [kw for kw in keywords if kw.lower() not in response.lower()],
    }


def structural_fail(response: str) -> bool:
    return response.strip().startswith("[STRUCTURAL_FAIL")


async def run_single(query_item: dict, kb: KnowledgeBase, client: AsyncOpenAI) -> dict:
    qid      = query_item["id"]
    query    = query_item["query"]
    keywords = query_item["expected_keywords"]

    print(f"  [{qid}] {query[:70]}...")
    t0 = time.monotonic()
    try:
        rag_data = await call_rag_agent(kb=kb, user_input=query, strategy="cot", depth=1)
        response     = rag_data.get("answer", "")
        paper_count  = rag_data.get("paper_count", 0)
        source_used  = rag_data.get("source_used", "")
        tokens       = rag_data.get("tokens", {"input": 0, "output": 0})
        error        = None
    except Exception as exc:
        response    = f"[ERROR] {exc}"
        paper_count = 0
        source_used = ""
        tokens      = {"input": 0, "output": 0}
        error       = str(exc)
    latency = time.monotonic() - t0

    scores   = await judge_response(client, query, response)
    kw       = keyword_check(response, keywords)
    is_fail  = structural_fail(response)
    passed   = scores.get("total", 0) >= 7 and not is_fail

    print(f"    score={scores['total']}/10  kw={kw['hit']}/{kw['total']}  papers={paper_count}  latency={latency:.1f}s  {'PASS' if passed else 'FAIL'}")
    if is_fail:
        print(f"    ✗ STRUCTURAL_FAIL — no papers retrieved or answer too short")
    if kw["missing"]:
        print(f"    missing keywords: {kw['missing']}")
    print(f"    rationale: {scores.get('rationale', '')}")

    return {
        "query_id":      qid,
        "category":      query_item["category"],
        "query":         query,
        "latency_s":     round(latency, 2),
        "paper_count":   paper_count,
        "source_used":   source_used,
        "response_len":  len(response),
        "tokens":        tokens,
        "judge":         scores,
        "keywords":      kw,
        "structural_fail": is_fail,
        "passed":        passed,
        "error":         error,
    }


async def main(n_queries: int, out_path: Path):
    print("\n── RAG Agent Evaluation ─────────────────────────────────────")
    print("  Loading knowledge base (FAISS + cross-encoder)...")
    kb = KnowledgeBase()
    kb.load()
    print("  KB loaded.\n")

    client  = get_judge_client()
    queries = TEST_QUERIES[:n_queries]

    results = []
    for q in queries:
        r = await run_single(q, kb, client)
        results.append(r)

    # Summary
    pass_rate    = sum(r["passed"] for r in results) / len(results) * 100
    mean_score   = sum(r["judge"]["total"] for r in results) / len(results)
    mean_lat     = sum(r["latency_s"] for r in results) / len(results)
    mean_kw_hit  = sum(r["keywords"]["rate"] for r in results) / len(results) * 100
    mean_papers  = sum(r["paper_count"] for r in results) / len(results)
    fail_pct     = sum(r["structural_fail"] for r in results) / len(results) * 100
    total_tok_in  = sum(r["tokens"].get("input", 0) for r in results)
    total_tok_out = sum(r["tokens"].get("output", 0) for r in results)

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Queries run          : {len(results)}")
    print(f"  Pass rate            : {pass_rate:.0f}%  (threshold ≥ 7/10, no STRUCTURAL_FAIL)")
    print(f"  Mean judge score     : {mean_score:.2f} / 10")
    print(f"  Mean keyword hit     : {mean_kw_hit:.0f}%  (ground-truth domain terms)")
    print(f"  Mean papers retrieved: {mean_papers:.1f}")
    print(f"  Structural fail rate : {fail_pct:.0f}%")
    print(f"  Mean latency         : {mean_lat:.1f} s")
    print(f"  Total tokens (in/out): {total_tok_in} / {total_tok_out}")

    print(f"\n  {'ID':<6} {'Score':>6} {'KW%':>5} {'Papers':>7} {'Lat(s)':>7} {'Fail':>5} {'Pass':>5}  Category")
    print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*7} {'─'*7} {'─'*5} {'─'*5}  {'─'*22}")
    for r in results:
        kw_pct = r["keywords"]["rate"] * 100
        f_mark = "✗" if r["structural_fail"] else " "
        print(
            f"  {r['query_id']:<6} {r['judge']['total']:>6} {kw_pct:>5.0f}%"
            f" {r['paper_count']:>7} {r['latency_s']:>7.1f}"
            f" {f_mark:>5} {'✓' if r['passed'] else '✗':>5}  {r['category']}"
        )

    out = {
        "agent":                    "rag",
        "n_queries":                len(results),
        "pass_rate_pct":            round(pass_rate, 1),
        "mean_judge_score":         round(mean_score, 2),
        "mean_keyword_hit_pct":     round(mean_kw_hit, 1),
        "mean_papers_retrieved":    round(mean_papers, 1),
        "structural_fail_pct":      round(fail_pct, 1),
        "mean_latency_s":           round(mean_lat, 2),
        "total_tokens_in":          total_tok_in,
        "total_tokens_out":         total_tok_out,
        "results":                  results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  → {out_path}")
    print("RAG eval done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "eval_rag.json")
    args = parser.parse_args()
    asyncio.run(main(args.queries, args.out))

#  Summary ──────────────────────────────────────────────────
#   Queries run          : 10
#   Pass rate            : 80%  (threshold ≥ 7/10, no STRUCTURAL_FAIL)
#   Mean judge score     : 7.90 / 10
#   Mean keyword hit     : 82%  (ground-truth domain terms)
#   Mean papers retrieved: 1.6
#   Structural fail rate : 0%
#   Mean latency         : 6.3 s
#   Total tokens (in/out): 8264 / 7007

#   ID      Score   KW%  Papers  Lat(s)  Fail  Pass  Category
#   ────── ────── ───── ─────── ─────── ───── ─────  ──────────────────────
#   RQ01        6    50%       1     6.2           ✗  community_detection
#   RQ02        9   100%       1     4.4           ✓  community_detection
#   RQ03        8    80%       2    20.4           ✓  crispr_base_editing
#   RQ04        8    75%       2     4.4           ✓  crispr_base_editing
#   RQ05        8    60%       1     5.1           ✓  crispr_base_editing
#   RQ06       10   100%       2     4.1           ✓  deseq2
#   RQ07        6   100%       1     4.3           ✗  deseq2
#   RQ08        9    80%       2     5.3           ✓  single_cell
#   RQ09        7   100%       2     4.4           ✓  single_cell
#   RQ10        8    80%       2     4.4           ✓  star_alignment