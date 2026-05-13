"""
run_rag_evaluation.py
=====================
RAG-specific evaluation script.

Evaluates the RAG agent (/api/rag) on queries where RAG is the primary
or sole responsible agent — explanation, synthesis, and concept queries
extracted from the benchmark set.

Output: raw_telemetry/rag_eval/<run_id>.json
        raw_telemetry/rag_eval/_manifest.json
        raw_telemetry/rag_eval/_summary.json

Protocols:
  A. Strategy ablation     — same query × CoT / ToT / GoT
  B. Depth sensitivity     — same query × depth 1 / 2 / 3
  C. Consistency runs      — same query × 3 reps (stability)
  D. Cross-topic coverage  — one query per topic area
  E. Boundary conditions   — very short vs very long queries
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("raw_telemetry/rag_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── API config ────────────────────────────────────────────────────────────────
API_BASE     = "http://localhost:8000"
RAG_ENDPOINT = f"{API_BASE}/api/rag"

INTER_RUN_SLEEP = 3    # seconds between calls (Gemini rate limit)
ERROR_SLEEP     = 15   # seconds after a failed call

# ── Query banks ───────────────────────────────────────────────────────────────

# Pure RAG queries — explanation / synthesis only, no code or cloud
RAG_BENCHMARK_QUERIES = [
    # From Group 6 — single-agent baselines
    {
        "id": "RB01",
        "group": "baseline",
        "query": "What is the standard statistical method used by Scanpy to find marker genes across cell clusters?",
    },
    {
        "id": "RB02",
        "group": "baseline",
        "query": "Describe the tumor microenvironment in breast cancer based on recent scRNA-seq studies.",
    },

    # From Group 1 — explanation portions of full-stack queries
    {
        "id": "RB03",
        "group": "explanation",
        "query": "Explain the Leiden community detection algorithm for single-cell RNA-seq.",
    },
    {
        "id": "RB04",
        "group": "explanation",
        "query": "What are the mechanistic differences between PCA and UMAP in single-cell analysis?",
    },
    {
        "id": "RB05",
        "group": "explanation",
        "query": "Explain how AlphaFold 3 predicts protein structures.",
    },

    # From Group 3 — context overload explanation portions
    {
        "id": "RB06",
        "group": "synthesis",
        "query": "Compare the Needleman-Wunsch and Smith-Waterman sequence alignment algorithms.",
    },
    {
        "id": "RB07",
        "group": "synthesis",
        "query": "What is the standard file format for storing unaligned raw sequencing reads and what is the mathematical basis of its quality scores?",
    },
    {
        "id": "RB08",
        "group": "synthesis",
        "query": "Explain the biological function of AMPA receptor trafficking in long-term potentiation.",
    },
    {
        "id": "RB09",
        "group": "synthesis",
        "query": "Detail the algorithmic differences between Kallisto pseudoalignment and standard RNA-seq alignment.",
    },

    # From Group 2 — explanation after expert finding
    {
        "id": "RB10",
        "group": "coreference_explanation",
        "query": "Explain the primary detection methods for CRISPR base editing off-target effects.",
    },
    {
        "id": "RB11",
        "group": "coreference_explanation",
        "query": "How are graph neural networks applied to protein structures in biology?",
    },
    {
        "id": "RB12",
        "group": "coreference_explanation",
        "query": "Summarize the main findings on calcium signaling in synaptic plasticity STDP in the hippocampus.",
    },

    # From Group 4 — missing data / hard explanations
    {
        "id": "RB13",
        "group": "missing_data",
        "query": "Explain how the Harmony algorithm works for batch effect correction in single-cell RNA-seq.",
    },
    {
        "id": "RB14",
        "group": "missing_data",
        "query": "Compare the pros and cons of AlphaFold 3 versus Rosetta for protein structure prediction.",
    },
    {
        "id": "RB15",
        "group": "missing_data",
        "query": "Explain how the MAGIC imputation algorithm works for single-cell data.",
    },
]

# Boundary condition queries
BOUNDARY_QUERIES = [
    {"id": "BD01", "group": "boundary_short",  "query": "What does UMAP do?"},
    {"id": "BD02", "group": "boundary_short",  "query": "What is a FASTQ file?"},
    {"id": "BD03", "group": "boundary_long",
     "query": (
         "Provide a comprehensive comparison of all major dimensionality reduction methods "
         "used in single-cell RNA-seq analysis including PCA, UMAP, t-SNE, and diffusion maps, "
         "covering their mathematical foundations, computational complexity, preservation of "
         "local vs global structure, and typical use cases in the Scanpy workflow."
     )},
]

# Strategy ablation — run these with all 3 strategies
STRATEGY_ABLATION_QUERIES = [
    RAG_BENCHMARK_QUERIES[0],   # RB01 — marker genes (baseline)
    RAG_BENCHMARK_QUERIES[5],   # RB06 — sequence alignment (synthesis)
    RAG_BENCHMARK_QUERIES[9],   # RB10 — CRISPR off-target (coreference)
]

# Consistency queries — run 3x each
CONSISTENCY_QUERIES = [
    RAG_BENCHMARK_QUERIES[0],   # RB01
    RAG_BENCHMARK_QUERIES[3],   # RB04
    RAG_BENCHMARK_QUERIES[7],   # RB08
]

# ── Manifest ──────────────────────────────────────────────────────────────────
manifest: list[dict] = []
summary_stats: dict = {
    "total_runs": 0,
    "successful": 0,
    "failed": 0,
    "by_strategy": {"cot": [], "tot": [], "got": []},
    "by_group": {},
    "by_depth": {1: [], 2: [], 3: []},
}


# ── API call ──────────────────────────────────────────────────────────────────

async def call_rag(
    session: aiohttp.ClientSession,
    query: str,
    strategy: str = "cot",
    depth: int = 2,
    complexity: str = "intermediate",
) -> dict:
    payload = {
        "query": query,
        "strategy": strategy,
        "depth": depth,
        "complexity": complexity,
    }
    async with session.post(
        RAG_ENDPOINT,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json()


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_run(result: dict, metadata: dict, latency_ms: int, error: str = "") -> str:
    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id":     run_id,
        "timestamp":  datetime.utcnow().isoformat(),
        "latency_ms": latency_ms,
        "error":      error,
        "failed":     bool(error),
        **metadata,
        "result":     result,
    }
    out_path = OUTPUT_DIR / f"run_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    manifest.append({
        "run_id":   run_id,
        "file":     str(out_path),
        "protocol": metadata.get("protocol"),
        "query_id": metadata.get("query_id"),
        "strategy": metadata.get("strategy"),
        "depth":    metadata.get("depth"),
        "failed":   bool(error),
        "latency_ms": latency_ms,
    })

    # Update summary stats
    summary_stats["total_runs"] += 1
    group = metadata.get("group", "unknown")

    if error:
        summary_stats["failed"] += 1
    else:
        summary_stats["successful"] += 1
        strategy = metadata.get("strategy", "cot")
        depth = metadata.get("depth", 1)
        quality = result.get("quality", "")

        # Parse score from "Score: X/10 - ..."
        score = _parse_score(quality)
        if score is not None:
            summary_stats["by_strategy"][strategy].append(score)
            summary_stats["by_depth"][depth].append(score)
            if group not in summary_stats["by_group"]:
                summary_stats["by_group"][group] = []
            summary_stats["by_group"][group].append(score)

    return run_id


def _parse_score(quality: str):
    import re
    # Match "Score: 7/10" or "Score: 7" or "7/10"
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", quality)
    if match:
        return float(match.group(1))
    # Fallback — bare number after "Score:"
    match = re.search(r"[Ss]core:\s*(\d+(?:\.\d+)?)", quality)
    if match:
        val = float(match.group(1))
        return val if val <= 10 else None
    return None


def flush_manifest():
    with open(OUTPUT_DIR / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def flush_summary():
    # Compute averages
    final = {
        "total_runs":  summary_stats["total_runs"],
        "successful":  summary_stats["successful"],
        "failed":      summary_stats["failed"],
        "success_rate": round(
            summary_stats["successful"] / max(summary_stats["total_runs"], 1) * 100, 1
        ),
        "mean_score_by_strategy": {
            k: round(sum(v) / len(v), 2) if v else None
            for k, v in summary_stats["by_strategy"].items()
        },
        "mean_score_by_depth": {
            k: round(sum(v) / len(v), 2) if v else None
            for k, v in summary_stats["by_depth"].items()
        },
        "mean_score_by_group": {
            k: round(sum(v) / len(v), 2) if v else None
            for k, v in summary_stats["by_group"].items()
        },
    }
    with open(OUTPUT_DIR / "_summary.json", "w") as f:
        json.dump(final, f, indent=2)
    return final


async def execute_run(
    session: aiohttp.ClientSession,
    query_meta: dict,
    strategy: str,
    depth: int,
    protocol: str,
    complexity: str = "intermediate",
    sleep_after: int = INTER_RUN_SLEEP,
) -> bool:
    query = query_meta["query"]
    query_id = query_meta["id"]
    group = query_meta.get("group", "unknown")

    print(f"  ▶  [{query_id}] strategy={strategy} depth={depth} | {query[:70]}...")

    metadata = {
        "protocol": protocol,
        "query_id": query_id,
        "group":    group,
        "strategy": strategy,
        "depth":    depth,
        "query":    query,
        "complexity": complexity,
    }

    try:
        start = time.time()
        result = await call_rag(session, query, strategy, depth, complexity)
        latency_ms = int((time.time() - start) * 1000)

        run_id = save_run(result, metadata, latency_ms)
        score_str = result.get("quality", "N/A")[:50]
        print(f"  ✔  run_{run_id}.json  ({latency_ms}ms)  quality: {score_str}")
        await asyncio.sleep(sleep_after)
        return True

    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000) if 'start' in dir() else 0
        print(f"  ✘  error: {exc}")
        save_run({}, metadata, latency_ms, error=str(exc))
        await asyncio.sleep(ERROR_SLEEP)
        return False


# ── Protocol A: Strategy ablation ────────────────────────────────────────────

async def protocol_a(session: aiohttp.ClientSession):
    """
    Run selected queries under CoT / ToT / GoT at fixed depth=2.
    Measures: which strategy produces highest quality scores per query type.
    """
    print("\n" + "="*60)
    print("PROTOCOL A — Strategy ablation (CoT / ToT / GoT, depth=2)")
    print("="*60)

    for q in STRATEGY_ABLATION_QUERIES:
        print(f"\n── {q['id']} — {q['query'][:60]}")
        for strategy in ["cot", "tot", "got"]:
            await execute_run(session, q, strategy, depth=2, protocol="A_strategy_ablation")


# ── Protocol B: Depth sensitivity ────────────────────────────────────────────

async def protocol_b(session: aiohttp.ClientSession):
    """
    Run selected queries at depth 1 / 2 / 3 with fixed strategy=cot.
    Measures: does deeper reasoning improve quality? At what cost in latency?
    """
    print("\n" + "="*60)
    print("PROTOCOL B — Depth sensitivity (depth 1/2/3, strategy=cot)")
    print("="*60)

    depth_queries = RAG_BENCHMARK_QUERIES[:5]   # first 5 queries
    for q in depth_queries:
        print(f"\n── {q['id']} — {q['query'][:60]}")
        for depth in [1, 2, 3]:
            await execute_run(session, q, strategy="cot", depth=depth,
                              protocol="B_depth_sensitivity")


# ── Protocol C: Consistency / stability ──────────────────────────────────────

async def protocol_c(session: aiohttp.ClientSession):
    """
    Run same queries 3x with fixed strategy=cot, depth=2.
    Measures: answer stability, quality variance across runs.
    """
    print("\n" + "="*60)
    print("PROTOCOL C — Consistency runs (3× per query, cot depth=2)")
    print("="*60)

    for q in CONSISTENCY_QUERIES:
        print(f"\n── {q['id']} — {q['query'][:60]}")
        for rep in range(3):
            print(f"   Rep {rep+1}/3")
            await execute_run(session, q, strategy="cot", depth=2,
                              protocol="C_consistency")


# ── Protocol D: Cross-topic coverage ─────────────────────────────────────────

async def protocol_d(session: aiohttp.ClientSession):
    """
    Run all 15 RAG benchmark queries once with cot depth=2.
    Measures: FPSR across all topic areas and query groups.
    """
    print("\n" + "="*60)
    print("PROTOCOL D — Cross-topic coverage (all queries, cot depth=2)")
    print("="*60)

    for q in RAG_BENCHMARK_QUERIES:
        print(f"\n── {q['id']} ({q['group']})")
        await execute_run(session, q, strategy="cot", depth=2,
                          protocol="D_cross_topic")


# ── Protocol E: Boundary conditions ──────────────────────────────────────────

async def protocol_e(session: aiohttp.ClientSession):
    """
    Run very short and very long queries.
    Measures: performance at query length extremes.
    """
    print("\n" + "="*60)
    print("PROTOCOL E — Boundary conditions (short/long queries)")
    print("="*60)

    for q in BOUNDARY_QUERIES:
        print(f"\n── {q['id']} ({q['group']})")
        for strategy in ["cot", "tot"]:
            await execute_run(session, q, strategy=strategy, depth=1,
                              protocol="E_boundary")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         RAG AGENT EVALUATION — DATA COLLECTION          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Output directory : {OUTPUT_DIR.resolve()}")
    print(f"API endpoint     : {RAG_ENDPOINT}")
    print(f"Started          : {datetime.utcnow().isoformat()}")
    print(f"Total queries    : {len(RAG_BENCHMARK_QUERIES)} benchmark + {len(BOUNDARY_QUERIES)} boundary")

    # Check server is up
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE}/api/health",
                                   timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    print(f"\n✘ Server not responding at {API_BASE}. Start with: python main.py")
                    return
        except Exception:
            print(f"\n✘ Cannot reach {API_BASE}. Start the server first: python main.py")
            return

    print("\n✔ Server is up. Starting evaluation...\n")

    async with aiohttp.ClientSession() as session:
        await protocol_a(session)   # 3 queries × 3 strategies = 9 runs
        await protocol_b(session)   # 5 queries × 3 depths     = 15 runs
        await protocol_c(session)   # 3 queries × 3 reps       = 9 runs
        await protocol_d(session)   # 15 queries × 1           = 15 runs
        await protocol_e(session)   # 3 queries × 2 strategies = 6 runs
        # Total: ~54 runs

    flush_manifest()
    summary = flush_summary()

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"  Total runs    : {summary['total_runs']}")
    print(f"  Successful    : {summary['successful']}")
    print(f"  Failed        : {summary['failed']}")
    print(f"  Success rate  : {summary['success_rate']}%")
    print(f"\n  Mean score by strategy:")
    for s, v in summary["mean_score_by_strategy"].items():
        print(f"    {s.upper():5s} : {v if v is not None else 'N/A'}")
    print(f"\n  Mean score by depth:")
    for d, v in summary["mean_score_by_depth"].items():
        print(f"    depth={d} : {v if v is not None else 'N/A'}")
    print(f"\n  Mean score by query group:")
    for g, v in summary["mean_score_by_group"].items():
        print(f"    {g:30s} : {v if v is not None else 'N/A'}")
    print(f"\n  Results saved to: {OUTPUT_DIR.resolve()}")
    print(f"  Manifest : {OUTPUT_DIR / '_manifest.json'}")
    print(f"  Summary  : {OUTPUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())