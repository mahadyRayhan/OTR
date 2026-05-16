"""
run_profiler_ablation.py
========================
Standalone Profiler Ablation Study.

For each query, runs adaptive mode TWICE:
  1. profiler=ON  — style guideline injected into synthesizer and cloud agent
  2. profiler=OFF — no style guideline (empty string), profiler LLM call skipped

The DAG is decomposed ONCE per query and locked, so both conditions execute
the identical task graph. Only the synthesizer prompt changes.

Query set (8 queries) covers all three profiler dimensions:
  PROCEDURAL   (implement, write, script)
  ANALYTICAL   (compare, evaluate, trade-off)
  CONCEPTUAL   (explain, what is, theory)
  MIXED        (expert + cloud, theory + code)

Output: raw_telemetry_profiler_ablation/run_*.json
  Each file has engine_telemetry.use_profiler = True | False
  and engine_telemetry.profiler_state for ON runs.

Analysis:
  python scripts-dev/05_profiler_ablation.py
      --dir raw_telemetry_profiler_ablation
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import config
from profiler_agent.agent import user_profiler
from orchestrator.engine import OrchestratorEngine

# ---------------------------------------------------------------------------
# Queries — 8 items spanning all three profiler dimensions
# ---------------------------------------------------------------------------
PROFILER_QUERIES = [
    # PROCEDURAL ──────────────────────────────────────────────────────────────
    (
        "PA01",
        "PROCEDURAL",
        "Write a Python script using pysam to parse a BAM file, filter reads with MAPQ above 30, "
        "and print per-chromosome read counts.",
    ),
    (
        "PA02",
        "PROCEDURAL",
        "Write a Snakemake workflow for GATK variant calling that runs BQSR and HaplotypeCaller "
        "on paired-end WGS samples and outputs a filtered VCF.",
    ),
    (
        "PA03",
        "PROCEDURAL",
        "Write a Terraform script to deploy an AWS r6i.4xlarge EC2 instance with Ubuntu 22.04 "
        "and a 1 TB gp3 EBS volume for a bioinformatics STAR alignment workload.",
    ),
    # ANALYTICAL ──────────────────────────────────────────────────────────────
    (
        "PA04",
        "ANALYTICAL",
        "Compare BWA-MEM and Bowtie2 for short-read DNA alignment. Evaluate trade-offs in "
        "speed, memory, sensitivity, and suitability for whole-genome vs. targeted sequencing.",
    ),
    (
        "PA05",
        "ANALYTICAL",
        "Evaluate STAR versus HISAT2 for splice-aware RNA-seq alignment. Compare their "
        "two-pass strategies, memory requirements, junction sensitivity, and performance "
        "on long-read data.",
    ),
    # CONCEPTUAL ──────────────────────────────────────────────────────────────
    (
        "PA06",
        "CONCEPTUAL",
        "Explain the statistical model behind DESeq2 for differential expression analysis, "
        "including the negative binomial distribution and empirical Bayes dispersion shrinkage.",
    ),
    (
        "PA07",
        "CONCEPTUAL",
        "What is the Leiden community detection algorithm and how does it guarantee "
        "well-connected communities, improving upon the Louvain method?",
    ),
    # MIXED ───────────────────────────────────────────────────────────────────
    (
        "PA08",
        "MIXED",
        "Find 2 experts who have published papers on CRISPR base editing. Explain the key "
        "biochemical differences between cytosine base editors and adenine base editors, "
        "and design an AWS architecture for storing the resulting variant call files.",
    ),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Standalone profiler ablation benchmark (adaptive mode only)"
)
_parser.add_argument("--queries", type=int, default=0,
                     help="Run only first N queries (0 = all)")
_parser.add_argument("--out-dir", default="",
                     help="Output directory (default: raw_telemetry_profiler_ablation/)")
_args, _ = _parser.parse_known_args()

ROOT_DIR   = Path(os.path.dirname(__file__)).parent
OUTPUT_DIR = Path(_args.out_dir) if _args.out_dir else ROOT_DIR / "raw_telemetry_profiler_ablation"
OUTPUT_DIR.mkdir(exist_ok=True)

ENGINE_TELEMETRY_FILE = ROOT_DIR / "experiment_telemetry.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_engine_telemetry() -> dict:
    if not ENGINE_TELEMETRY_FILE.exists():
        return {}
    try:
        data = json.loads(ENGINE_TELEMETRY_FILE.read_text())
        if isinstance(data, list) and data:
            return data[-1]
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_run(telemetry: dict, metadata: dict) -> str:
    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id":    run_id,
        "timestamp": datetime.utcnow().isoformat(),
        **metadata,
        "engine_telemetry": telemetry,
    }
    path = OUTPUT_DIR / f"run_{run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_profiler_ablation():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           PROFILER ABLATION BENCHMARK                    ║")
    print("║   adaptive mode  ·  profiler ON vs OFF  ·  DAG locked   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    queries = PROFILER_QUERIES[:_args.queries] if _args.queries > 0 else PROFILER_QUERIES
    total   = len(queries) * 2  # ON + OFF each
    done    = 0

    print(f"\nQueries   : {len(queries)}")
    print(f"Total runs: {total}  ({len(queries)} × 2 conditions)")
    print(f"Output    : {OUTPUT_DIR}\n")

    engine = OrchestratorEngine()

    # Fresh profiler state for a clean session
    user_profiler.reset_all()

    for q_idx, (qid, dim_label, query) in enumerate(queries):
        print(f"\n══ {qid} [{dim_label}] ({q_idx+1}/{len(queries)}) ══════════════════════")
        print(f"   {query[:100]}{'...' if len(query) > 100 else ''}")

        # Decompose ONCE — identical DAG for both conditions
        cached_tasks, cached_decomp_name = None, None
        try:
            print("  Decomposing (shared plan)...")
            cached_tasks, cached_decomp_name = await engine.decompose_only(query)
            print(f"  Plan locked: {len(cached_tasks)} tasks via {cached_decomp_name.upper()}")
            await asyncio.sleep(3)
        except Exception as exc:
            print(f"  Decompose failed: {exc} — falling back to per-run decomposition")

        for profiler_on in (True, False):
            done += 1
            label = "ON " if profiler_on else "OFF"
            print(f"\n  ▶ Profiler {label} | Progress: {done}/{total}")

            # Toggle profiler at runtime
            config.Config.USE_PROFILER = profiler_on

            meta = {
                "protocol":      "profiler_ablation",
                "query_id":      qid,
                "query_dim":     dim_label,
                "ablation_mode": "adaptive",
                "profiler_on":   profiler_on,
                "query":         query,
                "dag_locked":    cached_tasks is not None,
            }

            try:
                t0 = time.time()
                await engine.run(
                    query,
                    ablation_mode="adaptive",
                    start_time=t0,
                    cached_tasks=cached_tasks,
                    cached_decomposer_name=cached_decomp_name,
                )
                latency_ms = int((time.time() - t0) * 1000)

                telemetry = read_engine_telemetry()
                telemetry["observed_latency_ms"] = latency_ms
                telemetry["dag_locked"] = cached_tasks is not None

                run_id = save_run(telemetry, {**meta, "wall_latency_ms": latency_ms})
                score  = telemetry.get("final_judge_score", "?")
                print(f"  ✔ Saved run_{run_id}.json | score={score} | {latency_ms}ms")

                await asyncio.sleep(2)

            except Exception as exc:
                print(f"  ✘ Error: {exc}")
                save_run({}, {**meta, "error": str(exc), "failed": True})
                await asyncio.sleep(10)

    # Restore config to default
    config.Config.USE_PROFILER = True

    print(f"\n✅ Done. {done} runs saved to {OUTPUT_DIR.resolve()}")
    print(f"\nAnalyze with:")
    print(f"  python scripts-dev/05_profiler_ablation.py --dir {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    if ENGINE_TELEMETRY_FILE.exists():
        os.remove(ENGINE_TELEMETRY_FILE)
    asyncio.run(run_profiler_ablation())
