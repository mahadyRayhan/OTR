"""
run_full_evaluation.py
======================
Phase 1 of the evaluation framework.
Runs ALL evaluation protocols and saves structured raw telemetry for every run.

Output: raw_telemetry/<run_id>.json  (one file per engine execution)
        raw_telemetry/_manifest.json (index of all runs with metadata)

Protocols executed:
  A. Three-mode ablation     — NONE / RAW / ADAPTIVE on all benchmark queries
  B. Consistency runs        — same query x3 in adaptive mode (stability test)
  C. Adversarial queries     — deliberately malformed / ambiguous queries
  D. Router determinism      — same query x5, structural DAG comparison
  E. Profile sensitivity     — inject different Bayesian profiles, same query
  F. Boundary condition      — queries tuned to produce near-threshold context sizes
"""

import asyncio
import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


from orchestrator.engine import OrchestratorEngine

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = Path("raw_telemetry")
OUTPUT_DIR.mkdir(exist_ok=True)

# Source telemetry file your engine already writes
ENGINE_TELEMETRY_FILE = Path("experiment_telemetry.json")

# ── Rate-limit helpers ───────────────────────────────────────────────────────
INTER_RUN_SLEEP   = 1   # seconds between normal runs
ERROR_SLEEP       = 3   # seconds after API error
LONG_SLEEP        = 2   # seconds between protocol blocks

# ── Query banks ─────────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    # Group 1 — Perfect hit (RAG → Code → Cloud)
    "What is the standard statistical method used by Scanpy to find marker genes across cell "
    "clusters? Explain the method, then write a Python script that applies this test to find "
    "differentially expressed genes across Leiden clusters and plots the top 25 genes as a dotplot.",

    "Explain the theoretical difference between PCA and UMAP in single-cell analysis. Based on "
    "this, write a Python script using Scanpy to load a dataset, perform PCA, construct a "
    "neighborhood graph, and compute the UMAP embeddings.",

    "Write a complete Python script using Scanpy that loads an H5AD file and runs the Leiden "
    "community detection algorithm with a resolution of 0.5. Once the script is written, design "
    "an AWS EC2 and EFS architecture tailored specifically for the memory bottlenecks of "
    "single-cell clustering.",

    # Group 2 — Extrapolation / AI warning stress tests
    "I want to run a protein structure prediction pipeline using AlphaFold 3. Write an AWS "
    "Terraform deployment architecture for this pipeline, making sure you recommend the exact "
    "GPU hardware and OS frameworks required.",

    "Find 1 expert on analyzing mass spectrometry proteomics data. What is the standard file "
    "format for raw mass spectrometry data? Write a Python script using the 'pyteomics' library "
    "to parse that format. Finally, design an AWS deployment architecture for the MaxQuant tool.",

    "Write a complete Nextflow pipeline script for trimming Illumina FASTQ reads using FastQC "
    "and Trimmomatic. Once the script is written, design an AWS Batch deployment architecture, "
    "assuming the use of EC2 Spot Instances for cost savings.",

    # Group 3 — Deep coreference / synthesis
    "Identify the standard file format for storing unaligned raw sequencing reads. What Python "
    "package is commonly used to parse it? Write a Python script using that package to count the "
    "number of reads, ensuring it handles both uncompressed and gzipped files.",

    "Find 2 human experts who have published recent papers on 'RNA-seq Splice-Aware Alignment'. "
    "Then, explain the primary algorithm those experts use. Finally, write a bash script that "
    "downloads the tool associated with that algorithm.",

    # Group 4 — Missing algorithm / hard refusal
    "I need to perform a highly specialized batch-effect correction using the 'Harmony' algorithm. "
    "Write a Python script using Scanpy that explicitly runs the Harmony integration tool on an "
    "AnnData object.",

    "Compare the pros and cons of using AlphaFold 3 versus Rosetta for protein structure "
    "prediction. Then, write a complete Python script that utilizes the Rosetta PyRosetta "
    "library to perform de novo folding of a short amino acid sequence.",
]

# Adversarial queries (Protocol C) — tests robustness / graceful degradation
ADVERSARIAL_QUERIES = [
    # Contradictory intent
    "Write a Python script that both normalizes AND denormalizes single-cell RNA data simultaneously.",
    # Completely out of domain
    "Design an AWS architecture for hosting a multiplayer video game backend using Scanpy.",
    # Underspecified (no clear task)
    "Do the thing with the data and then make it work on the cloud.",
    # Ambiguous pronoun chain stress test
    "Take it and run it on the results from before, then compare it with what we had previously.",
    # Hallucination trap (tool that does not exist)
    "Write a Python script using the 'BioPyML' library to perform automated genome annotation.",
]

# Boundary condition queries — produce context that sits near the CUGCR token threshold
BOUNDARY_QUERIES = [
    "Briefly explain what UMAP does in one sentence, then write a 10-line Python snippet.",
    "What is a FASTQ file? Write a 3-line Python script to open one.",
]

# Profile sensitivity test — same query, different injected profiles
PROFILE_SENSITIVITY_QUERY = (
    "Explain differential expression analysis in single-cell RNA-seq and write a complete "
    "Python script using Scanpy to perform it."
)
PROFILE_VARIANTS = [
    {"conceptual": 0.8, "procedural": 0.1, "analytical": 0.1},   # novice / conceptual
    {"conceptual": 0.1, "procedural": 0.8, "analytical": 0.1},   # practitioner / procedural
    {"conceptual": 0.1, "procedural": 0.1, "analytical": 0.8},   # researcher / analytical
]

# Router determinism — run same query N times, compare DAG topology
DETERMINISM_QUERY = BENCHMARK_QUERIES[0]
DETERMINISM_RUNS  = 5

# Consistency runs — run same N queries 3x each
CONSISTENCY_QUERIES = BENCHMARK_QUERIES[:3]
CONSISTENCY_REPS    = 3

# ── Manifest ─────────────────────────────────────────────────────────────────
manifest: list[dict] = []


# ── Helpers ──────────────────────────────────────────────────────────────────

# def read_engine_telemetry() -> dict:
#     """Read the telemetry file the engine writes after each run."""
#     if not ENGINE_TELEMETRY_FILE.exists():
#         return {}
#     try:
#         with open(ENGINE_TELEMETRY_FILE, "r") as f:
#             return json.load(f)
#     except (json.JSONDecodeError, OSError):
#         return {}

def read_engine_telemetry() -> dict:
    """Read the latest run from the telemetry file the engine writes."""
    if not ENGINE_TELEMETRY_FILE.exists():
        return {}
    try:
        with open(ENGINE_TELEMETRY_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[-1]  # Grab the most recent run
            elif isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_run(telemetry: dict, metadata: dict) -> str:
    """Merge engine telemetry with evaluation metadata and save to disk."""
    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id":    run_id,
        "timestamp": datetime.utcnow().isoformat(),
        **metadata,
        "engine_telemetry": telemetry,
    }
    out_path = OUTPUT_DIR / f"run_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    manifest.append({
        "run_id":   run_id,
        "file":     str(out_path),
        "protocol": metadata.get("protocol"),
        "mode":     metadata.get("ablation_mode"),
        "query_id": metadata.get("query_id"),
    })
    return run_id


def flush_manifest():
    with open(OUTPUT_DIR / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


async def execute_run(
    engine: OrchestratorEngine,
    query: str,
    ablation_mode: str,
    metadata: dict,
    sleep_after: int = INTER_RUN_SLEEP,
) -> bool:
    """Execute one engine run, capture telemetry, save to disk."""
    print(f"  ▶  mode={ablation_mode} | {query[:80]}...")
    try:
        start = time.time()
        await engine.run(query, ablation_mode=ablation_mode, start_time=start)
        latency_ms = int((time.time() - start) * 1000)

        telemetry = read_engine_telemetry()
        telemetry["observed_latency_ms"] = latency_ms

        run_id = save_run(telemetry, {**metadata, "ablation_mode": ablation_mode,
                                      "query": query, "wall_latency_ms": latency_ms})
        print(f"  ✔  saved run_{run_id}.json  ({latency_ms}ms)")
        await asyncio.sleep(sleep_after)
        return True

    except Exception as exc:
        print(f"  ✘  error: {exc}")
        save_run({}, {**metadata, "ablation_mode": ablation_mode,
                      "query": query, "error": str(exc), "failed": True})
        await asyncio.sleep(ERROR_SLEEP)
        return False


# ── Protocol A: Three-mode ablation ─────────────────────────────────────────

async def protocol_a(engine: OrchestratorEngine):
    """
    Run every benchmark query under NONE / RAW / ADAPTIVE.
    Enables: ΔQ, win rate, CRS/CUS/DSR comparison, token compression ratio,
             CE (communication efficiency), Pareto frontier data.
    """
    print("\n" + "="*60)
    print("PROTOCOL A — Three-mode ablation (NONE / RAW / ADAPTIVE)")
    print("="*60)
    modes = ["none", "raw", "adaptive"]

    for q_idx, query in enumerate(BENCHMARK_QUERIES):
        print(f"\n── Query {q_idx+1}/{len(BENCHMARK_QUERIES)}")
        for mode in modes:
            meta = {
                "protocol": "A_ablation",
                "query_id": f"BQ{q_idx+1:02d}",
                "query_group": _query_group(q_idx),
            }
            await execute_run(engine, query, mode, meta)

    await asyncio.sleep(LONG_SLEEP)


# ── Protocol B: Consistency / stability ──────────────────────────────────────

async def protocol_b(engine: OrchestratorEngine):
    """
    Run CONSISTENCY_QUERIES × CONSISTENCY_REPS times in adaptive mode.
    Enables: GTS variance (stability), answer consistency score,
             stochastic instability detection.
    """
    print("\n" + "="*60)
    print("PROTOCOL B — Consistency runs (3× per query)")
    print("="*60)

    for q_idx, query in enumerate(CONSISTENCY_QUERIES):
        print(f"\n── Query {q_idx+1}/{len(CONSISTENCY_QUERIES)}")
        for rep in range(CONSISTENCY_REPS):
            print(f"   Rep {rep+1}/{CONSISTENCY_REPS}")
            meta = {
                "protocol": "B_consistency",
                "query_id": f"BQ{q_idx+1:02d}",
                "rep":      rep + 1,
            }
            await execute_run(engine, query, "adaptive", meta)

    await asyncio.sleep(LONG_SLEEP)


# ── Protocol C: Adversarial robustness ───────────────────────────────────────

async def protocol_c(engine: OrchestratorEngine):
    """
    Run adversarial queries in adaptive mode.
    Enables: failure floor measurement, graceful degradation score,
             cascading failure detection, placeholder leak stress test.
    """
    print("\n" + "="*60)
    print("PROTOCOL C — Adversarial robustness")
    print("="*60)

    for q_idx, query in enumerate(ADVERSARIAL_QUERIES):
        print(f"\n── Adversarial {q_idx+1}/{len(ADVERSARIAL_QUERIES)}")
        meta = {
            "protocol": "C_adversarial",
            "query_id": f"AQ{q_idx+1:02d}",
        }
        await execute_run(engine, query, "adaptive", meta)

    await asyncio.sleep(LONG_SLEEP)


# ── Protocol D: Router determinism ───────────────────────────────────────────

async def protocol_d(engine: OrchestratorEngine):
    """
    Run the same query DETERMINISM_RUNS times in adaptive mode.
    Enables: symbolic router determinism score (DAG topology variance),
             validates 'no routing hallucination' claim.
    """
    print("\n" + "="*60)
    print(f"PROTOCOL D — Router determinism ({DETERMINISM_RUNS}× same query)")
    print("="*60)

    for rep in range(DETERMINISM_RUNS):
        print(f"\n── Rep {rep+1}/{DETERMINISM_RUNS}")
        meta = {
            "protocol": "D_determinism",
            "query_id": "BQ01",
            "rep":      rep + 1,
        }
        await execute_run(engine, DETERMINISM_QUERY, "adaptive", meta)

    await asyncio.sleep(LONG_SLEEP)


# ── Protocol E: Profile sensitivity ─────────────────────────────────────────

async def protocol_e(engine: OrchestratorEngine):
    """
    Inject different Bayesian user profiles for the same query.
    Enables: user profiler behavioral impact measurement, depth budget
             utilization per profile, output style delta.
    Requires: OrchestratorEngine to accept a user_profile kwarg.
              If not supported, runs are tagged for manual analysis.
    """
    print("\n" + "="*60)
    print("PROTOCOL E — User profile sensitivity")
    print("="*60)

    for p_idx, profile in enumerate(PROFILE_VARIANTS):
        profile_name = ["conceptual", "procedural", "analytical"][p_idx]
        print(f"\n── Profile: {profile_name}")
        meta = {
            "protocol":    "E_profile_sensitivity",
            "query_id":    "PS01",
            "profile_name": profile_name,
            "injected_profile": profile,
        }
        try:
            # Try passing profile if engine supports it
            start = time.time()
            await engine.run(
                PROFILE_SENSITIVITY_QUERY,
                ablation_mode="adaptive",
                start_time=start,
                user_profile=profile,           # may be ignored if unsupported
            )
        except TypeError:
            # Engine does not accept user_profile kwarg — run without it
            start = time.time()
            await engine.run(
                PROFILE_SENSITIVITY_QUERY,
                ablation_mode="adaptive",
                start_time=start,
            )

        latency_ms = int((time.time() - start) * 1000)
        telemetry  = read_engine_telemetry()
        telemetry["observed_latency_ms"] = latency_ms
        run_id = save_run(telemetry, {**meta, "query": PROFILE_SENSITIVITY_QUERY,
                                      "ablation_mode": "adaptive",
                                      "wall_latency_ms": latency_ms})
        print(f"  ✔  saved run_{run_id}.json")
        await asyncio.sleep(INTER_RUN_SLEEP)

    await asyncio.sleep(LONG_SLEEP)


# ── Protocol F: Boundary conditions ─────────────────────────────────────────

async def protocol_f(engine: OrchestratorEngine):
    """
    Run queries that produce context near the CUGCR token threshold.
    Enables: gateway decision boundary analysis, context boundary behavior,
             RAW vs CUGCR switching edge-case detection.
    """
    print("\n" + "="*60)
    print("PROTOCOL F — Context boundary conditions")
    print("="*60)

    for q_idx, query in enumerate(BOUNDARY_QUERIES):
        print(f"\n── Boundary query {q_idx+1}/{len(BOUNDARY_QUERIES)}")
        for mode in ["raw", "adaptive"]:
            meta = {
                "protocol": "F_boundary",
                "query_id": f"BND{q_idx+1:02d}",
            }
            await execute_run(engine, query, mode, meta)

    await asyncio.sleep(LONG_SLEEP)


# ── Utility ──────────────────────────────────────────────────────────────────

def _query_group(idx: int) -> str:
    if idx < 3:  return "perfect_hit"
    if idx < 6:  return "extrapolation"
    if idx < 8:  return "coreference"
    return "missing_algorithm"


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         FULL SYSTEM EVALUATION — DATA COLLECTION        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Output directory : {OUTPUT_DIR.resolve()}")
    print(f"Started          : {datetime.utcnow().isoformat()}")

    engine = OrchestratorEngine()

    await protocol_a(engine)   # ~10 queries × 3 modes = 30 runs
    await protocol_b(engine)   # ~3 queries × 3 reps   =  9 runs
    await protocol_c(engine)   # 5 adversarial          =  5 runs
    await protocol_d(engine)   # 1 query × 5 reps       =  5 runs
    await protocol_e(engine)   # 1 query × 3 profiles   =  3 runs
    await protocol_f(engine)   # 2 queries × 2 modes    =  4 runs
    # Total: ~56 runs

    flush_manifest()
    print(f"\n✅ Data collection complete. {len(manifest)} runs saved.")
    print(f"   Manifest: {OUTPUT_DIR / '_manifest.json'}")


if __name__ == "__main__":
    asyncio.run(main())