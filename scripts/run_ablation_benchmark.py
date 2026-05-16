"""
run_ablation_benchmark.py
=========================
Executes the Causal Ablation Study for the Agent-to-Agent Communication paper.
Runs the benchmark queries in exactly three modes:
  1. "none"     - Absolute blind passing. Only questions sent.
  2. "raw"      - Complete context dumping.
  3. "adaptive" - Dynamic Gateway (RAW if < budget, CUGCR compression if > budget).

Saves the raw JSON telemetry for each run into the `raw_telemetry/` folder.
"""

import argparse
import asyncio
import json
import sys
import os
import time
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv()
from orchestrator.engine import OrchestratorEngine

# ── CLI args — parsed early so OUTPUT_DIR can be set before module-level use ─
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--modes",   default="none,raw,adaptive",
                     help="Comma-separated modes to run (default: none,raw,adaptive)")
_parser.add_argument("--queries", type=int, default=0,
                     help="Run only the first N queries (0 = all)")
_parser.add_argument("--out-dir", default="",
                     help="Output directory for telemetry (default: raw_telemetry/)")
_args, _ = _parser.parse_known_args()

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
OUTPUT_DIR = Path(_args.out_dir) if _args.out_dir else ROOT_DIR / "raw_telemetry"
OUTPUT_DIR.mkdir(exist_ok=True)
ENGINE_TELEMETRY_FILE = ROOT_DIR / "experiment_telemetry.json"


BENCHMARK_QUERIES = [
    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 1: The "Perfect Hit" Full-Stack Pipeline (Tests CUS & Pareto Frontier)
    # Goal: RAG -> Code -> Cloud. Tests if massive context compresses cleanly.
    # KB coverage: Leiden, STAR, DESeq2, SAM/SAMtools, Seurat WNN
    # ══════════════════════════════════════════════════════════════════════════════
    "Explain the Leiden community detection algorithm and how it guarantees well-connected communities, improving upon Louvain. Write a Python script using Scanpy to cluster single-cell RNA-seq cells with Leiden. Design an AWS Terraform architecture using memory-optimized instances for clustering 1 million cells.",
    "Describe the splice-aware alignment algorithm used by STAR for RNA-seq, including its suffix array index and two-pass strategy. Write a shell script to run STAR alignment on paired-end FASTQ files. Design an AWS Batch architecture for parallelizing STAR across 100 samples using EC2 Spot Instances.",
    "Explain the statistical model behind DESeq2 for differential expression analysis, including the negative binomial distribution and empirical Bayes dispersion shrinkage. Write a Python script using pydeseq2 on a count matrix. Deploy this pipeline on an AWS EC2 r6i instance.",
    "Explain how the SAM and BAM file formats store sequencing alignment data, including CIGAR strings, FLAG bit encoding, and optional tags. Write a Python script using pysam to parse a BAM file and compute per-read alignment statistics. Design an AWS S3 tiered-storage architecture for managing BAM files from a large cohort.",
    "Describe the Weighted Nearest Neighbor (WNN) multimodal integration method from the Seurat paper for joint RNA and protein analysis. Write a Python script using Scanpy to perform WNN-based joint dimensionality reduction. Design an AWS architecture for multimodal single-cell analysis at scale.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 2: Coreference & Knowledge Transfer (Tests Protected Atom DSR)
    # Goal: Scholar -> Code/Cloud. Tests if the Atomizer protects exact human names.
    # KB coverage: Leiden, CRISPR base editing, STAR, Seurat WNN, DESeq2
    # ══════════════════════════════════════════════════════════════════════════════
    "Find 3 experts on Leiden community detection in single-cell genomics. Use their exact names to create a Pandas dataframe in a Python script, and print the dataframe.",
    "Find 2 human experts who have published papers on programmable CRISPR base editing. Explain the key biochemical differences between cytosine base editors and adenine base editors, and design an AWS cloud architecture for storing the resulting variant call files.",
    "Identify 3 top researchers working on splice-aware RNA-seq alignment algorithms. Summarize their main algorithmic contributions, and write a Python script to run STAR alignment with their recommended two-pass parameters.",
    "Search for 2 experts on multimodal single-cell data integration. Based on their research field, explain the WNN graph construction approach, and design an AWS infrastructure optimized for storing and querying paired RNA and ATAC-seq datasets.",
    "Find 1 expert on differential expression analysis using count-based RNA-seq models. What statistical framework do they recommend? Write a Python script implementing that framework on a raw count matrix.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 3: Context Overload Stress Tests (Tests The U-Curve & Compression)
    # Goal: Massive RAG theory passed to Code. Tests if CUGCR prevents LLM distraction.
    # KB coverage: Leiden vs Louvain, STAR two-pass, DESeq2 math, base editing, SAM spec
    # ══════════════════════════════════════════════════════════════════════════════
    "Compare the Leiden and Louvain community detection algorithms in deep mathematical detail, including their modularity optimization guarantees and why Leiden produces well-connected communities while Louvain does not. Write a Python script using leidenalg to cluster a graph with custom resolution parameters.",
    "Detail the STAR aligner's genome index construction using suffix arrays and its two-pass strategy for novel splice junction discovery. Explain the mathematical basis of its seed-and-extend alignment. Write a comprehensive shell script for a STAR two-pass alignment pipeline on paired-end FASTQ files.",
    "Explain the DESeq2 dispersion estimation pipeline in detail: Cox-Reid adjusted profile likelihood, maximum likelihood dispersion estimates, and empirical Bayes shrinkage toward the fitted trend. Write a Python script using pydeseq2 to reproduce this workflow and plot a dispersion estimates figure.",
    "Describe the biochemical mechanisms of both cytosine base editing (C·G to T·A via deamination) and adenine base editing (A·T to G·C via adenosine deaminase), including the role of UGI and the DNA repair pathways involved. Write a Python script to scan a FASTA genome for NGG PAM-adjacent base editing windows.",
    "Explain the full SAM file format specification, including all mandatory fields, every CIGAR string operation (M, I, D, N, S, H, P, =, X), FLAG bitmask encoding, and the MAPQ quality formula. Write a Python script using pysam to filter a BAM file keeping only properly paired primary alignments with MAPQ above 30.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 4: "Missing Data" & Refusals (Tests System Resilience & Reflexion)
    # Goal: Ask for things absent from the KB. Tests low-CRS warnings & Reflexion.
    # These topics are intentionally NOT in the offline knowledge base.
    # ══════════════════════════════════════════════════════════════════════════════
    "I need to perform single-cell trajectory inference using the Monocle3 algorithm. Write a Python script using Scanpy that explicitly runs Monocle3 pseudotime analysis on an AnnData object.",
    "Compare the pros and cons of using Salmon versus Kallisto for RNA-seq quantification. Write a complete Python script using Kallisto to perform pseudoalignment on paired-end FASTQ files.",
    "Explain how the MAGIC imputation algorithm reconstructs gene expression for single-cell data, and write a Scanpy script to run it on a preprocessed AnnData object.",
    "Find 2 experts on Quantum Computing in Genomics. Write a Qiskit Python script to run a quantum sequence alignment algorithm.",
    "Explain how to process a whole-genome sequencing dataset through GATK variant calling. Write a Snakemake workflow for BQSR and HaplotypeCaller, and design an AWS Batch deployment for this pipeline.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 5: Adversarial & Ambiguous Routing (Tests the Neuro-Symbolic Decomposer)
    # Goal: Trick the system. Tests if Phase 1 LLM accurately resolves dependencies.
    # ══════════════════════════════════════════════════════════════════════════════
    "Write a Python script that both aligns and simultaneously un-aligns RNA-seq reads to the reference genome using STAR, then deploy it to AWS Lambda.",
    "Do the thing with the single-cell data where it finds the communities using that graph algorithm from the Leiden paper, and make it work on the cloud with Terraform.",
    "Take the outputs from the top 3 researchers in RNA-seq alignment and feed their recommended STAR parameters into a pipeline that doesn't use the cloud.",
    "Write a script to perform differential expression with DESeq2, but only if DESeq2 can also predict the secondary structure of the resulting DE genes.",
    "Design an AWS architecture for hosting a multiplayer video game backend using DESeq2.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 6: Single-Agent Baselines (Tests First-Pass Success Rate - FPSR)
    # Goal: Isolated single-agent tasks. Provides clean Boxplot baseline data.
    # Includes one RAG-routing diagnostic (should hit RAG, not Scholar).
    # ══════════════════════════════════════════════════════════════════════════════
    "What negative binomial distribution parameters does DESeq2 use to model RNA-seq read counts, and how does it estimate dispersion? --strategy=cot",
    "Find 4 experts who have published papers on 'STAR RNA-seq splice junction alignment'.",
    "Write a Terraform script to deploy an AWS r6i.4xlarge instance for STAR alignment, including Ubuntu 22.04 and a 1TB gp3 EBS volume.",
    "Write a simple Python script using pysam to open a BAM file, iterate over reads, and print the name and mapping quality of the first 10 alignments.",
    "retrieve the exact two-pass alignment parameters from the STAR paper and write a Python script using those specific parameter values.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 7: Context Volume Stress Tests (Tests CUGCR at High Token Load)
    # Goal: Force large upstream contexts (>1500 tokens) through the pipeline.
    # These are designed so that RAW over-floods the downstream agent while
    # ADAPTIVE's compression preserves only the signal.
    # ══════════════════════════════════════════════════════════════════════════════
    "Retrieve ALL algorithmic details from papers on the STAR aligner including suffix array construction, two-pass alignment strategy, seed-and-extend scoring, and CIGAR string encoding. Then write a comprehensive Python library with minimum 12 functions (full docstrings, type hints, error handling) covering all major STAR parameters, input validation, and output parsing. Finally, design a complete AWS Batch + S3 + EC2 Spot Instance architecture for processing 500 paired-end samples in parallel, including IAM roles, cost optimization, and CloudWatch monitoring.",
    "Retrieve comprehensive documentation on BOTH the Leiden community detection algorithm AND the Seurat Weighted Nearest Neighbor (WNN) multimodal integration method. Synthesize the complete mathematical theory behind both methods including modularity optimization, well-connectedness guarantees, and weighted graph construction. Then write a unified Python pipeline that preprocesses scRNA-seq data, applies WNN-based joint embedding, clusters with Leiden at multiple resolutions, performs differential expression between clusters, and visualizes results in UMAP space. Deploy this complete pipeline on an AWS p3.2xlarge GPU instance with automated dependency installation.",
    "Retrieve the full SAM/BAM format specification including ALL mandatory fields, every CIGAR string operation (M, I, D, N, S, H, P, =, X), the complete FLAG bitmask encoding for all 12 bits, the MAPQ quality formula, and optional tag specifications. Then write a complete Python library implementing: a SAM/BAM parser, CIGAR string interpreter for all 9 operations, FLAG decoder with named bit fields, per-read quality statistics calculator, proper-pair filter, and duplicate read detector using coordinate sorting. Deploy this library as an AWS Lambda function behind API Gateway that accepts BAM file S3 URIs and returns JSON alignment statistics.",
    "Find 4 researchers who have published on DESeq2 statistical modelling and dispersion estimation. For each researcher retrieve their key publications and extract every statistical formula described including: negative binomial GLM parameterization, Cox-Reid adjusted profile likelihood, maximum likelihood dispersion estimates, empirical Bayes shrinkage toward the fitted trend, and independent filtering thresholds. Write a Python implementation that reproduces the complete DESeq2 dispersion estimation workflow using pydeseq2, including visualization of the dispersion estimates plot. Then deploy this reproducible analysis on AWS EC2 r6i.2xlarge with automated R and Python environment setup via user-data scripts.",
    "Retrieve all available papers on CRISPR base editing covering both cytosine base editors (CBE) and adenine base editors (ABE), including: the biochemical deamination mechanisms, UGI roles, DNA repair pathway involvement, PAM sequence requirements, editing window positions, and known off-target profiles. Write a comprehensive Python toolkit with: a PAM site scanner supporting NGG, NG, and NGA, an editing window calculator for both CBE and ABE, a guide RNA scorer, an off-target predictor using seed region hashing, and a variant effect annotator that maps edits to coding consequences. Design an AWS Batch architecture for genome-scale analysis of 100,000 potential edit sites with results stored in S3 Parquet format and queryable via Athena.",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def read_engine_telemetry() -> dict:
    """Reads the telemetry file generated by the engine after a run."""
    if not ENGINE_TELEMETRY_FILE.exists():
        return {}
    try:
        with open(ENGINE_TELEMETRY_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[-1]
            elif isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_run(telemetry: dict, metadata: dict) -> str:
    """Saves the run data cleanly into the raw_telemetry directory."""
    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        **metadata,
        "engine_telemetry": telemetry,
    }
    out_path = OUTPUT_DIR / f"run_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return run_id

# ── Main Execution Loop ──────────────────────────────────────────────────────
async def run_ablation():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          STARTING CAUSAL ABLATION BENCHMARK              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    engine = OrchestratorEngine()

    modes = [m.strip() for m in _args.modes.split(",") if m.strip()]
    queries = BENCHMARK_QUERIES[:_args.queries] if _args.queries > 0 else BENCHMARK_QUERIES
    total_runs = len(queries) * len(modes)
    current_run = 0

    print(f"Modes     : {modes}")
    print(f"Queries   : {len(queries)}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Total runs: {total_runs}  ({len(queries)} queries × {len(modes)} modes)")
    print("DAG plan is decomposed ONCE per query and reused across all modes.\n")

    for q_idx, query in enumerate(queries):
        print(f"\n══ Query {q_idx+1}/{len(queries)} ══════════════════════════════════")
        print(f"   {query[:100]}{'...' if len(query) > 100 else ''}")

        # ── Decompose once — shared across all modes ──────────────────────────
        # This ensures NONE / RAW / ADAPTIVE all execute the identical task graph.
        # Without this, each mode call runs the LLM decomposer independently and
        # produces a different DAG, making the ablation non-comparable.
        cached_tasks, cached_decomp_name = None, None
        try:
            print(f"  🔧 Decomposing (shared plan)...")
            cached_tasks, cached_decomp_name = await engine.decompose_only(query)
            print(f"  🔧 Plan locked: {len(cached_tasks)} tasks via {cached_decomp_name.upper()}")
            # Brief pause after the decomposer LLM call before the mode runs begin
            await asyncio.sleep(5)
        except Exception as exc:
            print(f"  ✘ Decompose failed: {exc} — falling back to per-mode decomposition")

        for mode in modes:
            current_run += 1
            print(f"\n  ▶ Mode: {mode.upper()} | Progress: {current_run}/{total_runs}")

            meta = {
                "protocol":      "A_ablation",
                "query_id":      f"BQ{q_idx+1:02d}",
                "ablation_mode": mode,
                "query":         query,
                "dag_locked":    cached_tasks is not None,
            }

            try:
                start_time = time.time()
                await engine.run(
                    query,
                    ablation_mode=mode,
                    start_time=start_time,
                    cached_tasks=cached_tasks,
                    cached_decomposer_name=cached_decomp_name,
                )
                latency_ms = int((time.time() - start_time) * 1000)

                telemetry = read_engine_telemetry()
                telemetry["observed_latency_ms"] = latency_ms
                telemetry["dag_locked"] = cached_tasks is not None

                run_id = save_run(telemetry, {**meta, "wall_latency_ms": latency_ms})
                print(f"  ✔ Saved: run_{run_id}.json ({latency_ms}ms)")

                print("  ⏳ Resting 1s...")
                await asyncio.sleep(1)

            except Exception as exc:
                print(f"  ✘ Error during execution: {exc}")
                save_run({}, {**meta, "error": str(exc), "failed": True})
                print("  ⏳ API Error. Resting 60s...")
                await asyncio.sleep(2)

    print(f"\n✅ Ablation Benchmark Complete. All data saved to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    # Clean the slate before running
    if ENGINE_TELEMETRY_FILE.exists():
        os.remove(ENGINE_TELEMETRY_FILE)

    # Validate modes
    valid = {"none", "raw", "adaptive"}
    bad = [m for m in _args.modes.split(",") if m.strip() not in valid]
    if bad:
        print(f"Unknown mode(s): {bad}. Choose from: none, raw, adaptive")
        sys.exit(1)

    asyncio.run(run_ablation())