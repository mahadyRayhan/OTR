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

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
OUTPUT_DIR = ROOT_DIR / "raw_telemetry"
OUTPUT_DIR.mkdir(exist_ok=True)
ENGINE_TELEMETRY_FILE = ROOT_DIR / "experiment_telemetry.json"


BENCHMARK_QUERIES = [
    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 1: The "Perfect Hit" Full-Stack Pipeline (Tests CUS & Pareto Frontier)
    # Goal: RAG -> Code -> Cloud. Tests if massive context compresses cleanly.
    # KB coverage: Leiden, STAR, DESeq2, SAM/SAMtools, Seurat WNN
    # ══════════════════════════════════════════════════════════════════════════════
    "Explain the Leiden community detection algorithm and how it guarantees well-connected communities, improving upon Louvain. Write a Python script using Scanpy to cluster single-cell RNA-seq cells with Leiden. Design an AWS Terraform architecture using memory-optimized instances for clustering 1 million cells.",
#     "Describe the splice-aware alignment algorithm used by STAR for RNA-seq, including its suffix array index and two-pass strategy. Write a shell script to run STAR alignment on paired-end FASTQ files. Design an AWS Batch architecture for parallelizing STAR across 100 samples using EC2 Spot Instances.",
#     "Explain the statistical model behind DESeq2 for differential expression analysis, including the negative binomial distribution and empirical Bayes dispersion shrinkage. Write a Python script using pydeseq2 on a count matrix. Deploy this pipeline on an AWS EC2 r6i instance.",
#     "Explain how the SAM and BAM file formats store sequencing alignment data, including CIGAR strings, FLAG bit encoding, and optional tags. Write a Python script using pysam to parse a BAM file and compute per-read alignment statistics. Design an AWS S3 tiered-storage architecture for managing BAM files from a large cohort.",
#     "Describe the Weighted Nearest Neighbor (WNN) multimodal integration method from the Seurat paper for joint RNA and protein analysis. Write a Python script using Scanpy to perform WNN-based joint dimensionality reduction. Design an AWS architecture for multimodal single-cell analysis at scale.",

#     # ══════════════════════════════════════════════════════════════════════════════
#     # GROUP 2: Coreference & Knowledge Transfer (Tests Protected Atom DSR)
#     # Goal: Scholar -> Code/Cloud. Tests if the Atomizer protects exact human names.
#     # KB coverage: Leiden, CRISPR base editing, STAR, Seurat WNN, DESeq2
#     # ══════════════════════════════════════════════════════════════════════════════
#     "Find 3 experts on Leiden community detection in single-cell genomics. Use their exact names to create a Pandas dataframe in a Python script, and print the dataframe.",
#     "Find 2 human experts who have published papers on programmable CRISPR base editing. Explain the key biochemical differences between cytosine base editors and adenine base editors, and design an AWS cloud architecture for storing the resulting variant call files.",
#     "Identify 3 top researchers working on splice-aware RNA-seq alignment algorithms. Summarize their main algorithmic contributions, and write a Python script to run STAR alignment with their recommended two-pass parameters.",
#     "Search for 2 experts on multimodal single-cell data integration. Based on their research field, explain the WNN graph construction approach, and design an AWS infrastructure optimized for storing and querying paired RNA and ATAC-seq datasets.",
#     "Find 1 expert on differential expression analysis using count-based RNA-seq models. What statistical framework do they recommend? Write a Python script implementing that framework on a raw count matrix.",

#     # ══════════════════════════════════════════════════════════════════════════════
#     # GROUP 3: Context Overload Stress Tests (Tests The U-Curve & Compression)
#     # Goal: Massive RAG theory passed to Code. Tests if CUGCR prevents LLM distraction.
#     # KB coverage: Leiden vs Louvain, STAR two-pass, DESeq2 math, base editing, SAM spec
#     # ══════════════════════════════════════════════════════════════════════════════
#     "Compare the Leiden and Louvain community detection algorithms in deep mathematical detail, including their modularity optimization guarantees and why Leiden produces well-connected communities while Louvain does not. Write a Python script using leidenalg to cluster a graph with custom resolution parameters.",
#     "Detail the STAR aligner's genome index construction using suffix arrays and its two-pass strategy for novel splice junction discovery. Explain the mathematical basis of its seed-and-extend alignment. Write a comprehensive shell script for a STAR two-pass alignment pipeline on paired-end FASTQ files.",
#     "Explain the DESeq2 dispersion estimation pipeline in detail: Cox-Reid adjusted profile likelihood, maximum likelihood dispersion estimates, and empirical Bayes shrinkage toward the fitted trend. Write a Python script using pydeseq2 to reproduce this workflow and plot a dispersion estimates figure.",
#     "Describe the biochemical mechanisms of both cytosine base editing (C·G to T·A via deamination) and adenine base editing (A·T to G·C via adenosine deaminase), including the role of UGI and the DNA repair pathways involved. Write a Python script to scan a FASTA genome for NGG PAM-adjacent base editing windows.",
#     "Explain the full SAM file format specification, including all mandatory fields, every CIGAR string operation (M, I, D, N, S, H, P, =, X), FLAG bitmask encoding, and the MAPQ quality formula. Write a Python script using pysam to filter a BAM file keeping only properly paired primary alignments with MAPQ above 30.",

#     # ══════════════════════════════════════════════════════════════════════════════
#     # GROUP 4: "Missing Data" & Refusals (Tests System Resilience & Reflexion)
#     # Goal: Ask for things absent from the KB. Tests ⚠️ low-CRS warnings & Reflexion.
#     # These topics are intentionally NOT in the offline knowledge base.
#     # ══════════════════════════════════════════════════════════════════════════════
#     "I need to perform single-cell trajectory inference using the Monocle3 algorithm. Write a Python script using Scanpy that explicitly runs Monocle3 pseudotime analysis on an AnnData object.",
#     "Compare the pros and cons of using Salmon versus Kallisto for RNA-seq quantification. Write a complete Python script using Kallisto to perform pseudoalignment on paired-end FASTQ files.",
#     "Explain how the MAGIC imputation algorithm reconstructs gene expression for single-cell data, and write a Scanpy script to run it on a preprocessed AnnData object.",
#     "Find 2 experts on Quantum Computing in Genomics. Write a Qiskit Python script to run a quantum sequence alignment algorithm.",
#     "Explain how to process a whole-genome sequencing dataset through GATK variant calling. Write a Snakemake workflow for BQSR and HaplotypeCaller, and design an AWS Batch deployment for this pipeline.",

#     # ══════════════════════════════════════════════════════════════════════════════
#     # GROUP 5: Adversarial & Ambiguous Routing (Tests the Neuro-Symbolic Decomposer)
#     # Goal: Trick the system. Tests if Phase 1 LLM accurately resolves dependencies.
#     # ══════════════════════════════════════════════════════════════════════════════
#     "Write a Python script that both aligns and simultaneously un-aligns RNA-seq reads to the reference genome using STAR, then deploy it to AWS Lambda.",
#     "Do the thing with the single-cell data where it finds the communities using that graph algorithm from the Leiden paper, and make it work on the cloud with Terraform.",
#     "Take the outputs from the top 3 researchers in RNA-seq alignment and feed their recommended STAR parameters into a pipeline that doesn't use the cloud.",
#     "Write a script to perform differential expression with DESeq2, but only if DESeq2 can also predict the secondary structure of the resulting DE genes.",
#     "Design an AWS architecture for hosting a multiplayer video game backend using DESeq2.",

#     # ══════════════════════════════════════════════════════════════════════════════
#     # GROUP 6: Single-Agent Baselines (Tests First-Pass Success Rate - FPSR)
#     # Goal: Isolated single-agent tasks. Provides clean Boxplot baseline data.
#     # Includes one RAG-routing diagnostic (should hit RAG, not Scholar).
#     # ══════════════════════════════════════════════════════════════════════════════
#     "What negative binomial distribution parameters does DESeq2 use to model RNA-seq read counts, and how does it estimate dispersion? --strategy=cot",
#     "Find 4 experts who have published papers on 'STAR RNA-seq splice junction alignment'.",
#     "Write a Terraform script to deploy an AWS r6i.4xlarge instance for STAR alignment, including Ubuntu 22.04 and a 1TB gp3 EBS volume.",
#     "Write a simple Python script using pysam to open a BAM file, iterate over reads, and print the name and mapping quality of the first 10 alignments.",
#     "retrieve the exact two-pass alignment parameters from the STAR paper and write a Python script using those specific parameter values.",
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
    
    # 🚀 Strictly adhering to the 3 conditions defined in the prompt.
    modes = ["none", "raw", "adaptive"]
    total_runs = len(BENCHMARK_QUERIES) * len(modes)
    current_run = 0

    print(f"Total planned executions: {total_runs}\n")

    for q_idx, query in enumerate(BENCHMARK_QUERIES):
        print(f"── Query {q_idx+1}/{len(BENCHMARK_QUERIES)}")
        
        for mode in modes:
            current_run += 1
            print(f"  ▶ Mode: {mode.upper()} | Progress: {current_run}/{total_runs}")
            
            meta = {
                "protocol": "A_ablation",
                "query_id": f"BQ{q_idx+1:02d}",
                "ablation_mode": mode,
                "query": query
            }
            
            try:
                start_time = time.time()
                # Run the engine, explicitly passing the mode down
                await engine.run(query, ablation_mode=mode, start_time=start_time)
                latency_ms = int((time.time() - start_time) * 1000)

                # Capture and save telemetry
                telemetry = read_engine_telemetry()
                telemetry["observed_latency_ms"] = latency_ms
                
                run_id = save_run(telemetry, {**meta, "wall_latency_ms": latency_ms})
                print(f"  ✔ Saved: run_{run_id}.json ({latency_ms}ms)")
                
                # Sleep to respect API rate limits
                print("  ⏳ Resting 30s...")
                await asyncio.sleep(30)
                
            except Exception as exc:
                print(f"  ✘ Error during execution: {exc}")
                save_run({}, {**meta, "error": str(exc), "failed": True})
                print("  ⏳ API Error. Resting 60s...")
                await asyncio.sleep(60)

    print(f"\n✅ Ablation Benchmark Complete. All data saved to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    # Clean the slate before running
    if ENGINE_TELEMETRY_FILE.exists():
        os.remove(ENGINE_TELEMETRY_FILE)
        
    asyncio.run(run_ablation())