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

# ── Benchmark Queries ────────────────────────────────────────────────────────
# BENCHMARK_QUERIES = [
#     # --- Group 1: The "Perfect Hit" Pipelining (RAG -> Code -> Cloud) ---
#     "What is the standard statistical method used by Scanpy to find marker genes across cell clusters? Explain the method, then write a Python script that applies this test to find differentially expressed genes across Leiden clusters and plots the top 25 genes as a dotplot.",
#     # "Explain the theoretical difference between PCA and UMAP in single-cell analysis. Based on this, write a Python script using Scanpy to load a dataset, perform PCA, construct a neighborhood graph, and compute the UMAP embeddings.",
#     # "Write a complete Python script using Scanpy that loads an H5AD file and runs the Leiden community detection algorithm with a resolution of 0.5. Once the script is written, design an AWS EC2 and EFS architecture tailored specifically for the memory bottlenecks of single-cell clustering.",

#     # --- Group 2: Extrapolation Stress Tests (Forcing the AI Warning Labels) ---
#     "I want to run a protein structure prediction pipeline using AlphaFold 3. Write an AWS Terraform deployment architecture for this pipeline, making sure you recommend the exact GPU hardware and OS frameworks required.",
#     # "Find 1 expert on analyzing mass spectrometry proteomics data. What is the standard file format for raw mass spectrometry data? Write a Python script using the 'pyteomics' library to parse that format. Finally, design an AWS deployment architecture for the MaxQuant tool.",
#     # "Write a complete Nextflow pipeline script for trimming Illumina FASTQ reads using FastQC and Trimmomatic. Once the script is written, design an AWS Batch deployment architecture, assuming the use of EC2 Spot Instances for cost savings.",

#     # --- Group 3: Deep Coreference and Synthesis (Testing the Decomposer) ---
#     "Identify the standard file format for storing unaligned raw sequencing reads. What Python package is commonly used to parse it? Write a Python script using that package to count the number of reads, ensuring it handles both uncompressed and gzipped files.",
#     # "Find 2 human experts who have published recent papers on 'RNA-seq Splice-Aware Alignment'. Then, explain the primary algorithm those experts use. Finally, write a bash script that downloads the tool associated with that algorithm.",
    
#     # --- Group 4: The "Missing Algorithm" Refusal Test ---
#     "I need to perform a highly specialized batch-effect correction using the 'Harmony' algorithm. Write a Python script using Scanpy that explicitly runs the Harmony integration tool on an AnnData object.",
#     # "Compare the pros and cons of using AlphaFold 3 versus Rosetta for protein structure prediction. Then, write a complete Python script that utilizes the Rosetta PyRosetta library to perform de novo folding of a short amino acid sequence."
# ]

BENCHMARK_QUERIES = [
    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 1: The "Perfect Hit" Full-Stack Pipeline (Tests CUS & Pareto Frontier)
    # Goal: RAG -> Code -> Cloud. Tests if massive context compresses cleanly.
    # ══════════════════════════════════════════════════════════════════════════════
    "Explain the Leiden community detection algorithm for single-cell RNA-seq. Based on this, write a Python script using Scanpy to cluster cells using Leiden. Finally, design an AWS Terraform architecture using memory-optimized instances suitable for clustering 1 million cells.",
    "What are the mechanistic differences between PCA and UMAP in Scanpy? Write a Python script to compute both on a preprocessed AnnData object. Then, write a Terraform script to deploy this pipeline on an AWS EC2 instance.",
    "Describe the Nextflow pipeline architecture for RNA-seq using FastQC and Trimmomatic. Write a basic Nextflow script for this process, and design an AWS Batch deployment architecture utilizing EC2 Spot Instances.",
    "Explain how AlphaFold 3 predicts protein structures. Write an AWS Terraform deployment architecture for AlphaFold 3, specifically recommending the exact GPU hardware, VRAM, and OS frameworks required.",
    "What is the standard statistical method used by Scanpy to find marker genes across cell clusters? Explain the math, then write a Python script that applies this test to find differentially expressed genes and plots the top 25 genes as a dotplot.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 2: Coreference & Knowledge Transfer (Tests Protected Atom DSR)
    # Goal: Scholar -> Code/Cloud. Tests if the Atomizer protects exact human names.
    # ══════════════════════════════════════════════════════════════════════════════
    "Find 3 experts on single-cell clustering. Use their exact names to create a Pandas dataframe in a Python script, and print the dataframe.",
    "Find 2 human experts who have published recent papers on 'CRISPR base editing off-target effects'. Explain the primary detection methods these experts discuss, and design an AWS cloud architecture to process the resulting genomic data.",
    "Identify 3 top researchers working on 'synaptic plasticity STDP in the hippocampus'. Summarize their main findings regarding calcium signaling, and write a Python script to simulate basic STDP weight updates.",
    "Search for 2 experts on 'graph neural networks for protein biology'. Based on their field, explain how GNNs are applied to protein structures, and design an AWS infrastructure optimized for training custom GNN models.",
    "Find 1 expert on 'single-cell RNA sequencing tumor microenvironment'. What Python package is standard for clustering this data? Write a script using that package to find differentially expressed genes.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 3: Context Overload Stress Tests (Tests The U-Curve & Compression)
    # Goal: Massive RAG theory passed to Code. Tests if CUGCR prevents LLM distraction.
    # ══════════════════════════════════════════════════════════════════════════════
    "Compare the Needleman-Wunsch and Smith-Waterman sequence alignment algorithms in deep mathematical detail. Then, write a Python script using Biopython's pairwise2 module to execute a local alignment.",
    "Detail the algorithmic differences between Kallisto pseudoalignment and standard RNA-seq alignment. Write a shell or Python script to run Kallisto on paired-end FASTQ files.",
    "Identify the standard file format for storing unaligned raw sequencing reads and the mathematical basis of its quality scores. What Python package parses it? Write a Python script using that package to count reads in both uncompressed and gzipped files.",
    "Explain the biological function of AMPA receptor trafficking in long-term potentiation. Based on this, write a Python script to model a simple differential equation representing AMPA receptor insertion rates.",
    "What are the cloud storage tiering strategies recommended for archiving processed FASTQ and BAM files? Write a Python script using Boto3 to programmatically move files to Glacier Deep Archive based on these rules.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 4: "Missing Data" & Refusals (Tests System Resilience & Reflexion)
    # Goal: Ask for things missing from DB. Tests if AI correctly triggers ⚠️ WARNINGs.
    # ══════════════════════════════════════════════════════════════════════════════
    "I need to perform a highly specialized batch-effect correction using the 'Harmony' algorithm. Write a Python script using Scanpy that explicitly runs the Harmony integration tool on an AnnData object.",
    "Compare the pros and cons of using AlphaFold 3 versus Rosetta for protein structure prediction. Then, write a complete Python script that utilizes the Rosetta PyRosetta library to perform de novo folding.",
    "Explain how the 'Magic' imputation algorithm works for single-cell data, and write a Scanpy script to run it.",
    "Find 2 experts on Quantum Computing in Genomics. Write a Qiskit Python script to run a quantum genome assembly algorithm.",
    "Explain how to process a mass spectrometry proteomics dataset. Write a Python script to parse the raw files, and design an AWS deployment for MaxQuant.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 5: Adversarial & Ambiguous Routing (Tests the Neuro-Symbolic Decomposer)
    # Goal: Trick the system. Tests if Phase 1 LLM accurately extracts dependencies.
    # ══════════════════════════════════════════════════════════════════════════════
    "Write a Python script that both normalizes and explicitly denormalizes single-cell RNA data simultaneously using Scanpy, then deploy it to AWS Lambda.",
    "Do the thing with the single-cell data where it finds the markers, and then make it work on the cloud with terraform.",
    "Take the outputs from the top 3 researchers in transcriptomics and feed their data into a pipeline that doesn't use the cloud.",
    "Write a script to cluster cells, but only if AlphaFold 3 can predict the structure of the resulting marker genes.",
    "Design an AWS architecture for hosting a multiplayer video game backend using Scanpy.",

    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 6: Single-Agent Baselines (Tests First-Pass Success Rate - FPSR)
    # Goal: Isolated tasks. Gives the Boxplots (Figure 11 & 13) clean baseline data.
    # ══════════════════════════════════════════════════════════════════════════════
    "What is the standard statistical method used by Scanpy to find marker genes? --strategy=cot",
    "Find 4 experts who have published papers on 'Kallisto RNA-seq pseudoalignment'.",
    "Write a Terraform script to deploy an AWS r6i.4xlarge instance for GATK variant calling, including Ubuntu 22.04 and a 1TB gp3 EBS volume.",
    "Write a simple Python script using Scanpy to load an H5AD file and run PCA.",
    "Describe the tumor microenvironment in breast cancer based on recent scRNA-seq studies."
]

BENCHMARK_QUERIES = [
    # ══════════════════════════════════════════════════════════════════════════════
    # GROUP 1: The "Perfect Hit" Full-Stack Pipeline (Tests CUS & Pareto Frontier)
    # Goal: RAG -> Code -> Cloud. Tests if massive context compresses cleanly.
    # ══════════════════════════════════════════════════════════════════════════════
    "Explain the Leiden community detection algorithm for single-cell RNA-seq. Based on this, write a Python script using Scanpy to cluster cells using Leiden. Finally, design an AWS Terraform architecture using memory-optimized instances suitable for clustering 1 million cells."   
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