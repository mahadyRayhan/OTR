import asyncio
import time
from dotenv import load_dotenv
load_dotenv()

from orchestrator.engine import OrchestratorEngine

BENCHMARK_QUERIES =[
    # --- Group 1: RAG -> Code ---
    "What are the key mathematical differences between global alignment (Needleman-Wunsch) and local alignment (Smith-Waterman)? Based on those differences, write a Biopython script that executes a local alignment between two sample DNA sequences.",
    "Explain the central dogma of molecular biology. Based specifically on the translation step, write a Python script that translates a given DNA sequence into Amino Acids, ensuring you handle stop codons appropriately.",
    "Compare FASTQ and FASTA formats, and compare SAM and BAM formats. Write a Python script using Biopython that converts a FASTQ file to FASTA, and explain why you cannot convert a SAM file to FASTQ directly based on your format comparisons.",
    # --- Group 2: Code -> Cloud ---
    "Write a complete Nextflow pipeline script for trimming Illumina FASTQ reads using FastQC and Trimmomatic. Once the script is written, design an AWS Batch and S3 Terraform deployment architecture to run that exact pipeline.",
    "Write a Python script that computes the Burrows-Wheeler Transform (BWT) of a short DNA string. Then, write a complete Dockerfile and AWS ECS deployment script to run that specific Python script as a containerized microservice.",
    "Find me 2 experts on GATK, write a Snakemake script for running it, and deploy it.",
    # --- Group 3: Multi-Hop ---
    "Find 2 human experts who have published recent papers on 'RNA-seq Splice-Aware Alignment'. Then, explain the primary algorithm those experts use. Finally, write a bash script that downloads the tool associated with that algorithm.",
    "Find 3 experts on 'Spatial Transcriptomics'. What are the main computational memory bottlenecks of this specific technology? Propose an AWS EC2 and EFS architecture that explicitly addresses those exact memory bottlenecks.",
    "Find an expert who has published on 'Kallisto RNA-seq pseudoalignment'. Explain how pseudoalignment differs from traditional alignment, and write a bash script to index a transcriptome using Kallisto.",
    # --- Group 4: Aggregation & Synthesis ---
    "Compare the pros and cons of using AlphaFold 3 versus Rosetta for protein structure prediction. Then, find 3 researchers who specialize in AlphaFold. Summarize the pros/cons while explicitly citing the researchers you found.",
    "What is CRISPR screening, and what volume of data does it typically generate? Find 2 experts on CRISPR data analysis. Write a synthesis report combining the data volume challenges with the names of the experts to contact for help.",
    "Find 2 experts on single-cell RNA-seq clustering. What is the most common community-detection algorithm they use (e.g., Louvain/Leiden)? Write a Python script to implement that specific clustering algorithm using the AnnData and Scanpy libraries.",
    # --- Group 5: Vague Referent / Coreference ---
    "What is the standard text file format used for storing unaligned sequencing reads? Find a Python package designed to parse it, and write a script using that package to count the number of reads.",
    "Which cloud provider introduced the 'Fargate' serverless compute engine? Write a Terraform script to deploy a basic container to that specific service, pulling an image that runs a standard BLAST search.",
    "Identify the standard file format for storing genetic variants. Then, write a Python script using the 'pysam' library to extract the allele frequencies from it, and explain what a 'VCF header' does."
]

async def run_study():
    engine = OrchestratorEngine()
    modes = ["none", "raw", "adaptive"]
    total_runs = len(BENCHMARK_QUERIES) * len(modes)
    current_run = 0

    print(f"🚀 Starting Benchmark: {total_runs} total executions planned.")

    for idx, query in enumerate(BENCHMARK_QUERIES):
        for mode in modes:
            current_run += 1
            print(f"\n=======================================================")
            print(f"🚀 Query {idx+1}/{len(BENCHMARK_QUERIES)} | Mode: {mode.upper()} | Progress: {current_run}/{total_runs}")
            print(f"=======================================================")
            try:
                start_time = time.time()
                # Run the pipeline (passing start_time so engine can log latency)
                await engine.run(query, ablation_mode=mode, start_time=start_time)
                
                print("⏳ Resting 20s for Gemini API limits...")
                await asyncio.sleep(20) 
            except Exception as e:
                print(f"❌ Failed Run {current_run}: {e}")
                print("⏳ API blocked us. Resting 60s...")
                await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run_study())