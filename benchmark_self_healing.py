import asyncio
import time
import json
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
from orchestrator.engine import OrchestratorEngine

BENCHMARK_QUERIES = [
    # 1. RAG -> CODE
    "What are the mathematical differences between Needleman-Wunsch and Smith-Waterman? Based on those differences, write a Biopython script executing local alignment.",
    # 2. CODE -> CLOUD
    "Write a complete Nextflow pipeline script for trimming Illumina FASTQ reads using FastQC and Trimmomatic. Then design an AWS Batch Terraform deployment architecture to run it.",
    # 3. SCHOLAR -> RAG -> CODE
    "Find 2 human experts who published recent papers on 'RNA-seq Splice-Aware Alignment'. Explain the primary algorithm those experts use. Finally, write a bash script downloading the tool associated with that algorithm.",
    # 4. RAG -> CODE (Vague Referent)
    "What is the standard text file format used for storing unaligned sequencing reads? Find a Python package designed to parse it, and write a script using that package to count the number of reads.",
    # 5. RAG -> SCHOLAR -> CODE
    "Find 2 experts on single-cell RNA-seq clustering. What is the most common community-detection algorithm used in single-cell RNA-seq clustering? Finally, write a Python script to implement that algorithm using Scanpy."
]

async def run_benchmark():
    engine = OrchestratorEngine()
    total_runs = len(BENCHMARK_QUERIES)
    
    print(f"🚀 Starting Self-Healing Benchmark: {total_runs} queries planned.")

    for idx, query in enumerate(BENCHMARK_QUERIES):
        print(f"\n=======================================================")
        print(f"🚀 Running Query {idx+1}/{total_runs}")
        print(f"=======================================================")
        try:
            start_time = time.time()
            # Force "adaptive" mode so the CUGCR and Reflexion loops fire
            await engine.run(query, ablation_mode="adaptive", start_time=start_time)
            
            print("⏳ Resting 20s for Gemini API limits...")
            await asyncio.sleep(20) 
        except Exception as e:
            print(f"❌ Failed Query {idx+1}: {e}")
            await asyncio.sleep(40)

    print("\n✅ Benchmark Complete. Generating Self-Healing Report...")
    generate_healing_report()

def generate_healing_report(json_path="experiment_telemetry.json"):
    if not os.path.exists(json_path):
        print("Telemetry file not found.")
        return

    with open(json_path, "r") as f:
        logs = json.load(f)

    # We only care about the most recent 5 runs (the ones we just executed)
    recent_logs = logs[-5:] if len(logs) >= 5 else logs
    
    print("\n" + "="*80)
    print("🏥 SYSTEM SELF-HEALING & RESILIENCE REPORT")
    print("="*80)

    total_fatal_errors = 0
    total_successful_heals = 0

    for i, log in enumerate(recent_logs):
        print(f"\n🔹 Query {i+1}: {log.get('user_query')[:70]}...")
        
        # 1. & 2. In-Agent Judge Scores and REFINE Reflexion Scores
        first_draft_scores = {}
        refine_scores = {}
        
        for node in log.get("got_dag_nodes", []):
            agent = node["agent"].upper()
            score = node.get("quality_score", 0.0)
            op = node.get("operator", "unknown")
            
            # If it's a standard root/branch node
            if op != "refine" and op != "aggregate":
                if agent not in first_draft_scores:
                    first_draft_scores[agent] = score
            # If it's a REFINE node (Reflexion triggered)
            elif op == "refine":
                refine_scores[agent] = score

        print("   [Node-Level Execution]")
        for agent, score in first_draft_scores.items():
            print(f"      - {agent} initial score: {score:.2f}/1.0")
            if agent in refine_scores:
                print(f"        -> Reflexion Triggered! New score: {refine_scores[agent]:.2f}/1.0")

        # 3. & 4. Global Judge and Diagnostician Retry Loop
        final_score = log.get("final_judge_score", 0)
        retries = log.get("global_retries", 0)
        
        print(f"\n   [System-Level Synthesis]")
        if retries > 0:
            # If there was a retry, the system tried to heal a bad first draft
            print(f"      - First Draft Score: < 8.0/10 (Diagnostician Retry Triggered)")
            print(f"      - Final Score After Retry: {final_score}/10")
        else:
            print(f"      - Final Score: {final_score}/10 (No global retry needed)")

        # 5. Healing Math
        # Did the system experience a fatal node (<0.75) or a poor first draft (retry triggered)?
        had_failure = any(s < 0.75 for s in first_draft_scores.values()) or (retries > 0)
        
        if had_failure:
            total_fatal_errors += 1
            # If it had a failure, but the final score is 8 or higher, it successfully healed itself!
            if final_score >= 8:
                total_successful_heals += 1
                print("      ✅ System Successfully Self-Healed!")
            else:
                print("      ❌ System Failed to Self-Heal.")

    # 5. Final Healing Percentage
    print("\n" + "="*80)
    print("📈 OVERALL SELF-HEALING METRICS")
    print("="*80)
    print(f"Total Queries with Internal Failures: {total_fatal_errors}")
    print(f"Total Successful Recoveries (Final Score >= 8): {total_successful_heals}")
    
    healing_rate = (total_successful_heals / total_fatal_errors * 100) if total_fatal_errors > 0 else 100.0
    print(f"System Self-Healing Rate: {healing_rate:.1f}%")

if __name__ == "__main__":
    # Ensure we start with a clean slate for the 5-question test
    if os.path.exists("experiment_telemetry.json"):
        os.remove("experiment_telemetry.json")
    
    asyncio.run(run_benchmark())