"""
run_adaptive_and_compare.py
===========================
1. Runs the adaptive pipeline ONLY for all benchmark queries (no none/raw).
2. Saves results to raw_telemetry_v2/ (leaves the original raw_telemetry/ untouched).
3. Loads already-saved GPT-5 and Gemini answers from
   results/llm_baseline_comparison/comparison_results.json
4. Judges ONLY the new adaptive answers (baselines already have scores).
5. Saves updated comparison JSON + new chart.

Usage:
    python scripts/run_adaptive_and_compare.py
    python scripts/run_adaptive_and_compare.py --queries 5   # quick test
    python scripts/run_adaptive_and_compare.py --skip-judge  # skip re-judging, just plot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Parse args early (needed before engine import) ───────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--queries",    type=int, default=0)
_parser.add_argument("--skip-judge", action="store_true")
_parser.add_argument("--out-dir",    default="")
_args, _ = _parser.parse_known_args()

OUTPUT_DIR   = Path(_args.out_dir) if _args.out_dir else ROOT / "raw_telemetry_v2"
BASELINE_FILE = ROOT / "results" / "llm_baseline_comparison" / "comparison_results.json"
OUT_JSON      = ROOT / "results" / "llm_baseline_comparison" / "comparison_results_v2.json"
OUT_CHART     = ROOT / "results" / "llm_baseline_comparison" / "comparison_chart_v2.png"
ENGINE_TEL    = ROOT / "experiment_telemetry.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from orchestrator.engine import OrchestratorEngine

BENCHMARK_QUERIES = [
    # Group 1 — Full-stack pipeline
    "Explain the Leiden community detection algorithm and how it guarantees well-connected communities, improving upon Louvain. Write a Python script using Scanpy to cluster single-cell RNA-seq cells with Leiden. Design an AWS Terraform architecture using memory-optimized instances for clustering 1 million cells.",
    "Describe the splice-aware alignment algorithm used by STAR for RNA-seq, including its suffix array index and two-pass strategy. Write a shell script to run STAR alignment on paired-end FASTQ files. Design an AWS Batch architecture for parallelizing STAR across 100 samples using EC2 Spot Instances.",
    "Explain the statistical model behind DESeq2 for differential expression analysis, including the negative binomial distribution and empirical Bayes dispersion shrinkage. Write a Python script using pydeseq2 on a count matrix. Deploy this pipeline on an AWS EC2 r6i instance.",
    "Explain how the SAM and BAM file formats store sequencing alignment data, including CIGAR strings, FLAG bit encoding, and optional tags. Write a Python script using pysam to parse a BAM file and compute per-read alignment statistics. Design an AWS S3 tiered-storage architecture for managing BAM files from a large cohort.",
    "Describe the Weighted Nearest Neighbor (WNN) multimodal integration method from the Seurat paper for joint RNA and protein analysis. Write a Python script using Scanpy to perform WNN-based joint dimensionality reduction. Design an AWS architecture for multimodal single-cell analysis at scale.",
    # Group 2 — Scholar coreference
    "Find 3 experts on Leiden community detection in single-cell genomics. Use their exact names to create a Pandas dataframe in a Python script, and print the dataframe.",
    "Find 2 human experts who have published papers on programmable CRISPR base editing. Explain the key biochemical differences between cytosine base editors and adenine base editors, and design an AWS cloud architecture for storing the resulting variant call files.",
    "Identify 3 top researchers working on splice-aware RNA-seq alignment algorithms. Summarize their main algorithmic contributions, and write a Python script to run STAR alignment with their recommended two-pass parameters.",
    "Search for 2 experts on multimodal single-cell data integration. Based on their research field, explain the WNN graph construction approach, and design an AWS infrastructure optimized for storing and querying paired RNA and ATAC-seq datasets.",
    "Find 1 expert on differential expression analysis using count-based RNA-seq models. What statistical framework do they recommend? Write a Python script implementing that framework on a raw count matrix.",
    # Group 3 — Context overload
    "Compare the Leiden and Louvain community detection algorithms in deep mathematical detail, including their modularity optimization guarantees and why Leiden produces well-connected communities while Louvain does not. Write a Python script using leidenalg to cluster a graph with custom resolution parameters.",
    "Detail the STAR aligner's genome index construction using suffix arrays and its two-pass strategy for novel splice junction discovery. Explain the mathematical basis of its seed-and-extend alignment. Write a comprehensive shell script for a STAR two-pass alignment pipeline on paired-end FASTQ files.",
    "Explain the DESeq2 dispersion estimation pipeline in detail: Cox-Reid adjusted profile likelihood, maximum likelihood dispersion estimates, and empirical Bayes shrinkage toward the fitted trend. Write a Python script using pydeseq2 to reproduce this workflow and plot a dispersion estimates figure.",
    "Describe the biochemical mechanisms of both cytosine base editing (C·G to T·A via deamination) and adenine base editing (A·T to G·C via adenosine deaminase), including the role of UGI and the DNA repair pathways involved. Write a Python script to scan a FASTA genome for NGG PAM-adjacent base editing windows.",
    "Explain the full SAM file format specification, including all mandatory fields, every CIGAR string operation (M, I, D, N, S, H, P, =, X), FLAG bitmask encoding, and the MAPQ quality formula. Write a Python script using pysam to filter a BAM file keeping only properly paired primary alignments with MAPQ above 30.",
    # Group 4 — Missing data / resilience
    "I need to perform single-cell trajectory inference using the Monocle3 algorithm. Write a Python script using Scanpy that explicitly runs Monocle3 pseudotime analysis on an AnnData object.",
    "Compare the pros and cons of using Salmon versus Kallisto for RNA-seq quantification. Write a complete Python script using Kallisto to perform pseudoalignment on paired-end FASTQ files.",
    "Explain how the MAGIC imputation algorithm reconstructs gene expression for single-cell data, and write a Scanpy script to run it on a preprocessed AnnData object.",
    "Find 2 experts on Quantum Computing in Genomics. Write a Qiskit Python script to run a quantum sequence alignment algorithm.",
    "Explain how to process a whole-genome sequencing dataset through GATK variant calling. Write a Snakemake workflow for BQSR and HaplotypeCaller, and design an AWS Batch deployment for this pipeline.",
    # Group 5 — Adversarial
    "Write a Python script that both aligns and simultaneously un-aligns RNA-seq reads to the reference genome using STAR, then deploy it to AWS Lambda.",
    "Do the thing with the single-cell data where it finds the communities using that graph algorithm from the Leiden paper, and make it work on the cloud with Terraform.",
    "Take the outputs from the top 3 researchers in RNA-seq alignment and feed their recommended STAR parameters into a pipeline that doesn't use the cloud.",
    "Write a script to perform differential expression with DESeq2, but only if DESeq2 can also predict the secondary structure of the resulting DE genes.",
    "Design an AWS architecture for hosting a multiplayer video game backend using DESeq2.",
    # Group 6 — Single-agent baselines
    "What negative binomial distribution parameters does DESeq2 use to model RNA-seq read counts, and how does it estimate dispersion? --strategy=cot",
    "Find 4 experts who have published papers on 'STAR RNA-seq splice junction alignment'.",
    "Write a Terraform script to deploy an AWS r6i.4xlarge instance for STAR alignment, including Ubuntu 22.04 and a 1TB gp3 EBS volume.",
    "Write a simple Python script using pysam to open a BAM file, iterate over reads, and print the name and mapping quality of the first 10 alignments.",
    "retrieve the exact two-pass alignment parameters from the STAR paper and write a Python script using those specific parameter values.",
    # Group 7 — Context volume stress
    "Retrieve ALL algorithmic details from papers on the STAR aligner including suffix array construction, two-pass alignment strategy, seed-and-extend scoring, and CIGAR string encoding. Then write a comprehensive Python library with minimum 12 functions (full docstrings, type hints, error handling) covering all major STAR parameters, input validation, and output parsing. Finally, design a complete AWS Batch + S3 + EC2 Spot Instance architecture for processing 500 paired-end samples in parallel, including IAM roles, cost optimization, and CloudWatch monitoring.",
    "Retrieve comprehensive documentation on BOTH the Leiden community detection algorithm AND the Seurat Weighted Nearest Neighbor (WNN) multimodal integration method. Synthesize the complete mathematical theory behind both methods including modularity optimization, well-connectedness guarantees, and weighted graph construction. Then write a unified Python pipeline that preprocesses scRNA-seq data, applies WNN-based joint embedding, clusters with Leiden at multiple resolutions, performs differential expression between clusters, and visualizes results in UMAP space. Deploy this complete pipeline on an AWS p3.2xlarge GPU instance with automated dependency installation.",
    "Retrieve the full SAM/BAM format specification including ALL mandatory fields, every CIGAR string operation (M, I, D, N, S, H, P, =, X), the complete FLAG bitmask encoding for all 12 bits, the MAPQ quality formula, and optional tag specifications. Then write a complete Python library implementing: a SAM/BAM parser, CIGAR string interpreter for all 9 operations, FLAG decoder with named bit fields, per-read quality statistics calculator, proper-pair filter, and duplicate read detector using coordinate sorting. Deploy this library as an AWS Lambda function behind API Gateway that accepts BAM file S3 URIs and returns JSON alignment statistics.",
    "Find 4 researchers who have published on DESeq2 statistical modelling and dispersion estimation. For each researcher retrieve their key publications and extract every statistical formula described including: negative binomial GLM parameterization, Cox-Reid adjusted profile likelihood, maximum likelihood dispersion estimates, empirical Bayes shrinkage toward the fitted trend, and independent filtering thresholds. Write a Python implementation that reproduces the complete DESeq2 dispersion estimation workflow using pydeseq2, including visualization of the dispersion estimates plot. Then deploy this reproducible analysis on AWS EC2 r6i.2xlarge with automated R and Python environment setup via user-data scripts.",
    "Retrieve all available papers on CRISPR base editing covering both cytosine base editors (CBE) and adenine base editors (ABE), including: the biochemical deamination mechanisms, UGI roles, DNA repair pathway involvement, PAM sequence requirements, editing window positions, and known off-target profiles. Write a comprehensive Python toolkit with: a PAM site scanner supporting NGG, NG, and NGA, an editing window calculator for both CBE and ABE, a guide RNA scorer, an off-target predictor using seed region hashing, and a variant effect annotator that maps edits to coding consequences. Design an AWS Batch architecture for genome-scale analysis of 100,000 potential edit sites with results stored in S3 Parquet format and queryable via Athena.",
]

DIMS = ["Correctness", "Completeness", "Grounding", "Specificity", "Code_Quality", "Coherence"]
COLORS = {"adaptive": "#2ecc71", "gemini": "#3498db", "gpt": "#e67e22"}


# ── Telemetry helpers ─────────────────────────────────────────────────────────

def read_engine_telemetry() -> dict:
    if not ENGINE_TEL.exists():
        return {}
    try:
        data = json.loads(ENGINE_TEL.read_text())
        if isinstance(data, list) and data:
            return data[-1]
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_run(telemetry: dict, metadata: dict) -> str:
    run_id = str(uuid.uuid4())[:8]
    record = {"run_id": run_id, "timestamp": datetime.utcnow().isoformat(),
              **metadata, "engine_telemetry": telemetry}
    path = OUTPUT_DIR / f"run_{run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return run_id


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator of bioinformatics AI systems.

QUESTION: "{query}"

ANSWER TO EVALUATE:
{answer}

Score this answer on each dimension from 0-10. Return ONLY valid JSON, no prose:

{{
  "Correctness":   <0-10>,
  "Completeness":  <0-10>,
  "Grounding":     <0-10>,
  "Specificity":   <0-10>,
  "Code_Quality":  <0-10>,
  "Coherence":     <0-10>,
  "reasoning":     "<one sentence>"
}}

Scoring guide:
  Correctness  - factual accuracy, no hallucinations
  Completeness - covers all parts of the question
  Grounding    - cites specific algorithms, papers, tools, or data (not generic claims)
  Specificity  - concrete parameter values, exact tool names, real examples
  Code_Quality - runnable code with proper structure (10 if no code needed, 0 if broken)
  Coherence    - logical flow and clarity"""


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


async def judge_answer(query: str, answer: str, client) -> dict:
    from openai import AsyncOpenAI
    prompt = JUDGE_PROMPT.format(query=query[:300], answer=answer[:3000])
    try:
        resp = await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "developer", "content": "You are a strict JSON-only evaluator. Output only valid JSON."},
                {"role": "user",      "content": prompt},
            ],
        )
        data = _extract_json(resp.choices[0].message.content.strip())
        if not data:
            raise ValueError("No JSON in response")
        scores = {d: float(data.get(d, 0)) for d in DIMS}
        scores["mean"] = round(sum(scores[d] for d in DIMS) / len(DIMS), 2)
        scores["reasoning"] = data.get("reasoning", "")
        return scores
    except Exception as e:
        print(f"   ✘ Judge error: {e}")
        return {d: 0.0 for d in DIMS} | {"mean": 0.0, "reasoning": f"Judge error: {e}"}


# ── Chart ─────────────────────────────────────────────────────────────────────

def make_chart(results: list[dict], out: Path):
    systems = [s for s in ["adaptive", "gemini", "gpt"] if any(s in r for r in results)]

    sys_means = {s: [] for s in systems}
    dim_means = {s: {d: [] for d in DIMS} for s in systems}

    for r in results:
        for s in systems:
            if s in r and r[s].get("scores"):
                sys_means[s].append(r[s]["scores"].get("mean", 0))
                for d in DIMS:
                    dim_means[s][d].append(r[s]["scores"].get(d, 0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    _display = {"adaptive": "Adaptive\n(fixed)", "gemini": "Gemini", "gpt": "GPT-5"}

    # Left — overall
    ax = axes[0]
    means = [np.mean(sys_means[s]) for s in systems]
    stds  = [np.std(sys_means[s])  for s in systems]
    x = np.arange(len(systems))
    bars = ax.bar(x, means, 0.5, color=[COLORS[s] for s in systems], edgecolor="white",
                  yerr=stds, capsize=5, error_kw={"linewidth": 1, "color": "#555"}, zorder=3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_display[s] for s in systems], fontsize=10)
    ax.set_ylabel("Mean Judge Score (0–10)", fontsize=9)
    ax.set_title("Overall Quality — After Fixes", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Right — per-dimension
    ax2 = axes[1]
    x2 = np.arange(len(DIMS))
    width = 0.25
    offsets = np.linspace(-(len(systems)-1)*width/2, (len(systems)-1)*width/2, len(systems))
    for s, offset in zip(systems, offsets):
        vals = [np.mean(dim_means[s][d]) for d in DIMS]
        ax2.bar(x2 + offset, vals, width, label=_display[s].replace("\n", " "),
                color=COLORS[s], edgecolor="white", zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([d.replace("_", "\n") for d in DIMS], fontsize=8)
    ax2.set_ylabel("Score (0–10)", fontsize=9)
    ax2.set_title("Per-Dimension Breakdown — After Fixes", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, color="#eee", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    n = len(results)
    fig.suptitle(f"Adaptive (fixed) vs Vanilla LLM Baselines  (N={n} queries)", fontsize=11, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    queries = BENCHMARK_QUERIES[:_args.queries] if _args.queries > 0 else BENCHMARK_QUERIES

    # ── Load existing baseline answers ───────────────────────────────────────
    baselines: dict[str, dict] = {}   # query → {gemini: {...}, gpt: {...}}
    if BASELINE_FILE.exists():
        for rec in json.loads(BASELINE_FILE.read_text()):
            q = rec.get("query", "").strip()
            baselines[q] = rec
        print(f"Loaded {len(baselines)} baseline records from {BASELINE_FILE.name}")
    else:
        print(f"WARNING: No baseline file found at {BASELINE_FILE}")
        print("Run scripts/eval/llm_baseline_comparison.py first.")

    if not _args.skip_judge:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    # ── Run adaptive pipeline ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Running ADAPTIVE ONLY — {len(queries)} queries → {OUTPUT_DIR.name}/")
    print(f"{'═'*60}\n")

    engine = OrchestratorEngine()
    if ENGINE_TEL.exists():
        ENGINE_TEL.unlink()

    adaptive_answers: dict[str, str] = {}   # query → final_answer

    for i, query in enumerate(queries, 1):
        print(f"\n── Query {i}/{len(queries)} ──────────────────────────────")
        print(f"   {query[:90]}{'...' if len(query) > 90 else ''}")
        try:
            cached_tasks, cached_decomp = await engine.decompose_only(query)
            await asyncio.sleep(2)

            t0 = time.time()
            await engine.run(query, ablation_mode="adaptive", start_time=t0,
                             cached_tasks=cached_tasks,
                             cached_decomposer_name=cached_decomp)
            latency = int((time.time() - t0) * 1000)

            tel = read_engine_telemetry()
            final_answer = tel.get("final_answer", "")
            adaptive_answers[query] = final_answer

            run_id = save_run(tel, {
                "protocol": "B_adaptive_fixed", "query_id": f"BQ{i:02d}",
                "ablation_mode": "adaptive", "query": query,
                "wall_latency_ms": latency,
            })
            print(f"  ✔ Saved run_{run_id}.json ({latency}ms) | answer={len(final_answer)} chars")
            await asyncio.sleep(1)

        except Exception as e:
            print(f"  ✘ Error: {e}")
            adaptive_answers[query] = ""
            await asyncio.sleep(3)

    # ── Judge new adaptive answers ────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Judging {len(adaptive_answers)} adaptive answers...")
    print(f"{'═'*60}")

    results = []
    for i, query in enumerate(queries, 1):
        adaptive_ans = adaptive_answers.get(query, "")
        baseline = baselines.get(query, {})

        print(f"\n── Q{i:02d} Judging ──────────────────────────────────────")
        print(f"   {query[:80]}...")

        if _args.skip_judge:
            adaptive_scores = baseline.get("adaptive", {}).get("scores", {d: 0 for d in DIMS})
            adaptive_scores.setdefault("mean", 0)
            adaptive_scores.setdefault("reasoning", "skip-judge mode")
        elif adaptive_ans:
            adaptive_scores = await judge_answer(query, adaptive_ans, openai_client)
            print(f"   Adaptive(fixed) mean={adaptive_scores['mean']:.2f} | {adaptive_scores['reasoning'][:80]}")
        else:
            adaptive_scores = {d: 0.0 for d in DIMS} | {"mean": 0.0, "reasoning": "No answer generated"}

        record = {
            "query_id": f"Q{i:02d}",
            "query": query,
            "adaptive": {"answer": adaptive_ans[:500], "scores": adaptive_scores},
        }
        if "gemini" in baseline:
            record["gemini"] = baseline["gemini"]
        if "gpt" in baseline:
            record["gpt"] = baseline["gpt"]

        results.append(record)
        await asyncio.sleep(0.5)

    # ── Save + print table ────────────────────────────────────────────────────
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_JSON}")

    systems = [s for s in ["adaptive", "gemini", "gpt"] if any(s in r for r in results)]
    print(f"\n{'─'*58}")
    print(f"  {'System':<16} {'N':>4}  {'Mean Score':>11}  {'Std':>7}")
    print(f"{'─'*58}")
    for s in systems:
        scores = [r[s]["scores"].get("mean", 0) for r in results if s in r]
        label = "Adaptive(fixed)" if s == "adaptive" else s.capitalize()
        print(f"  {label:<16} {len(scores):>4}  {np.mean(scores):>9.2f}/10  {np.std(scores):>7.2f}")
    print(f"{'─'*58}")

    print("\nPer-dimension:")
    print(f"  {'Dim':<16}", end="")
    for s in systems:
        label = "Adapt(fix)" if s == "adaptive" else s.capitalize()
        print(f"  {label:>10}", end="")
    print()
    for d in DIMS:
        print(f"  {d:<16}", end="")
        for s in systems:
            vals = [r[s]["scores"].get(d, 0) for r in results if s in r]
            print(f"  {np.mean(vals):>10.2f}", end="")
        print()

    make_chart(results, OUT_CHART)
    print(f"\nDone. Chart → {OUT_CHART}")


if __name__ == "__main__":
    asyncio.run(main())
