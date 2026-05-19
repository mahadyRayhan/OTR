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
    # ══════════════════════════════════════════════════════════════════════════
    # LEIDEN  (10 queries) — RAG→Code→Cloud on community detection paper
    # ══════════════════════════════════════════════════════════════════════════
    "Explain the Leiden community detection algorithm and how its refinement phase guarantees well-connected communities, improving upon Louvain's tendency to produce poorly-connected clusters. Write a Python script using Scanpy to cluster a single-cell RNA-seq AnnData with Leiden at resolution 0.3 and 0.8, generate UMAP embeddings colored by cluster, and compute silhouette scores to select the optimal resolution. Design an AWS EC2 r6i.4xlarge Terraform architecture with EFS for shared AnnData storage.",

    "Explain how the Leiden algorithm uses the Constant Potts Model (CPM) quality function and how the resolution parameter γ controls community size — specifically why increasing γ leads to smaller communities. Write a Python script using leidenalg to cluster a NetworkX graph at five resolution values (0.01, 0.1, 0.5, 1.0, 2.0), plot the number of detected communities vs resolution, and compute the modularity score at each level. Deploy as an AWS Batch job that processes multiple graph files from S3.",

    "Explain how Leiden's local node-move phase works and why it achieves Pareto-optimal communities that Louvain cannot guarantee. Write a Python script using Scanpy and leidenalg to: (1) load a preprocessed AnnData, (2) cluster with Leiden at resolution 0.5, (3) run rank_genes_groups() to identify marker genes per cluster, (4) generate a heatmap of the top 5 marker genes, and (5) export per-cluster summary statistics to CSV. Design an AWS SageMaker notebook environment with an EBS-backed instance for interactive exploration.",

    "Describe the three algorithmic phases of Leiden — local moves, refinement, and network aggregation — and explain why network aggregation allows detection of communities at multiple scales efficiently. Write a Python script using Scanpy to run Leiden on a PBMC single-cell dataset, compute UMAP, annotate clusters with known marker genes (CD3E for T cells, CD19 for B cells, CD14 for monocytes), and export an annotated H5AD. Deploy on AWS EC2 r6i.2xlarge with a user-data script that installs all Python dependencies.",

    "Explain the well-connectedness guarantee of the Leiden algorithm: why Leiden ensures that each community has no subset S where the internal cut is weaker than the external cut, and how this prevents the disconnected-community problem found in Louvain. Write a Python script using leidenalg and NetworkX to: (1) generate a planted-partition graph, (2) apply Leiden clustering, (3) verify well-connectedness of each community by checking internal vs external edge weights, and (4) compare to Louvain communities on the same graph. Deploy this validation pipeline on AWS Lambda.",

    "Explain how the Leiden algorithm handles weighted graphs and what happens to community structure when edge weights represent biological similarity scores such as shared gene expression. Write a Python script using Scanpy that: (1) constructs a kNN graph from PCA embeddings with cosine distances, (2) runs Leiden at resolution 0.5 and 1.0, (3) computes within-cluster gene expression variance as a stability metric, and (4) generates a violin plot of the top 3 marker genes per cluster. Design an AWS EC2 r6i.2xlarge architecture with S3 for input and output data.",

    "Describe how the Leiden algorithm achieves scalable community detection for very large graphs through hierarchical aggregation, and why this enables efficient clustering of million-cell single-cell datasets. Write a Python script using Scanpy to cluster 500,000 cells with Leiden at resolution 0.4, monitor memory usage during graph construction and clustering with tracemalloc, and export a UMAP with cluster labels as a publication-quality figure. Design an AWS EC2 r6i.8xlarge architecture with EFS for shared storage in a multi-user environment.",

    "Explain how the Leiden algorithm detects hierarchical community structure at different resolutions and why this is useful for identifying both coarse cell types and fine cell subtypes in single-cell data. Write a Python script using Scanpy that: (1) runs Leiden at three resolutions (0.2, 0.5, 1.5), (2) computes the Adjusted Rand Index between each pair of resolutions to quantify hierarchical consistency, (3) generates a three-level cluster annotation using marker gene expression, and (4) saves the annotated AnnData. Deploy on AWS EC2 r6i.4xlarge.",

    "Explain the difference between modularity-based and CPM-based community detection as used in Leiden, and why CPM avoids the resolution limit that affects modularity optimization. Write a Python script using leidenalg that: (1) clusters a NetworkX graph using both Leiden-modularity and Leiden-CPM, (2) computes community count and mean size for each method, (3) evaluates both partitions using ARI against a known ground-truth partition, and (4) generates comparison bar charts. Deploy as an AWS Batch job that processes multiple benchmark graphs from S3.",

    "Explain how Leiden community detection is applied in spatial transcriptomics to identify spatially coherent cell communities, and what graph construction approach captures spatial relationships. Write a Python script using Squidpy and Scanpy to: (1) load a Visium spatial transcriptomics dataset, (2) construct a spatial neighbor graph using sq.gr.spatial_neighbors, (3) run Leiden clustering on the spatial graph at resolution 0.4, and (4) plot clusters on the tissue image with spatial coordinates. Design an AWS EC2 r6i.4xlarge deployment with S3 storage for spatial data files.",

    # ══════════════════════════════════════════════════════════════════════════
    # CRISPR ADENINE BASE EDITING  (10 queries)
    # ══════════════════════════════════════════════════════════════════════════
    "Explain the adenine base editing mechanism by which ABE editors convert A·T to G·C base pairs through an engineered tRNA adenosine deaminase domain (TadA), and why this required a fundamentally different protein engineering strategy than cytosine base editors. Write a Python script to scan a FASTA file for NGG PAM sites in a target locus and identify adenines within the ABE editing window (positions 4–7). Design an AWS Batch architecture for genome-wide ABE site prediction on a reference genome stored in S3.",

    "Describe how the ABE7.10 editor achieves adenine deamination in single-stranded DNA using an evolved TadA heterodimer, and what protein engineering technique (PACE — phage-assisted continuous evolution) was used to create this DNA-acting deaminase. Write a Python script that takes a gene FASTA, scans for all NGG PAM sites, scores each guide by on-target A count within positions 4–7 and off-target PAM frequency, and exports a ranked TSV. Deploy as an AWS Lambda function reading FASTA from S3 and writing TSV results to S3.",

    "Explain why ABE editors required an entirely different protein scaffold than CBE editors — specifically why no naturally occurring DNA adenosine deaminase exists and what the evolutionary rationale was for engineering from a tRNA deaminase. Write a Python script using Biopython to: (1) read a reference genome FASTA, (2) scan a target chromosome for all NGG and NGA PAM sites, (3) extract the 20-nt guide + PAM for each, (4) compute ABE editing window positions (4–7), and (5) output a BED file of editable adenines with guide sequences. Design an AWS Batch Spot Instance architecture for genome-wide scanning.",

    "Describe the product of adenine base editing — inosine — and the DNA repair mechanism that converts inosine to guanine, resulting in the final A·T to G·C transition. Write a Python script that: (1) takes a VCF of known disease-causing A-to-G variants, (2) for each variant checks whether it falls within positions 4–7 of any NGG PAM window within 23 bp upstream, (3) outputs a TSV of editable vs unreachable variants with the nearest guide sequence and PAM, and (4) plots a pie chart of editable vs unreachable fractions. Deploy on AWS EC2 r5.large with input VCF read from S3.",

    "Explain the key differences between ABE7.10 and the improved ABEmax editor — specifically what architectural changes (codon optimization, bipartite NLS) improve editing efficiency and nuclear localization. Write a Python script using Biopython that: (1) reads a FASTA file, (2) scans for both NGG and NGAG PAMs, (3) for each guide computes the ABE editing window, GC content, and self-complementarity score via a simple hairpin check, (4) ranks guides by predicted efficiency (high seed-region GC, no internal hairpin), and (5) writes the top 20 guides per gene to a TSV. Design an AWS Batch pipeline for cohort-scale variant reachability analysis.",

    "Explain the concept of bystander editing in ABE — what happens when multiple adenines fall within the editing window — and what strategies are used to minimize unintended A-to-G edits at non-target positions. Write a Python script that: (1) scans a target genomic region for NGG PAM sites, (2) for each guide counts all adenines within positions 4–7, (3) flags guides with adenines at non-target positions as bystander risk, (4) ranks guides by lowest bystander count, and (5) generates a heatmap of per-window adenine positions across all guide candidates. Deploy on AWS EC2 r5.xlarge.",

    "Explain how ABE editors are delivered into cells — plasmid transfection, lentiviral, AAV, and ribonucleoprotein — and what constraints the large size of ABE (>5 kb coding sequence) imposes on AAV-based in vivo delivery. Write a Python script using pysam that: (1) reads a BAM file from an ABE amplicon sequencing experiment, (2) identifies reads with A-to-G edits at expected guide positions, (3) computes on-target editing efficiency (fraction of reads edited at position 6), (4) computes bystander rate at each other window position, and (5) exports a per-guide editing summary. Design an AWS Batch pipeline for processing thousands of amplicon sequencing BAM files from S3.",

    "Describe the clinical applications of ABE editors for correcting pathogenic point mutations — specifically sickle cell disease (HbS: Glu6Val, GAG→GTG) and beta-thalassemia — and explain why ABE is well-suited for reversing G-to-A transition mutations. Write a Python script that: (1) reads a ClinVar pathogenic variant CSV filtered for G-to-A transitions, (2) for each variant searches for NGG PAM sites within 23 bp where the target A falls in positions 4–7, (3) scores each guide candidate, and (4) exports a therapeutic guide design TSV. Deploy on AWS EC2 r5.large with input and output on S3.",

    "Explain the differences between ABE8e and earlier ABE versions — specifically what hyperactive TadA mutations increase editing efficiency and what tradeoffs exist with respect to RNA off-target edits. Write a Python script to scan a reference FASTA for all ABE guide candidates in a specified interval: (1) collect all NGG PAMs, (2) extract 20-nt guides, (3) compute ABE editing window and flag multi-adenine windows, (4) use Bowtie2 via subprocess to count off-target alignment sites per guide, and (5) rank guides by on-target specificity score. Deploy as an AWS Batch job with S3 input and output.",

    "Explain how prime editing compares to ABE for installing A-to-G corrections — specifically what makes ABE more efficient for simple transition edits while prime editing enables insertions and deletions. Write a Python script that: (1) reads a variant list CSV, (2) for each A-to-G variant finds the optimal ABE guide (NGG PAM, A at positions 4–7), (3) for variants where ABE is not feasible designs a pegRNA scaffold including RT template and PBS sequence, and (4) outputs a combined design file with recommended edit strategy per variant. Design an AWS Batch architecture that processes a full exome VCF and outputs a guide design report.",

    # ══════════════════════════════════════════════════════════════════════════
    # CRISPR CYTOSINE BASE EDITING  (8 queries)
    # ══════════════════════════════════════════════════════════════════════════
    "Explain the cytosine base editing mechanism in BE3 — how APOBEC1 deaminates cytosine to uracil in a single-stranded DNA loop, and how UGI blocks uracil-DNA glycosylase to prevent base excision repair and increase C-to-T editing efficiency. Write a Python script to scan a FASTA file for all NGG PAM sites and identify cytosines within the CBE editing window (positions 4–8). Design an AWS Batch architecture for genome-wide CBE site scanning on a reference FASTA stored in S3.",

    "Describe the evolution of cytosine base editors from BE1 to BE4max — what each iteration improved (UGI addition in BE3, codon optimization and second UGI in BE4, reduced off-target in evoAPOBEC), and what role the D10A nickase Cas9 plays in improving editing outcomes. Write a Python script that: (1) reads a target gene FASTA, (2) scans for NGG PAM sites, (3) for each guide scores the CBE editing window (positions 4–8) for C count and GC content, (4) flags guides with bystander Cs outside the target position, and (5) ranks by fewest bystander edits. Deploy on AWS EC2 r5.large with output written to S3.",

    "Explain the role of the UGI domain in cytosine base editors — specifically what enzyme it inhibits, what repair pathway it blocks, and why two UGI copies (as in BE4) increase C-to-T conversion purity by further suppressing uracil excision. Write a Python script using Biopython that: (1) parses a genomic FASTA, (2) identifies all CBE-targetable cytosines in a gene (positions 4–8 of any NGG PAM), (3) classifies each as target C or potential bystander, and (4) outputs a ranked guide design TSV sorted by editing window C-count and bystander risk. Deploy as an AWS Lambda function reading from S3.",

    "Describe how cytosine base editors generate C-to-T edits via a C-to-U intermediate and why C-to-G or C-to-A transversions sometimes occur as unintended outcomes depending on which repair pathway resolves the U:G mismatch. Write a Python script that reads a BAM file from a CBE amplicon sequencing experiment: (1) for each read at the target locus classify the edit at each position in the CBE window as C→T, C→G, C→A, or unedited, (2) compute per-position editing frequencies per outcome, (3) generate a stacked bar plot of edit outcomes per window position, and (4) export a TSV summary. Design an AWS Batch pipeline for 96-well CBE screening.",

    "Explain what determines CBE editing efficiency at a given site — including PAM-proximal context, guide RNA GC content, and the cytosine's position within the editing window (positions 4–5 show higher efficiency than 7–8). Write a Python script using Biopython that: (1) reads a FASTA file and target region, (2) enumerates all NGG PAM sites, (3) for each guide computes GC content, number of cytosines per window sub-position, and a composite efficiency score, (4) ranks guides by the efficiency score, and (5) exports the top 10 guides per target with predicted editing outcomes to a TSV. Deploy on AWS Batch with Spot Instances.",

    "Describe the therapeutic applications of CBE for correcting C-to-T pathogenic mutations — including HBB beta-thalassemia mutations, PCSK9 gain-of-function variants, and the introduction of protective APOE variants — and why CBE is preferred over HDR for these single-nucleotide corrections. Write a Python script that: (1) reads a ClinVar pathogenic variant TSV filtered for C-to-T transitions, (2) for each variant searches for NGG PAM sites where the target C is at positions 4–8, (3) scores guide candidates, and (4) exports a therapeutic guide design summary with chromosomal positions. Design an AWS EC2 r5.large pipeline with S3 for clinical genomics data.",

    "Explain the difference between base editing (CBE) and CRISPR-Cas9 HDR for correcting point mutations — specifically why CBE is more efficient in non-dividing post-mitotic cells where HDR is suppressed, and what limitations CBE faces for non-C-to-T edits. Write a Python script that takes a pathogenic SNP CSV and for each SNP: (1) determines if CBE is applicable (C-to-T, within NGG window), (2) designs the optimal BE3 guide, (3) estimates efficiency from window position and GC content, (4) flags variants requiring ABE or prime editing. Export a variant-to-editor assignment table. Deploy on AWS Batch.",

    "Explain how narrow-window CBE variants (eA3A, YE1) differ from standard BE3 — specifically what amino acid substitutions in APOBEC reduce the editing window from positions 4–8 to a narrower C5-preferred window, reducing bystander edits. Write a Python script that: (1) scans a target gene FASTA for all NGG PAM sites, (2) for each guide computes Cs in the narrow window (position 5) and full window (positions 4–8), (3) simulates expected editing outcomes for both BE3 and eA3A, (4) generates a per-guide comparison showing bystander reduction from window narrowing, and (5) exports results to TSV. Deploy on AWS EC2 r5.xlarge.",

    # ══════════════════════════════════════════════════════════════════════════
    # DESEQ2  (8 queries)
    # ══════════════════════════════════════════════════════════════════════════
    "Explain the statistical model behind DESeq2 for differential expression analysis, including why the negative binomial distribution is used for count data, how size factors normalize for sequencing depth, and how empirical Bayes dispersion shrinkage improves statistical power in small-sample experiments. Write a Python script using pydeseq2 on a raw count matrix CSV with two conditions and 3 replicates per group. Deploy this pipeline on an AWS EC2 r6i.2xlarge instance with S3 input and output.",

    "Explain how DESeq2 estimates size factors using the median-of-ratios method and why this is more robust than total-count normalization for RNA-seq data with a small number of highly expressed genes. Write a Python script using pydeseq2 that: (1) loads a raw count matrix, (2) computes size factors manually using the geometric mean per gene and median per sample, (3) verifies they match pydeseq2's internal estimates, (4) generates a boxplot of normalized vs raw counts per sample, and (5) exports normalized counts to CSV. Deploy on AWS EC2 r6i.xlarge.",

    "Describe DESeq2's empirical Bayes dispersion shrinkage: how it fits a mean-dispersion trend, computes maximum-likelihood estimates per gene, and shrinks these toward the trend to reduce variance for lowly expressed genes. Write a Python script using pydeseq2 that: (1) loads count data, (2) fits dispersions with both parametric and local fit types, (3) generates a dispersion-mean scatter plot overlaying raw MLE and shrunken MAP estimates with the fitted trend line, and (4) highlights genes with unusually high dispersion. Deploy on AWS EC2 r6i.2xlarge.",

    "Explain the DESeq2 Wald test: how the test statistic is computed from the log2 fold-change and its standard error, how the Benjamini-Hochberg procedure controls FDR, and what the lfcShrink function does to reduce noise in fold-change estimates for gene ranking. Write a Python script using pydeseq2 that: (1) loads a 2-condition count matrix, (2) runs the Wald test, (3) applies lfcShrink, (4) generates an MA plot and volcano plot, and (5) exports genes with padj < 0.05 and |log2FC| > 1 to TSV. Deploy on AWS EC2 r6i.2xlarge.",

    "Explain how DESeq2 handles multi-factor experimental designs with batch effects — specifically how including a batch covariate in the design formula (~batch + condition) accounts for technical variation while preserving biological signal. Write a Python script using pydeseq2 that: (1) loads a count matrix with condition and batch metadata, (2) fits a two-factor model, (3) compares results to a single-factor model, (4) generates PCA plots colored by batch and condition before and after batch inclusion, and (5) exports differential expression results. Deploy on AWS EC2 r6i.4xlarge.",

    "Describe how DESeq2 performs likelihood ratio testing (LRT) for multi-condition and time series comparisons, and when LRT is preferred over Wald tests. Write a Python script using pydeseq2 that: (1) loads a count matrix with 4 time points, (2) runs LRT with full model (~time) vs reduced model (~1), (3) identifies genes with significant time-dependent expression (padj < 0.05), (4) clusters these genes by expression trajectory using K-means on normalized counts, and (5) generates a line plot of mean expression per cluster over time. Deploy on AWS EC2 r6i.2xlarge.",

    "Explain how DESeq2 independent filtering increases statistical power — specifically how it uses mean normalized count as a filter criterion and re-applies BH correction only to the non-filtered gene set. Write a Python script using pydeseq2 that: (1) loads a count matrix, (2) runs DESeq2, (3) manually applies independent filtering at three mean-count thresholds (1, 5, 20), (4) compares the number of significant genes discovered at each threshold, and (5) generates a plot of significant gene count vs filtering threshold. Deploy on AWS EC2 r6i.xlarge.",

    "Describe how DESeq2 is integrated into a Snakemake workflow for bulk RNA-seq analysis starting from raw FASTQ — covering STAR alignment, featureCounts quantification, and DESeq2 differential expression. Write a Snakemake workflow defining rules for: (1) STAR two-pass alignment of paired-end FASTQ, (2) featureCounts quantification of aligned BAM, (3) merging count matrices across samples, and (4) running pydeseq2 with volcano plot output. Design an AWS Batch architecture with Spot Instances and FSx Lustre for fast shared storage to execute this workflow at scale.",

    # ══════════════════════════════════════════════════════════════════════════
    # WNN / SEURAT MULTIMODAL  (10 queries)
    # ══════════════════════════════════════════════════════════════════════════
    "Describe the Weighted Nearest Neighbor (WNN) multimodal integration method from the Seurat paper for joint analysis of RNA and protein (CITE-seq) data, and explain how per-cell modality weights capture which data type is more informative for each individual cell. Write a Python script using Scanpy and muon to perform WNN on a CITE-seq AnnData with RNA and surface protein modalities and visualize per-cell modality weights on UMAP. Design an AWS EC2 r6i.4xlarge architecture for multimodal single-cell analysis at scale.",

    "Explain the WNN algorithm's approach to computing per-cell weights by comparing within-modality and cross-modality kNN distances, and describe why cells in different biological states (cycling vs quiescent) may rely on different modalities for neighborhood definition. Write a Python script using Scanpy that: (1) loads a paired RNA+protein AnnData, (2) computes PCA on each modality, (3) builds a WNN joint graph, (4) clusters with Leiden at resolution 0.5, (5) visualizes per-cell RNA weight as a continuous gradient on UMAP, and (6) identifies clusters dominated by protein vs RNA signal. Deploy on AWS EC2 r6i.2xlarge.",

    "Describe how WNN is applied to paired RNA and chromatin accessibility (ATAC-seq) data — specifically how LSI (latent semantic indexing) is used for ATAC dimensionality reduction before WNN integration. Write a Python script using Scanpy and snapatac2 to: (1) load a paired 10X Multiome AnnData, (2) run PCA on RNA and LSI on ATAC, (3) build a WNN graph from both modalities, (4) cluster with Leiden, and (5) identify gene regulatory elements by linking peak accessibility to gene expression. Design an AWS r6i.8xlarge architecture with EFS for multi-modal reference genome data.",

    "Explain the theoretical basis for using WNN-derived joint embeddings for trajectory inference, and how the combined RNA and protein signal improves pseudotime estimation over single-modality methods. Write a Python script using Scanpy that: (1) builds a WNN graph on a paired RNA+protein AnnData, (2) computes UMAP from the WNN graph, (3) runs diffusion pseudotime (DPT) from a specified root cell, (4) plots pseudotime as a continuous color on WNN UMAP, and (5) computes correlation between RNA-only and WNN pseudotime. Deploy on AWS EC2 r6i.2xlarge.",

    "Describe the difference between WNN-based integration and simple feature concatenation for multimodal single-cell analysis — specifically why WNN produces better-resolved clusters for cells where one modality is more informative. Write a Python script using Scanpy that: (1) loads a CITE-seq AnnData, (2) runs Leiden clustering on RNA only, protein only, and WNN, (3) computes ARI between each pair of clusterings, and (4) generates three side-by-side UMAPs. Deploy on AWS EC2 r6i.4xlarge with outputs saved to S3.",

    "Explain how WNN is used for cell type annotation in CITE-seq experiments — specifically how the high specificity of surface protein markers combined with RNA transcriptome data enables accurate identification of rare cell populations. Write a Python script using Scanpy and muon that: (1) runs WNN on a PBMC CITE-seq dataset, (2) annotates clusters using known protein markers (CD3E, CD19, CD14, CD56), (3) validates annotations with RNA marker genes using rank_genes_groups(), and (4) generates a dot plot of protein and RNA markers per cell type. Design an AWS SageMaker JupyterLab environment for interactive annotation.",

    "Describe how batch correction interacts with WNN multimodal integration — whether batch correction should be applied per modality before WNN graph construction, and what tools (Harmony, scVI) are compatible with this workflow. Write a Python script using Scanpy and harmonypy that: (1) loads a CITE-seq AnnData with two batches, (2) applies Harmony correction separately on RNA and protein PCA embeddings, (3) builds a WNN graph from corrected embeddings, (4) clusters with Leiden, and (5) generates side-by-side UMAPs colored by batch before and after correction. Deploy on AWS EC2 r6i.4xlarge.",

    "Explain how WNN-based analysis scales computationally for datasets with millions of cells — what approximate nearest neighbor algorithms (HNSW) are used, and how Scanpy implements these for efficient kNN graph construction. Write a Python script using Scanpy that: (1) simulates a 100,000-cell paired RNA+protein AnnData, (2) benchmarks WNN graph construction time with exact vs approximate kNN (n_neighbors=30), (3) clusters with Leiden, and (4) measures wall-clock time and peak memory with tracemalloc. Design an AWS EC2 r6i.8xlarge instance with EBS gp3 for large in-memory operations.",

    "Describe the WNN paper's validation using CITE-seq data from 211 surface proteins and RNA — specifically what biological insights the per-cell modality weight analysis revealed about which cell types rely more on protein vs RNA signal. Write a Python script using Scanpy and muon that: (1) runs WNN integration on a PBMC CITE-seq dataset, (2) computes per-cell RNA vs protein weight, (3) generates a violin plot of RNA modality weight per annotated cell type, and (4) identifies cell types with bimodally distributed weights suggesting heterogeneous modality reliance. Deploy on AWS EC2 r6i.4xlarge.",

    "Explain how WNN integration is used in reference mapping to project new single-cell datasets onto a pre-computed multimodal reference atlas, and what computational advantages this approach offers over re-running WNN on each new dataset. Write a Python script using Scanpy and scArches that: (1) builds a WNN reference model on a PBMC CITE-seq dataset, (2) projects a new query dataset (RNA only) onto the reference WNN UMAP, (3) transfers cell type labels from reference to query using weighted kNN, and (4) computes label transfer confidence per cell. Design an AWS SageMaker multi-instance architecture for serving a reference atlas.",

    # ══════════════════════════════════════════════════════════════════════════
    # SAM / BAM / STAR  (4 queries)
    # ══════════════════════════════════════════════════════════════════════════
    "Explain how the SAM and BAM file formats store sequencing alignment data — including how CIGAR strings encode alignment operations (match, insertion, deletion, splice junction), how FLAG bits encode read properties, and what the NH, HI, AS, and NM optional tags record. Write a Python script using pysam to parse a STAR-aligned BAM file, compute per-read alignment statistics, and generate a summary JSON report. Design an AWS S3 intelligent-tiering storage architecture for managing BAM files from a 10,000-sample cohort.",

    "Describe the STAR RNA-seq aligner's two-pass alignment strategy: what novel splice junction information is collected in pass 1 (SJ.out.tab), how it is incorporated into the genome index for pass 2, and what the quantitative improvement in junction-spanning read alignment is from pass 1 to pass 2. Write a comprehensive shell script for a STAR two-pass pipeline on paired-end FASTQ with a Python QC script that parses Log.final.out and reports uniquely mapped read percentages for both passes. Deploy as an AWS Batch Spot Instance job with FASTQ from S3 and BAM output to S3.",

    "Explain how STAR handles chimeric read alignment for detecting structural variants and gene fusions — what the --chimSegmentMin parameter controls and what the Chimeric.out.junction file contains. Write a Python script that: (1) runs STAR alignment with --chimSegmentMin 20 via subprocess, (2) parses Chimeric.out.junction to extract fusion gene candidates by chromosome, breakpoint, and strand, (3) filters fusions with at least 5 supporting reads and breakpoints on different chromosomes or more than 100 kb apart, and (4) exports a fusion candidate BED file. Design an AWS Batch architecture for tumor RNA-seq fusion detection at scale.",

    "Explain what the SAM optional tags NH (number of hits), HI (hit index), AS (alignment score), and NM (edit distance) encode, and how these are used for downstream filtering to retain only uniquely mapped primary alignments. Write a Python script using pysam that: (1) reads a STAR-aligned BAM file, (2) extracts NH, HI, AS, and NM tags per read, (3) computes the distribution of each tag across all alignments, (4) filters reads to retain only primary alignments (NH=1), and (5) exports filtered reads to a new BAM. Design an AWS Lambda function that performs this tag-based filtering on BAM files in S3.",
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