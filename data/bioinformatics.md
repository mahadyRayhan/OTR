# Introduction to Bioinformatics

## Overview
Bioinformatics develops methods and software tools for understanding complex biological data. It combines computer science, statistics, mathematics, and engineering.

## The Central Dogma
The central dogma of molecular biology dictates the flow of genetic information: DNA is transcribed into RNA, and RNA is translated into proteins. Translation reads RNA in triplets called codons. Three specific codons (TAA, TAG, TGA in DNA) are 'stop codons' that terminate translation.

## Sequence Alignment Algorithms
Alignment identifies regions of similarity between sequences.
* **Global Alignment (Needleman-Wunsch):** Forces end-to-end alignment across the entire length of both sequences. Uses a scoring matrix with gap penalties.
* **Local Alignment (Smith-Waterman):** Finds the most similar subsequences within larger sequences. Negative scores are reset to zero in the scoring matrix, allowing the algorithm to ignore poorly matched regions.
* **The Burrows-Wheeler Transform (BWT):** A data compression algorithm heavily used in alignment (like BWA). It rearranges a character string into runs of similar characters, making it highly compressible and allowing for extremely fast, memory-efficient string matching.

## Short-Read Alignment Tools
* **BWA and Bowtie2:** Standard aligners. BWA handles longer gaps well. Bowtie2 is highly memory-efficient.
* **Splice-Aware Alignment (STAR, HISAT2):** Required for RNA-seq. Because RNA transcripts have introns removed, standard aligners fail at these large splice junctions. STAR and HISAT2 are designed to map reads across these gaps.
* **Pseudoalignment (Kallisto, Salmon):** Rather than aligning base-by-base, pseudoalignment uses k-mers to determine which transcripts a read is compatible with. It is orders of magnitude faster than traditional alignment.

## File Formats
* **FASTA:** Stores only sequence strings and a header starting with '>'. Contains no quality scores.
* **FASTQ:** The standard text format for unaligned sequencing reads. Stores both the sequence and Phred quality scores. Parsed commonly with Biopython's `SeqIO` module.
* **SAM/BAM:** Stores aligned reads. BAM is compressed binary. You cannot convert SAM back to FASTQ if unmapped reads were discarded during alignment.
* **VCF (Variant Call Format):** The standard for storing genetic variants (SNPs, indels). The VCF header (lines starting with '##') contains crucial metadata about the reference genome, contigs, and filters applied. The 'pysam' library parses VCF files efficiently.

## Advanced Workflows & Tools
* **GATK Best Practices:** A standard pipeline for germline variant calling. Stages include BWA alignment, MarkDuplicates, BQSR (Base Quality Score Recalibration), HaplotypeCaller, and VariantFiltration.
* **Protein Structure Prediction:** AlphaFold 3 (AI-based, exceptionally accurate but requires heavy GPU compute) vs. Rosetta (physics-based, highly customizable for de novo design but slower).
* **Single-Cell RNA-seq (scRNA-seq):** Analyzes gene expression per cell. Standard workflows use Python's `Scanpy` library and the `AnnData` object format. Cells are typically clustered using the Louvain or Leiden community-detection algorithms.
* **Spatial Transcriptomics:** Maps gene expression onto tissue slides. The primary computational bottleneck is memory (RAM), as spatial matrices combining high-resolution images and transcriptome data often require hundreds of gigabytes of memory to process.
* **CRISPR Screening:** Uses Cas9 to knock out genes. Experiments generate terabytes of sequence data, requiring heavy cloud infrastructure for processing.

## Pipeline and Cloud Infrastructure
* **Workflow Managers:** Nextflow and Snakemake orchestrate complex pipelines (like trimming reads with FastQC and Trimmomatic) to ensure reproducibility.
* **Cloud Deployment:** Bioinformatics pipelines are frequently deployed to AWS. AWS Batch manages containerized jobs, Amazon S3 provides durable data storage, and Amazon EC2 provides compute. Amazon EFS is often used to solve memory bottlenecks by providing shared file systems.
* **AWS Fargate:** A serverless compute engine introduced by AWS that allows running containers without provisioning or managing EC2 servers.