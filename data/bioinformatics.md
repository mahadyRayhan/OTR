# Introduction to Bioinformatics

## Overview
Bioinformatics is an interdisciplinary field that develops methods and software tools for understanding biological data, especially when the data sets are large and complex. It combines computer science, statistics, mathematics, and engineering to analyze and interpret biological data.

## The Central Dogma
The central dogma of molecular biology explains the flow of genetic information: DNA is transcribed into RNA, and RNA is translated into proteins. Translation reads RNA in triplets (codons) to assemble amino acids. Three specific codons (TAA, TAG, TGA in DNA) are 'stop codons' that terminate translation.

## Sequence Alignment Algorithms
Alignment is the process of matching sequences to identify regions of similarity. 
* **Global Alignment (Needleman-Wunsch):** Forces the alignment to span the entire length of both sequences end-to-end. Best for highly similar sequences of similar length.
* **Local Alignment (Smith-Waterman):** Finds the most similar subsequences within the larger sequences. Best for finding conserved domains across divergent sequences of different lengths.

## Short-Read Alignment Tools
When analyzing Next-Generation Sequencing (NGS) data, aligners match reads to a reference genome.
* **BWA and Bowtie2:** Standard aligners. BWA handles longer gaps, Bowtie2 is highly memory-efficient.
* **Splice-Aware Alignment (STAR, HISAT2):** Required for RNA-seq data because RNA transcripts are spliced (introns are removed). Standard aligners fail at these large splice junctions, so STAR and HISAT2 are explicitly designed to map reads across them.
* **Pseudoalignment (Kallisto, Salmon):** Rather than aligning base-by-base, pseudoalignment determines which transcripts a read is compatible with using k-mers. It is orders of magnitude faster than traditional alignment.

## File Formats
Bioinformatics relies heavily on standardized file formats:
* **FASTA:** Stores only sequence strings and a header starting with '>'. Contains no quality data.
* **FASTQ:** The standard text file format for storing unaligned raw sequencing reads. It stores both the sequence and Phred quality scores. Commonly parsed with Biopython's SeqIO.
* **SAM/BAM:** Stores aligned reads. BAM is the compressed binary version of SAM. You cannot easily convert SAM back to FASTQ if the unaligned/unmapped reads were filtered out during alignment.
* **VCF (Variant Call Format):** The standard file format for storing genetic variants (SNPs, indels). The VCF header contains crucial metadata about the reference genome and filters applied. The 'pysam' library is widely used to parse it.

## Advanced Biological Workflows
* **GATK Best Practices:** A standard pipeline for variant calling. Stages include Alignment (BWA), MarkDuplicates, BQSR, HaplotypeCaller, and VariantFiltration.
* **Protein Structure Prediction:** AlphaFold 3 (AI-based, highly accurate but resource-intensive) and Rosetta (physics-based, highly customizable for *de novo* design).
* **Single-Cell RNA-seq (scRNA-seq):** Analyzes gene expression at the individual cell level. The standard analysis workflow uses Python's Scanpy library and the AnnData object format. Clustering cells typically relies on the Louvain or Leiden community-detection algorithms.
* **Spatial Transcriptomics:** Maps gene expression directly onto tissue slides. The primary computational bottleneck is memory: spatial data matrixes are massive and require hundreds of gigabytes of RAM to process high-resolution images alongside transcriptome data.
* **CRISPR Screening:** Uses Cas9 to knock out genes to study their function. Experiments generate terabytes of sequence data requiring heavy cloud infrastructure to process.

## Pipeline and Cloud Infrastructure
* **Workflow Managers:** Nextflow and Snakemake orchestrate complex pipelines (like running FastQC and Trimmomatic) to ensure reproducibility.
* **Cloud Deployment:** Bioinformatics pipelines are frequently deployed to AWS. AWS Batch manages containerized jobs, S3 provides durable data storage, and EC2 provides compute. "Fargate" is a serverless compute engine introduced by AWS that allows running containers without managing servers.