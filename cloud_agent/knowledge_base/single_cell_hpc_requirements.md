# HPC Hardware Guidelines: Single-Cell Transcriptomics

## Preprocessing & Alignment (CellRanger, STAR)
- **Compute Profile:** Highly memory-bound. Spliced alignment requires loading massive reference genomes into RAM.
- **Hardware Recommendation:** AWS R-Family or X-Family (e.g., `r6a.16xlarge` or `x2idn.16xlarge`).
- **Memory:** Absolute minimum of 64GB RAM. For large mammalian references, 128GB to 256GB RAM is strongly recommended.
- **Storage:** Standard General Purpose SSDs (gp3).

## Clustering & Trajectory Inference (Scanpy, Seurat)
- **Compute Profile:** Memory-bound during neighborhood graph construction (kNN) and UMAP dimensionality reduction.
- **Hardware Recommendation:** Memory-optimized instances (`r6i.4xlarge`).
- **Memory:** Varies by cell count. 100k cells require ~64GB RAM. 1M+ cells require ~256GB RAM.
- **OS & Frameworks:** Ubuntu 22.04. Python 3.10+ with `scanpy`, `anndata`, `umap-learn`, and `leidenalg`.