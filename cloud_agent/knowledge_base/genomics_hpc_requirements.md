# HPC Hardware Guidelines: Genomics & DNA Sequencing

## Read Alignment (BWA, Bowtie2)
- **Compute Profile:** CPU-bound. These algorithms require massive parallelization across CPU cores to align millions of short reads.
- **Hardware Recommendation:** AWS C-Family (e.g., `c6i.8xlarge` with 32 vCPUs). 
- **Memory:** Requires moderate RAM (~4GB per core). Usually, 32GB-64GB total RAM is sufficient.
- **Storage:** High IOPS SSDs (AWS io2 Block Express) are required to prevent I/O bottlenecks when reading multi-gigabyte FASTQ files.

## Variant Calling (GATK HaplotypeCaller)
- **Compute Profile:** Memory-bound and highly iterative. 
- **Hardware Recommendation:** AWS R-Family (e.g., `r6i.4xlarge`).
- **Memory:** GATK Java processes require heavy RAM overhead. Minimum 64GB RAM recommended.
- **OS & Frameworks:** Ubuntu 22.04 LTS. Requires Java 8/11. No GPUs required.