# Sourced resource requirements for scientific tools

Every statement below is quoted verbatim from the linked official source, retrieved 2026-11-04.
Use these before extrapolating. nf-core `base.config` labels (process_low/medium/high) are the defaults the pipeline requests for each step.

## nf-core/rnaseq (bioinformatics)

Source: resource labels used by every process (official pipeline config), https://raw.githubusercontent.com/nf-core/rnaseq/master/conf/base.config

```text
    cpus   = { 1      * task.attempt }
    memory = { 6.GB   * task.attempt }
    time   = { 4.h    * task.attempt }
    withLabel:process_single {
        cpus   = { 1                   }
        memory = { 6.GB * task.attempt }
        time   = { 4.h  * task.attempt }
    withLabel:process_low {
        cpus   = { 2     * task.attempt }
        memory = { 12.GB * task.attempt }
        time   = { 4.h   * task.attempt }
    withLabel:process_medium {
        cpus   = { 6     * task.attempt }
        memory = { 36.GB * task.attempt }
        time   = { 8.h   * task.attempt }
    withLabel:process_high {
        cpus   = { 12    * task.attempt }
        memory = { 72.GB * task.attempt }
        time   = { 16.h  * task.attempt }
    withLabel:process_long {
        time   = { 20.h  * task.attempt }
    withLabel:process_low_memory {
        memory = { 1.GB * task.attempt }
    withLabel:process_high_memory {
        memory = { 200.GB * task.attempt }
    withLabel: process_gpu {
        containerOptions = { params.gpu_container_options ?: (workflow.containerEngine in ['singularity', 'apptainer'] ? '--nv' : '--gpus all') }
```

## nf-core/rnaseq (bioinformatics)

Source: official usage documentation, https://raw.githubusercontent.com/nf-core/rnaseq/master/docs/usage.md

```text
| `smr_v4.3_fast_db`                 | SILVA 85/90 %, Rfam 90 % | ~17 MB     | Pipeline default. Routine bulk RNA-seq with good-quality input                    |
| `smr_v4.3_default_db`              | SILVA 90/95 %, Rfam 95 % | ~36 MB     | SortMeRNA upstream's balanced recommendation; moderate extra index footprint      |
| `smr_v4.3_sensitive_db`            | SILVA 97 %, Rfam 97 %    | ~93 MB     | High-rRNA samples (e.g. total RNA without rRNA depletion), rare-lineage organisms |
| `smr_v4.3_sensitive_db_rfam_seeds` | SILVA 97 % + Rfam seeds  | ~90 MB     | Maximum sensitivity, including the full Rfam seed set for non-rRNA ncRNA families |
Picking a variant: `fast` keeps the index footprint close to what earlier pipeline versions used (~1-2 GB); `sensitive` and `sensitive_rfam_seeds` roughly triple that. Index size is a one-off cost per work directory (results are cached), but per-sample SortMeRNA memory and runtime also scale with the reference.
#### GPU acceleration for RiboDetector
RiboDetector supports GPU acceleration via `--use_gpu_ribodetector`. When enabled, the pipeline uses a CUDA-enabled container with PyTorch GPU support and switches to the GPU binary automatically.
nextflow run nf-core/rnaseq --remove_ribo_rna --ribo_removal_tool ribodetector --use_gpu_ribodetector ...
- One or more NVIDIA GPUs (with appropriate drivers installed)
- x86_64 architecture (ARM GPU is not currently supported)
Container GPU flags (`--gpus all` for Docker, `--nv` for Singularity/Apptainer) are automatically applied and can be overridden via `--gpu_container_options`.
### Parabricks GPU acceleration for STAR
The STAR aligner can also be GPU-accelerated using NVIDIA Parabricks via the `--use_parabricks_star` parameter. Parabricks runs STAR alignment on NVIDIA GPUs, significantly reducing wall-clock time for large datasets.
Container GPU flags (`--gpus all` for Docker, `--nv` for Singularity/Apptainer) are automatically applied to GPU tasks based on the container engine. These flags are scoped to GPU tasks only, so non-GPU steps will run normally on CPU-only nodes in mixed clusters.
The container GPU flags can be overridden via `--gpu_container_options` (e.g. `--gpu_container_options '--gpus 1'`).
- One or more NVIDIA GPUs (with appropriate drivers installed)
The default resource configuration (`accelerator = 1` via the `process_gpu` label) is sized for small test datasets. Full-size genomes (e.g. GRCh37/GRCh38) require more GPUs, memory, and the `--low-memory` STAR flag to reduce GPU memory pressure.
The following config was tested with the `test_full` dataset (GRCh37, 8 paired-end samples) on AWS g5 instances (NVIDIA A10G, 24GB VRAM per GPU). Adjust for your cloud provider and GPU type:
        cpus           = { 48 * task.attempt }
        memory         = { task.attempt > 1 ? 370.GB : 186.GB }
        resourceLimits = [cpus: 96, memory: 370.GB]
And pass the low-memory flag:
--extra_star_align_args '--low-memory'
- **Multiple GPUs are needed** for full-size genomes. 4 GPUs with `--low-memory` was sufficient for GRCh37 on A10G (24GB VRAM). GPUs with more VRAM (e.g. A100 40GB/80GB) may work with fewer GPUs or without `--low-memory`.
- **Request all GPUs on retry** to prevent multiple Parabricks tasks being co-scheduled on the same node. GPU memory is not managed by the scheduler, so two 4-GPU tasks on an 8-GPU node will contend for VRAM and fail.
- **Parabricks returns exit code 255 for most failures** including GPU OOM, host OOM, and internal errors. The retry strategy above handles this, but be aware that not all 255 exits are resource-related.
- **Ensure your compute environment includes GPU instance types large enough for the retry**. For example, on AWS you might include both `g5.12xlarge` (4 GPUs, 192GB RAM) and `g5.48xlarge` (8 GPUs, 768GB RAM).
- **Memory**: aggregation steps (tximport `.collect()`, SummarizedExperiment, RSEM merge counts, merged MultiQC) load all samples into a single process, which can exceed available memory
Genomes with very large chromosomes (>500 Mb), such as plant genomes, may encounter failures in the RSeQC `inner_distance` module due to a known limitation in the underlying bx-python library. The bx-python BitSet implementation has a maximum capacity of approximately 537 million bases, which can be exceeded by chromosomes in organisms like wheat, barley, and other plants.
## Nextflow memory requirements
In some cases, the Nextflow Java virtual machines can start to request a large amount of memory.
```

## nf-core/sarek (variant calling, GATK) (bioinformatics)

Source: resource labels used by every process (official pipeline config), https://raw.githubusercontent.com/nf-core/sarek/master/conf/base.config

```text
    cpus   = { 1      * task.attempt }
    memory = { 6.GB   * task.attempt }
    time   = { 8.h    * task.attempt }
    withLabel:process_single {
        cpus   = { 1                   }
        memory = { 6.GB * task.attempt }
        time   = { 8.h  * task.attempt }
    withLabel:process_low {
        cpus   = { 2     * task.attempt }
        memory = { 12.GB * task.attempt }
        time   = { 8.h   * task.attempt }
    withLabel:process_medium {
        cpus   = { 6     * task.attempt }
        memory = { 36.GB * task.attempt }
        time   = { 16.h  * task.attempt }
    withLabel:process_high {
        cpus   = { 12    * task.attempt }
        memory = { 72.GB * task.attempt }
        time   = { 32.h  * task.attempt }
    withLabel:process_long {
        time   = { 40.h  * task.attempt }
    withLabel:process_low_memory {
        memory = { 1.GB * task.attempt }
    withLabel:process_high_memory {
        memory = { 200.GB * task.attempt }
    withLabel: process_gpu {
        accelerator = { workflow.profile.contains('gpu') ? 1 : null }
    withName: 'UNZIP.*|UNTAR.*|TABIX.*|HTSLIB_BGZIPTABIX|BUILD_INTERVALS|CREATE_INTERVALS_BED|VCFTOOLS|BCFTOOLS.*|SAMTOOLS_INDEX' {
        cpus   = { 1    * task.attempt }
        memory = { 1.GB * task.attempt }
    withName: 'FASTQC'{
        cpus   = { 4    * task.attempt }
        memory = { 4.GB * task.attempt }
    withName: 'FASTP'{
        cpus   = { 12   * task.attempt }
        memory = { 4.GB * task.attempt }
    withName: 'BWAMEM1_MEM|BWAMEM2_MEM' {
        cpus   = { 24    * task.attempt }
        memory = { 30.GB * task.attempt }
    withName:'CNVKIT_BATCH' {
        cpus   = { 12    * task.attempt }
        memory = { 36.GB * task.attempt }
    withName: 'GATK4_MARKDUPLICATES|GATK4SPARK_MARKDUPLICATES' {
        cpus   = { 6     * task.attempt }
        memory = { 30.GB * task.attempt }
    withName:'GATK4_APPLYBQSR|GATK4SPARK_APPLYBQSR|GATK4_BASERECALIBRATOR|GATK4SPARK_BASERECALIBRATOR|GATK4_GATHERBQSRREPORTS'{
        cpus   = { 2    * task.attempt }
        memory = { 4.GB * task.attempt }
    withName:'MOSDEPTH'{
        cpus   = { 4    * task.attempt }
        memory = { 4.GB * task.attempt }
    withName:'STRELKA.*|MANTA.*' {
        cpus   = { 10   * task.attempt }
        memory = { 8.GB * task.attempt }
    withName:'SAMTOOLS_CONVERT'{
        memory = { 4.GB * task.attempt }
    withName:'GATK4_MERGEVCFS'{
        cpus   = { 2    * task.attempt }
        memory = { 4.GB * task.attempt }
    withName: 'MULTIQC' {
```

## nf-core/scrnaseq (bioinformatics)

Source: resource labels used by every process (official pipeline config), https://raw.githubusercontent.com/nf-core/scrnaseq/master/conf/base.config

```text
    cpus   = { 1      * task.attempt }
    memory = { 6.GB   * task.attempt }
    time   = { 4.h    * task.attempt }
    withLabel:process_single {
        cpus   = { 1                   }
        memory = { 6.GB * task.attempt }
        time   = { 4.h  * task.attempt }
    withLabel:process_low {
        cpus   = { 2     * task.attempt }
        memory = { 12.GB * task.attempt }
        time   = { 4.h   * task.attempt }
    withLabel:process_medium {
        cpus   = { 6     * task.attempt }
        memory = { 36.GB * task.attempt }
        time   = { 8.h   * task.attempt }
    withLabel:process_high {
        cpus   = { 12    * task.attempt }
        memory = { 72.GB * task.attempt }
        time   = { 16.h  * task.attempt }
    withLabel:process_long {
        time   = { 20.h  * task.attempt }
    withLabel:process_high_memory {
        memory = { 200.GB * task.attempt }
    withLabel: process_gpu {
        ext.use_gpu = { workflow.profile.contains('gpu') }
        accelerator = { workflow.profile.contains('gpu') ? 1 : null }
```

## nf-core/atacseq (bioinformatics)

Source: resource labels used by every process (official pipeline config), https://raw.githubusercontent.com/nf-core/atacseq/master/conf/base.config

```text
    cpus   = { check_max( 1    * task.attempt, 'cpus'   ) }
    memory = { check_max( 6.GB * task.attempt, 'memory' ) }
    time   = { check_max( 4.h  * task.attempt, 'time'   ) }
    withLabel:process_single {
        cpus   = { check_max( 1                  , 'cpus'    ) }
        memory = { check_max( 6.GB * task.attempt, 'memory'  ) }
        time   = { check_max( 4.h  * task.attempt, 'time'    ) }
    withLabel:process_low {
        cpus   = { check_max( 2     * task.attempt, 'cpus'    ) }
        memory = { check_max( 12.GB * task.attempt, 'memory'  ) }
        time   = { check_max( 4.h   * task.attempt, 'time'    ) }
    withLabel:process_medium {
        cpus   = { check_max( 6     * task.attempt, 'cpus'    ) }
        memory = { check_max( 36.GB * task.attempt, 'memory'  ) }
        time   = { check_max( 8.h   * task.attempt, 'time'    ) }
    withLabel:process_high {
        cpus   = { check_max( 12    * task.attempt, 'cpus'    ) }
        memory = { check_max( 72.GB * task.attempt, 'memory'  ) }
        time   = { check_max( 16.h  * task.attempt, 'time'    ) }
    withLabel:process_long {
        time   = { check_max( 20.h  * task.attempt, 'time'    ) }
    withLabel:process_high_memory {
        memory = { check_max( 200.GB * task.attempt, 'memory' ) }
```

## nf-core/mag (metagenomics) (bioinformatics)

Source: resource labels used by every process (official pipeline config), https://raw.githubusercontent.com/nf-core/mag/master/conf/base.config

```text
    cpus          = { 1 * task.attempt }
    memory        = { 7.GB * task.attempt }
    time          = { 4.h * task.attempt }
    withLabel: process_single {
        cpus   = { 1 }
        memory = { 6.GB * task.attempt }
        time   = { 4.h * task.attempt }
    withLabel: process_low {
        cpus   = { 2 * task.attempt }
        memory = { 12.GB * task.attempt }
        time   = { 4.h * task.attempt }
    withLabel: process_medium {
        cpus   = { 6 * task.attempt }
        memory = { 36.GB * task.attempt }
        time   = { 8.h * task.attempt }
    withLabel: process_high {
        cpus   = { 12 * task.attempt }
        memory = { 72.GB * task.attempt }
        time   = { 16.h * task.attempt }
    withLabel: process_long {
        time = { 20.h * task.attempt }
    withLabel: process_high_memory {
        memory = { 200.GB * task.attempt }
    withLabel: process_gpu {
        ext.use_gpu = { workflow.profile.contains('gpu') }
        accelerator = { workflow.profile.contains('gpu') ? 1 : null }
    withName: ALE {
        memory = { params.coassemble_group ? 30.GB * task.attempt : 18.GB * task.attempt }
    //returns exit code 247 when running out of memory
    withName: BOWTIE2_ASSEMBLY_ALIGN {
        cpus          = 8
        memory        = { 12.GB * task.attempt }
        time          = { 8.h * task.attempt }
    withName: BOWTIE2_HOST_REMOVAL_ALIGN {
        cpus   = { 8 * task.attempt }
        memory = { 10.GB * task.attempt }
        time   = { 6.h * task.attempt }
    withName: BOWTIE2_HOST_REMOVAL_BUILD {
        cpus   = { 8 * task.attempt }
        memory = { 20.GB * task.attempt }
        time   = { 4.h * task.attempt }
    withName: BOWTIE2_PHIX_REMOVAL_ALIGN {
        cpus   = { 8 * task.attempt }
        memory = { 2.GB * task.attempt }
        time   = { 6.h * task.attempt }
    withName: BUSCO_BUSCO {
        cpus          = { 6 * task.attempt }
        memory        = { 16.GB * task.attempt }
        time          = { 12.h * task.attempt }
    withName: CATPACK_BINS {
        memory = { 120.GB * task.attempt }
    withName: CATPACK_CONTIGS {
        memory = { 60.GB * task.attempt }
    withName: CHECKM_LINEAGEWF {
        memory = 42.GB
    withName: CHOPPER {
        cpus   = { 2 * task.attempt }
        memory = { 4.GB * task.attempt }
    withName: 'CONCAT_BUSCO_TSV|CONCAT_CHECKM_TSV|CONCAT_CHECKM2_TSV|CONCAT_GUNC_TSV|CONCAT_GUNC_CHECKM_TSV' {
        cpus   = 1
```

## AlphaFold 2 (bioinformatics)

Source: official README, https://raw.githubusercontent.com/google-deepmind/alphafold/main/README.md

```text
operating systems. Full installation requires up to 3 TB of disk space to keep
genetic databases (SSD storage is recommended) and a modern NVIDIA GPU (GPUs
with more memory can predict larger protein structures).
        for GPU support.
1.  Check that AlphaFold will be able to use a GPU by running:
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    The output of this command should show a list of your GPUs. If it doesn't,
    W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
    E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease' is not signed.
:ledger: **Note: The total download size for the full databases is around 556 GB
and the total size when unzipped is 2.62 TB. Please make sure you have a large
$DOWNLOAD_DIR/                             # Total: ~ 2.62 TB (download: 556 GB)
    bfd/                                   # ~ 1.8 TB (download: 271.6 GB)
    mgnify/                                # ~ 120 GB (download: 67 GB)
    params/                                # ~ 5.3 GB (download: 5.3 GB)
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
    pdb_mmcif/                             # ~ 238 GB (download: 43 GB)
    pdb_seqres/                            # ~ 0.2 GB (download: 0.2 GB)
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
    uniref30/                              # ~ 206 GB (download: 52.5 GB)
    uniprot/                               # ~ 105 GB (download: 53 GB)
    uniref90/                              # ~ 67 GB (download: 34 GB)
was tested on Google Cloud with a machine using the `nvidia-gpu-cloud-image`
with 12 vCPUs, 85 GB of RAM, a 100 GB boot disk, the databases on an additional
3 TB disk, and an A100 GPU. For your first run, please follow the instructions
1.  By default, Alphafold will attempt to use all visible GPU devices. To use a
    subset, specify a comma-separated list of GPU UUID(s) or index(es) using the
    `--gpu_devices` flag. See
    [GPU enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration)
        requires 8 CPU cores (vCPUs), 8 GB of RAM, and 600 GB of disk space.
1.  The relaxation step can be run on GPU (faster, but could be less stable) or
    `--enable_gpu_relax=true` (default) or `--enable_gpu_relax=false`.
from a single A100 NVIDIA GPU. Prediction speed on A100 for smaller structures
    *   Distograms (`distogram/logits` contains a NumPy array of shape [N_res,
        N_res, N_bins] and `distogram/bin_edges` contains the definition of the
```

## ColabFold (bioinformatics)

Source: official README, https://raw.githubusercontent.com/sokrypton/ColabFold/main/README.md

```text
  - Limits depends on free GPU provided by Google Colab `fingers-crossed`
  - For GPUs with ~16G the max length is ~2000
# With CUDA GPU support
pip install colabfold[alphafold,openmm] jax[cuda12] openmm[cuda12]
# CUDA 13 is recommended if you have a Blackwell GPU or newer
pip install colabfold[alphafold,openmm] jax[cuda13] openmm[cuda13]
docker pull ghcr.io/sokrypton/colabfold:1.6.3-cuda12
When you pass a FASTA or CSV file containing your sequences to `colabfold_batch` it will automatically query the public MSA server to generate MSAs. You might want to split this into two steps for better GPU resource utilization:
# Query the MSA server and predict the structure on local GPU in one go:
# Split querying MSA server and GPU predictions into two steps
First create a directory for the databases on a disk with sufficient storage (940GB (!)). Depending on where you are, this will take a couple of hours:
# This needs a GPU
### GPU-accelerated search with ⁠`colabfold_search` ⁠
ColabFold supports GPU-accelerated MSA searches through [MMseqs2-GPU](https://www.biorxiv.org/content/10.1101/2024.11.13.623350v1).
#### GPU database setup
To setup the GPU databases, you will need to run the ⁠`setup_databases.sh`⁠ command with ⁠`GPU=1`⁠ as an environment variable:
GPU=1 ./setup_databases.sh /path/to/db_folder
This will download and setup the GPU databases in the specified folder. Note that here we do not pass ⁠`MMSEQS_NO_INDEX=1`⁠ as an argument since the indices are useful in the GPU search since we will keep them in the GPU memory.
#### GPU search
By default, running `colabfold_search` with the `--gpu 1` option uses all available GPUs for its search.
colabfold_search /path/to/bin/mmseqs input_sequences.fasta /path/to/db_folder msas --gpu 1
To select specific GPUs, set the `CUDA_VISIBLE_DEVICES` environment variable:
CUDA_VISIBLE_DEVICES=0,1 colabfold_search --mmseqs /path/to/bin/mmseqs input_sequences.fasta /path/to/db_folder msas --gpu 1
#### Optional GPU server for enhanced performance:
For frequent searches or to achieve minimal latency, you can run a dedicated GPU server. This server holds databases permanently in GPU memory, largely eliminating search overhead:
Start the GPU server(s):
mmseqs gpuserver /path/to/db_folder/colabfold_envdb_202108_db --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
mmseqs gpuserver /path/to/db_folder/uniref30_2302_db --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
By default, the GPU server distributes the database evenly across all visible GPUs. You can limit GPU usage by setting the CUDA_VISIBLE_DEVICES environment variable (e.g., `CUDA_VISIBLE_DEVICES=0,1`).
Important: Ensure that the `CUDA_VISIBLE_DEVICES` environment variable is set consistently for both `gpuserver` and `colabfold_search`, otherwise `colabfold_search` will try wait for the `gpuserver` to appear until a set timeout (by default 5 minutes). If your database exceeds GPU memory capacity, the GPU server efficiently streams data between host and GPU memory using asynchronous CUDA streams.
Run searches using the GPU server:
    --gpu 1 --gpu-server 1 --db-load-mode 2
For more details, see [GPU-accelerated search](https://github.com/soedinglab/MMseqs2/wiki#gpu-accelerated-search).
The saved model is reused only when the next run matches the previous one, including the GPU, driver, and JAX versions, and the same sequence length and MSA depth.
Add `--use-fast-kernels` to `colabfold_batch` to run AlphaFold2 with accelerated kernels, about 2.5x faster and with slightly lower memory. `--kernel-backend` picks the implementation based on the available GPU and defaults to `auto`:
- `pallas` on Ampere or newer generation GPUs, using Pallas/Triton kernels.
- `cuda_legacy` on older Volta and Turing GPUs.
```

## BWA-MEM2 (bioinformatics)

Source: official README, https://raw.githubusercontent.com/bwa-mem2/bwa-mem2/master/README.md

```text
***We are happy to announce that the index size on disk is down by 8 times and in memory by 4 times due to moving to only one type of FM-index (2bit.64 instead of 2bit.64 and 8bit.32) and 8x compression of suffix array. For example, for human genome, index size on disk is down to ~10GB from ~80GB and memory footprint is down to ~10GB from ~40GB.***
# Indexing the reference sequence (Requires 28N GB memory where N is the size of the reference sequence).
Processor: Intel(R) Xeon(R) 8280 CPU @ 2.70GHz
Memory: 100GB
For example,  in our double socket (56 threads each) and double numa compute node, we used the following command line to align D2 to human_g1k_v37.fasta reference genome.
4. The memory footprint of bwa-mem2-lisa index is ~120GB for human genome.
5. The memory foot print of the ert index is ~60GB
```

## minimap2 (bioinformatics)

Source: official README, https://raw.githubusercontent.com/lh3/minimap2/master/README.md

```text
Minimap2 is a versatile sequence alignment program that aligns DNA or mRNA
Minimap2 is optimized for x86-64 CPUs. You can acquire precompiled binaries from
operations (SAM and CRAM work). However, for ultra-long nanopore reads minimap2
you convert such SAM/CRAM to BAM, Picard and recent samtools will throw an
1. Read **-I** [=*4G*] reference bases, extract (**-k**,**-w**)-minimizers and
```

## Kraken 2 (bioinformatics)

Source: official manual, https://raw.githubusercontent.com/DerrickWood/kraken2/master/docs/MANUAL.markdown

```text
$k$-mer/LCA pairs as its database.  While fast, the large memory
structure, Kraken 2 is able to achieve faster speeds and lower memory
* **Disk space**: Construction of a Kraken 2 standard database requires
    approximately 100 GB of disk space.  A test on 01 Jan 2018 of the
    default installation showed 42 GB of disk space was used to store
    the genomic library files, 26 GB was used to store the taxonomy
    information from NCBI, and 29 GB was used to store the Kraken 2
* **Memory**: To run efficiently, Kraken 2 requires enough free memory
    to hold the database (primarily the hash table) in RAM.  While this
    the database into process-local RAM; the `--memory-mapping` switch
    to `kraken2`  will avoid doing so.  The default database size is 29 GB
    RAM if you want to build the default database.
* **MiniKraken**: At present, users with low-memory computing environments
and setup your Kraken 2 program directory.  Installation is successful if
The `--max-db-size` option supports sizes as either integers or units of measurement such "10GiB", "4TB", "10 gebibytes", "4 terabytes".
                  [--output FILENAME] [--memory-mapping]
  --memory-mapping      Avoids loading entire database into RAM
Adds support of the `--memory-mapping` and `--threads` options which allows for faster inspection of large indexes.
                   [--report-minimizer-data] [--memory-mapping]
  --use-daemon          Spawn a background process that keeps any loaded indexes in memory. Subsequent invokations of classify with
  --memory-mapping      Avoids loading entire database into RAM
Please note that the database will use approximately 100 GB of
disk space during creation, with the majority of that being reference
Using 32 threads on an AWS EC2 r4.8xlarge instance with 16 dual-core
hyperthreaded 2.30 GHz CPUs and 244 GB of RAM, the build process took
     This uses a low-memory method to reliably estimate the number of
likely because $k$ needs to be increased (reducing the overall memory
The `kraken2` program allows several different options:
to the well-known BLASTX program.  To do this, Kraken 2 uses a reduced
```

## DeepVariant (bioinformatics)

Source: official documentation, https://raw.githubusercontent.com/google/deepvariant/r1.6/docs/deepvariant-details.md

```text
`postprocess_variants`. More details about each program are described in detail
`make_examples` is a single-threaded program using 1-2 GB of RAM. Since the
the program will be 10% of the regions and the output will be written to
*   `--regions chr20:10,000,000-11,000,000` => only process 10-11mb of chr20
`call_variants` uses around 4 GB per process and uses TensorFlow for evaluation.
but scaling is sub-linear. In other words, `call_variants` on a 64 core machine
is less than 8x faster than running on a 8 core machine.
When using a GPU, `call_variants` is both faster, more efficient, and needs
configuration for `call_variants` on a GPU instance is 4-8 CPUs and 1 GPU.
speedup on the call_variants step using a single P100 GPU and 8 CPUs. Note that
currently `call_variants` can only use one GPU at most. So it doesn't improve
the speed if you get a multiple-GPU machine.
non-trivial amount of memory to run (20-30 GB), so it is best run on a
single/dual core machine with sufficient memory.
*   Vastly faster with reduced memory usage
## CRAM support
As of v0.7, DeepVariant accepts CRAM files as input in addition to BAM files.
`--ref` flag, instead of the path to the original reference in the CRAM file
For more information about CRAM, see the
[reference sequences in CRAM](http://www.htslib.org/doc/samtools.html#REFERENCE_SEQUENCES).
[benchmarking CRAM](http://www.htslib.org/benchmarks/CRAM.html) with information
on the effect of different CRAM options on file size and encoding/decoding
`make_examples` job on a 30x whole genome sample in BAM and CRAM format.
CRAM     | 37.85     | 96m53.477s
*   CRAM file:
    `gs://deepvariant/performance-testdata/HG002_NIST_150bp_downsampled_30x.cram`
We used a 64-core (vCPU) machine with 240GiB of memory and no GPU, on the Google
### Command for a GPU machine on Google Cloud Platform
gcloud compute instances create "${USER}-gpu" \
```

## Salmon (bioinformatics)

Source: official README, https://raw.githubusercontent.com/COMBINE-lab/salmon/master/README.md

```text
`salmon` is a **wicked**-fast program for highly-accurate, transcript-level
```

## nnU-Net (health_science)

Source: official installation documentation, https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/documentation/installation_instructions.md

```text
We support GPU (recommended), CPU and Apple M1/M2 as devices (currently Apple mps does not implement 3D
We recommend you use a GPU for training as this will take a really long time on CPU or MPS (Apple M1/M2).
For training a GPU with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080/3090 or RTX 4080/4090) is
required. We also recommend a strong CPU to go along with the GPU. 6 cores (12 threads)
input channels and target structures. Plus, the faster the GPU, the better the CPU should be!
Again we recommend a GPU to make predictions as this will be substantially faster than the other options. However,
inference times are typically still manageable on CPU and MPS (Apple M1/M2). If using a GPU, it should have at least
4 GB of available (unused) VRAM.
- GPU: RTX 3090 or RTX 4090
- RAM: 64GB
- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for fast GPUs such as the A100!
- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
- RAM: 1 TB
(nnU-net by default uses one GPU per training. The server configuration can run up to 8 model trainings simultaneously)
CPU/GPU ratio. For the server above (256 threads for 8 GPUs), a good value would be 24-30. You can do this by
it on with 4 threads in every data loader process. This matters most when `nnUNet_preprocessed` is on a network
file system and the dataset does not fit in RAM: we measured 1.8-3.2x more batches per second on NFS. Set
install the latest version with support for your hardware (cuda, mps, cpu).
```

## MONAI (health_science)

Source: official README, https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/README.md

```text
- multi-GPU multi-node data parallelism support.
To get started with the latest MONAI, use `docker run -ti --rm --gpus all projectmonai/monai:latest /bin/bash`.
```

## LAMMPS (GPU package) (materials_science)

Source: official documentation, https://raw.githubusercontent.com/lammps/lammps/develop/doc/src/Speed_gpu.rst

```text
GPU package
The GPU package was developed by Mike Brown while at SNL and ORNL (now
Northwestern).  Support for AMD GPUs via HIP was added by Vsevolod Nikolskiy
The GPU package provides GPU versions of many pair styles and for
* It is designed to exploit common GPU hardware configurations where one
  or more GPUs are coupled to many cores of one or more multicore CPUs,
  between the CPU(s) and GPU every timestep.
* Neighbor lists can be built on the CPU or on the GPU
  run on the GPU.  The FFT portion, which requires MPI communication
  can be performed concurrently on the GPU and CPU(s), respectively.
* It allows for GPU computations to be performed in single or double
* LAMMPS-specific code is in the GPU package.  It makes calls to a
  generic GPU library in the lib/gpu directory.  This library provides
  (for Nvidia GPUs, AMD GPUs, Intel GPUs, and multicore CPUs).
To compile and use this package in CUDA mode, you currently need
to have an NVIDIA GPU and install the corresponding NVIDIA CUDA
* Check if you have an NVIDIA GPU: ``cat /proc/driver/nvidia/gpus/\*/information``
* Go to https://developer.nvidia.com/cuda-downloads
* Run ``lammps/lib/gpu/nvc_get_devices`` (after building the GPU library, see below) to
(GPUs, CPUs, Accelerators) installed at the same time. OpenCL refers to those
as 'platforms'.  The GPU library will try to auto-select the best suitable platform,
command. run ``lammps/lib/gpu/ocl_get_devices`` to get a list of available
To compile and use this package for Intel GPUs, OpenCL or the Intel oneAPI
If you do not have a discrete GPU card installed, this package can still provide
significant speedups on some CPUs that include integrated GPUs. Additionally, for
in the ``lib/gpu`` directory.
Building LAMMPS with the GPU package
See the :ref:`Build extras <gpu>` page for
Run with the GPU package from the command-line
When using the GPU package, you cannot assign more than one GPU to a
single MPI task.  However multiple MPI tasks can share the same GPU,
# of CPU cores.  Assignment of multiple MPI tasks to a GPU will happen
GPUs/mode.  E.g. with 8 MPI tasks/node and 2 GPUs, each GPU will be
The GPU package also has limited support for OpenMP for both
This requires that the GPU library and LAMMPS are built with flags to
are also available in the GPU package. These run completely on the CPUs
Use the ``-sf gpu`` :doc:`command-line switch <Run_options>`, which will
automatically append "gpu" to styles that support it.  Use the ``-pk
gpu Ng`` :doc:`command-line switch <Run_options>` to set ``Ng`` = # of
GPUs/node to use. If ``Ng`` is 0, the number is selected automatically as
the number of matching GPUs that have the highest number of compute
   # 1 MPI task uses 1 GPU
   lmp -sf gpu -pk gpu 1 -in in.script
   # 12 MPI tasks share 2 GPUs on a single 16-core (or whatever) node
   mpirun -np 12 lmp -sf gpu -pk gpu 2 -in in.script
   mpirun -np 48 -ppn 12 lmp -sf gpu -pk gpu 2 -in in.script
Note that if the ``-sf gpu`` switch is used, it also issues a default
:doc:`package gpu 0 <package>` command, which will result in
automatic selection of the number of GPUs to use.
GPUs/node to use and additional options.  Its syntax is the same as
the ``package gpu`` command.  See the :doc:`package <package>`
Note that the default for the :doc:`package gpu <package>` command is to
GPU package pair styles.
Run with the GPU package by editing an input script
tasks/node, and use of multiple MPI tasks/GPU is the same.
Use the :doc:`suffix gpu <suffix>` command, or you can explicitly add an
"gpu" suffix to individual styles in your input script, e.g.
   pair_style lj/cut/gpu 2.5
You must also use the :doc:`package gpu <package>` command to enable the
GPU package, unless the ``-sf gpu`` or ``-pk gpu`` :doc:`command-line switches <Run_options>` were used.  It specifies the number of
```

## Quantum ESPRESSO (materials_science)

Source: official README, https://raw.githubusercontent.com/QEF/q-e/develop/README.md

```text
Quick installation instructions for CPU-based machines. For GPU execution, see
file [README_GPU.md](README_GPU.md). Go to the directory where this file is.
- devXlib: low-level utilities for GPU execution
with this program; if not, write to the Free Software Foundation, Inc.,
```

## MACE (ML interatomic potential) (materials_science)

Source: official README, https://raw.githubusercontent.com/ACEsuit/mace/main/README.md

```text
  - [CUDA acceleration with cuEquivariance](#cuda-acceleration-with-cuequivariance)
    --device=cuda \
#### Apple Silicon GPU acceleration
To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify `--device=mps`.
#### Multi-GPU training
For multi-GPU training, use the `--distributed` flag. This will use PyTorch's DistributedDataParallel module to train the model on multiple GPUs. Combine with on-line data loading for large datasets (see below). An example slurm script can be found in `mace/scripts/distributed_example.sbatch`.
## CUDA acceleration with cuEquivariance
MACE supports CUDA acceleration with the cuEquivariance library. To install the library and use the acceleration, see our documentation at https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html.
If you have a large dataset that might not fit into the GPU memory it is recommended to preprocess the data on a CPU and use on-line dataloading for training the model. To preprocess your dataset specified as an xyz file run the `preprocess_data.py` script. An example is given here:
    --device=cuda \
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')
calc = mace_off(model="medium", device='cuda')
  --device=cuda \
| `tests/backends` | e3nn ↔ cueq/oeq parity and converters | `cueq` / `oeq` extras (GPU for execution) |
| `tests/benchmarks` | performance measurements | GPU |
Capabilities a test needs (a GPU, an optional package, network access) are
```
