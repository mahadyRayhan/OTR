# HPC Hardware Guidelines: Computational Materials Science

## Plane-Wave DFT (VASP, Vienna Ab initio Simulation Package)
- **Compute Profile:** CPU-bound and *tightly coupled*. VASP splits work across MPI ranks that communicate continuously; 3D FFTs are central to its speed, and VASP's own documentation warns that poor MPI rank placement across nodes "deteriorates the performance". This is a communication-bound workload, not an embarrassingly parallel one.
- **Parallelization:** Hybrid MPI + OpenMP. `NCORE` controls FFT parallelization, `KPAR` splits the calculation into k-point groups. Note that "running on multiple OpenMP threads and/or GPUs switches off the NCORE parallelization" — the tuning changes entirely between CPU and GPU builds.
- **Hardware Recommendation:** AWS Hpc7a (192 physical cores, 3.7 GHz all-core turbo) or Hpc6a (96 cores, 3.6 GHz). EFA is mandatory for multi-node runs. Place instances in a cluster placement group.
- **Interconnect:** Hpc7a provides 300 Gbps EFA; Hpc6a provides 100 Gbps. For multi-node VASP the interconnect, not the core count, is usually the limiting factor.
- **Memory:** Hpc7a provides 768 GiB per instance (~4 GiB per core); Hpc6a provides 384 GB. `[VERIFY]` VASP memory scales with system size, k-points, and bands — no general per-core figure exists. Size against your own largest cell.
- **GPU:** VASP 6.x supports GPU execution; 5.4.4 is CPU-only. `[VERIFY]` GPU speedup is system-dependent and not benchmarked on AWS in any source consulted.
- **Spot Suitability:** POOR. Tightly-coupled MPI jobs die entirely when any one instance is reclaimed. Use On-Demand or Reserved capacity. `[VERIFY]` — this follows from the coupling, not from a cited AWS benchmark.
- **Licensing:** Commercial. Requires an active VASP license; per-core licence costs often favour fewer, faster cores over many slow ones.
- **Source:** VASP Wiki, Category:Parallelization (vasp.at/wiki); AWS HPC Blog, "Deep dive into Hpc7a" (aws.amazon.com/blogs/hpc).

## Open-Source Plane-Wave DFT (Quantum ESPRESSO, PWscf)
- **Compute Profile:** CPU-bound, tightly coupled — same class as VASP, without the licence constraint. The realistic default for cloud DFT.
- **Parallelization:** Five MPI levels, selected at run time: images (`-nimage`), pools (`-nk`), plane waves, task groups (`-ntg`), and linear algebra (`-nd`, via ScaLAPACK/ELPA). Images and pools are *loosely* coupled with modest communication; processors *within* a pool are tightly coupled.
- **Key Sizing Rule:** The QE user guide states that "fast communication hardware is needed if your pool extends over more than a few processors on different nodes." Practical consequence: keep a pool inside one instance where possible, and use `-nk` to spread k-points across instances — this converts a communication-bound job into a far more scalable one.
- **Hardware Recommendation:** AWS Hpc7a or Hpc6a with EFA and a cluster placement group. Because pools are loosely coupled, QE tolerates multi-node scaling better than a single large VASP run — provided `-nk` is set deliberately.
- **Memory:** `[VERIFY]` Scales with cell size, cutoff, and number of bands. No general figure published; benchmark your own system.
- **Linear Algebra:** `-ndiag` requires a square processor grid (n²) and a ScaLAPACK-enabled build.
- **Spot Suitability:** POOR for a single tightly-coupled run; ACCEPTABLE when using `-nimage` to run independent calculations that can be individually resubmitted. `[VERIFY]`
- **Source:** Quantum ESPRESSO User Guide, "Parallelization levels" (quantum-espresso.org/Doc/user_guide).

## Classical Molecular Dynamics (LAMMPS)
- **Compute Profile:** GPU-accelerable, moderately coupled. Substantially more GPU-friendly than DFT.
- **GPU Guidance:** The GPU package accelerates many pair styles plus the charge-assignment and force-interpolation parts of PPPM. The **FFT portion of PPPM stays on the CPU** because it requires MPI communication — so long-range electrostatics remains partly CPU-bound and interconnect-sensitive.
- **MPI/GPU Ratio:** LAMMPS documentation recommends **2–10 MPI tasks per GPU**, noting that multiple ranks sharing a GPU "will often give the best performance", but that "using too many MPI tasks will result in worse performance due to growing overhead". With few atoms per rank, use fewer tasks.
- **Hardware Recommendation:** GPU runs — AWS G5 or P-family. Pure-CPU runs — Hpc7a/Hpc6a. `[VERIFY]` Specific instance sizing depends on pair style and atom count; no AWS LAMMPS benchmark was located.
- **Precision:** Single, mixed, and double precision builds have materially different performance. Mixed precision is the usual starting point.
- **Overlap Optimisation:** GPU pair styles can run asynchronously with CPU-side bond/angle/dihedral terms, hiding GPU time.
- **Spot Suitability:** MODERATE. LAMMPS checkpoints and restarts cleanly, so Spot is viable *if* restart files are written frequently to durable storage. `[VERIFY]`
- **Source:** LAMMPS Documentation, "Speed: GPU package" (docs.lammps.org/Speed_gpu.html).

## Biomolecular & Soft-Matter MD (GROMACS)
- **Compute Profile:** GPU-accelerable, scales sub-linearly across nodes. The best-documented MD scaling behaviour on AWS.
- **Measured Scaling (hpc6a):** Single-instance performance of **10.8 ns/day** rises to roughly **60 ns/day on 8 instances** — a parallel efficiency of **69%** — and **115 ns/day on 32 instances**, an 11-fold gain over one instance. Note the efficiency collapse: 4× the instances beyond 8 buys under 2× the throughput.
- **Sizing Consequence:** There is a cost-optimal node count well below the maximum. Scaling past the knee wastes money for marginal wall-clock gain. Benchmark your own system to locate it.
- **Hardware Recommendation:** Hpc6a/Hpc7a with EFA for CPU runs; G5/P-family for GPU runs. `[VERIFY]` The measured figures above are for one unnamed benchmark system and will not transfer directly to yours.
- **Spot Suitability:** MODERATE — GROMACS checkpoints reliably; pair with frequent checkpoint writes.
- **Source:** `[VERIFY — SECONDARY]` The hpc6a scaling figures above were obtained from a search summary of "Comparing HPC Workload performance between Intel and AMD on AWS" (Intel Tech Innovation blog), NOT from the primary page. Every other section in this file was drafted from a directly retrieved source. Confirm these numbers against the original before relying on them.

## Machine-Learning Interatomic Potentials (MACE, NequIP, Allegro)
- **Compute Profile:** Two distinct phases with opposite requirements. **Training** is GPU-bound and benefits from high VRAM. **Inference** (running MD with the trained potential) is far cheaper and often single-GPU.
- **Hardware Recommendation:** Training — AWS P-family (A100/H100 class) for large models, G5 (A10G, 24 GB) for smaller ones. Inference — a single G5 is frequently sufficient.
- **Why This Matters Architecturally:** Do not size the whole pipeline for training. Train once on expensive GPU capacity, then serve inference on much cheaper instances.
- **Data Generation Dependency:** Training sets are usually produced by DFT (VASP/QE), so an MLIP project is really a *two-stage* pipeline — a tightly-coupled DFT campaign feeding a GPU training job. These two stages need different clusters.
- **Spot Suitability:** GOOD for training with checkpointing; GOOD for inference.
- **Memory / Storage:** `[VERIFY]` No verified figures — depends on model size, cutoff radius, and training set size.
- **Source:** `[VERIFY]` — This section is reasoned from the general structure of MLIP workflows, not from a specific cited benchmark. Confirm against MACE/NequIP repository documentation before relying on it.

## High-Throughput Screening (pymatgen, atomate2, FireWorks)
- **Compute Profile:** Embarrassingly parallel — thousands of small, *independent* DFT calculations. **The architectural opposite of a single large VASP run**, despite using the same underlying code.
- **Hardware Recommendation:** Many small/medium instances rather than few large ones. C-family or Hpc6a. **EFA is not required** — jobs do not communicate with each other.
- **Spot Suitability:** EXCELLENT. Each calculation is independent and individually restartable; a reclaimed instance loses one calculation, not the campaign. Expect substantial cost savings versus On-Demand.
- **Orchestration:** FireWorks or atomate2 manages the job queue and retries. VASP's own documentation recommends FireWorks when running many similar jobs, and warns against bundling more than ~10 jobs into a single scheduler submission.
- **Storage:** Output volume, not compute, is often the constraint — thousands of calculations each producing multi-MB outputs. Use S3 with lifecycle policies to Glacier for completed campaigns.
- **Key Discriminator:** If a user asks about "running VASP", the correct architecture depends entirely on whether they mean *one large calculation* (EFA, On-Demand, tightly coupled) or *a screening campaign* (Spot, no EFA, embarrassingly parallel). Ask which.
- **Source:** NERSC VASP documentation, "Running multiple VASP jobs simultaneously" (docs.nersc.gov/applications/vasp); atomate2/FireWorks project documentation.

## Interconnect and Coupling Requirements (cross-cutting)
- **Why This Section Exists:** Materials simulation is dominated by *tightly-coupled MPI*, a workload class with no analogue in genomics or single-cell pipelines. Recommending a general-purpose instance family for DFT is a common and expensive mistake.
- **Tightly Coupled (needs EFA + cluster placement group):** VASP, Quantum ESPRESSO within a pool, CP2K, large LAMMPS runs with long-range electrostatics, FEM/phase-field solvers.
- **Loosely Coupled (EFA unnecessary):** High-throughput screening, QE `-nimage` runs, parameter sweeps, MLIP inference, post-processing and analysis.
- **EFA Availability:** Provided on the Hpc instance families — Hpc7a offers 300 Gbps, Hpc6a 100 Gbps. Hpc7a delivers roughly 2× the cores, 1.5× the memory channels, and 3× the network bandwidth of Hpc6a, yielding up to 2.7× better per-instance performance and about 29% better per-core performance across AWS's tested CFD, molecular dynamics, and weather workloads.
- **Placement:** Always use a cluster placement group for tightly-coupled jobs so instances land on the same network spine.
- **Cost Rule of Thumb:** Tightly-coupled work should generally run On-Demand or Reserved; loosely-coupled work should run on Spot. Applying the genomics default (Spot everywhere, because Nextflow retries) to a DFT job will cause it to fail repeatedly and cost more, not less.
- **Source:** AWS HPC Blog, "Deep dive into Hpc7a" (aws.amazon.com/blogs/hpc); AWS EFA documentation; Quantum ESPRESSO User Guide (coupling characteristics).
