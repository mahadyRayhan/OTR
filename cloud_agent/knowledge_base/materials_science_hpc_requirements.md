# HPC Hardware Guidelines: Computational Materials Science

## Plane-Wave DFT (VASP, Vienna Ab initio Simulation Package)
- **Compute Profile:** CPU-bound and tightly coupled. 3D FFTs are central to VASP's speed; poor MPI rank placement across nodes "deteriorates the performance". Communication-bound, not embarrassingly parallel.
- **Parallelization:** Hybrid MPI + OpenMP. `NCORE` controls FFT parallelization, `KPAR` splits k-point groups. Note: "running on multiple OpenMP threads and/or GPUs switches off the NCORE parallelization" — CPU and GPU builds tune differently.
- **Hardware Recommendation:** Hpc7a (192 cores, 3.7 GHz, 300 Gbps EFA, 768 GiB ≈ 4 GiB/core) or Hpc6a (96 cores, 3.6 GHz, 100 Gbps, 384 GB). EFA plus a cluster placement group is mandatory for multi-node runs. For multi-node VASP the interconnect, not core count, is usually limiting.
- **Storage:** Shared POSIX scratch required — see `hpc_storage_and_mpi_standards.md`.
- **Memory:** `[VERIFY]` Scales with cell size, k-points, and bands. No general per-core figure exists; size against your largest cell.
- **GPU:** VASP 6.x supports GPU; 5.4.4 is CPU-only. `[VERIFY]` Speedup is system-dependent, not benchmarked on AWS in any source consulted.
- **Spot Suitability:** POOR — tightly-coupled MPI dies when any instance is reclaimed. `[VERIFY]` Follows from the coupling, not a cited benchmark.
- **Licensing:** Commercial. Per-core licence cost often favours fewer, faster cores.
- **Source:** VASP Wiki, Category:Parallelization (vasp.at/wiki); AWS HPC Blog, "Deep dive into Hpc7a".

## Open-Source Plane-Wave DFT (Quantum ESPRESSO, PWscf)
- **Compute Profile:** CPU-bound, tightly coupled — same class as VASP without the licence. The realistic default for cloud DFT.
- **Parallelization:** Five MPI levels set at run time — images (`-nimage`), pools (`-nk`), plane waves, task groups (`-ntg`), linear algebra (`-nd`, via ScaLAPACK/ELPA). Images and pools are loosely coupled with modest communication; processors *within* a pool are tightly coupled.
- **Key Sizing Rule:** The QE user guide states "fast communication hardware is needed if your pool extends over more than a few processors on different nodes." Keep a pool inside one instance where possible and use `-nk` to spread k-points across instances — this converts a communication-bound job into a far more scalable one.
- **Hardware Recommendation:** Hpc7a/Hpc6a with EFA and a cluster placement group. QE tolerates multi-node scaling better than a single large VASP run, provided `-nk` is set deliberately.
- **Linear Algebra:** `-ndiag` requires a square processor grid (n²) and a ScaLAPACK-enabled build.
- **Memory:** `[VERIFY]` Scales with cell size, cutoff, and band count; no published general figure.
- **Spot Suitability:** POOR for one tightly-coupled run; ACCEPTABLE using `-nimage` for independent, individually restartable calculations. `[VERIFY]`
- **Source:** Quantum ESPRESSO User Guide, "Parallelization levels".

## Classical Molecular Dynamics (LAMMPS)
- **Compute Profile:** GPU-accelerable, moderately coupled. Substantially more GPU-friendly than DFT.
- **GPU Guidance:** The GPU package accelerates many pair styles plus charge assignment and force interpolation in PPPM. The **FFT portion of PPPM stays on the CPU** because it requires MPI communication — long-range electrostatics remains partly CPU-bound and interconnect-sensitive.
- **MPI/GPU Ratio:** Use **2–10 MPI tasks per GPU**; multiple ranks sharing a GPU "will often give the best performance", but "using too many MPI tasks will result in worse performance due to growing overhead". With few atoms per rank, use fewer tasks.
- **Hardware Recommendation:** G5 or P-family for GPU runs; Hpc7a/Hpc6a for CPU runs. `[VERIFY]` Sizing depends on pair style and atom count; no AWS LAMMPS benchmark located.
- **Precision:** Single, mixed, and double builds differ materially in performance. Mixed is the usual starting point.
- **Overlap:** GPU pair styles can run asynchronously with CPU-side bond/angle/dihedral terms, hiding GPU time.
- **Spot Suitability:** MODERATE — restarts cleanly if restart files are written frequently to durable storage. `[VERIFY]`
- **Source:** LAMMPS Documentation, "Speed: GPU package" (docs.lammps.org/Speed_gpu.html).

## Biomolecular and Soft-Matter MD (GROMACS)
- **Compute Profile:** GPU-accelerable, scales sub-linearly across nodes.
- **Measured Scaling (hpc6a):** 10.8 ns/day on one instance rises to ~60 ns/day on 8 instances (69% parallel efficiency) and 115 ns/day on 32 instances — an 11× gain. Note the efficiency collapse: 4× the instances beyond 8 buys under 2× the throughput.
- **Sizing Consequence:** A cost-optimal node count exists well below the maximum. Scaling past the knee wastes money for marginal wall-clock gain; benchmark to locate it.
- **Hardware Recommendation:** Hpc6a/Hpc7a with EFA for CPU runs; G5/P-family for GPU runs. `[VERIFY]` Figures above are for one unnamed benchmark system and will not transfer directly.
- **Spot Suitability:** MODERATE — checkpoints reliably; pair with frequent checkpoint writes.
- **Source:** `[VERIFY — SECONDARY]` The hpc6a scaling figures were obtained from a search summary of "Comparing HPC Workload performance between Intel and AMD on AWS" (Intel Tech Innovation blog), NOT from the primary page. Every other section in this file was drafted from a directly retrieved source. Confirm before relying on these numbers.

## Machine-Learning Interatomic Potentials (MACE, NequIP, Allegro)
- **Compute Profile:** Two phases with opposite requirements. **Training** is GPU-bound and VRAM-sensitive. **Inference** (running MD with the trained potential) is far cheaper, often single-GPU.
- **Hardware Recommendation:** Training — P-family (A100/H100 class) for large models, G5 (A10G, 24 GB) for smaller. Inference — a single G5 is frequently sufficient.
- **Architectural Consequence:** Do not size the whole pipeline for training. Train once on expensive GPU capacity, then serve inference on much cheaper instances.
- **Two-Stage Dependency:** Training sets are usually generated by DFT (VASP/QE), so an MLIP project is a tightly-coupled DFT campaign feeding a GPU training job. The two stages need different clusters.
- **Spot Suitability:** GOOD for training with checkpointing; GOOD for inference.
- **Source:** `[VERIFY]` Reasoned from the general structure of MLIP workflows, not a cited benchmark. Confirm against MACE/NequIP repository documentation.

## High-Throughput Screening (pymatgen, atomate2, FireWorks)
- **Compute Profile:** Embarrassingly parallel — thousands of small, independent DFT calculations. **The architectural opposite of a single large VASP run**, despite using the same code.
- **Hardware Recommendation:** Many small/medium instances rather than few large. C-family or Hpc6a. **EFA is not required** — jobs do not communicate.
- **Spot Suitability:** EXCELLENT. Each calculation is independent and restartable; a reclaimed instance loses one calculation, not the campaign.
- **Orchestration:** FireWorks or atomate2 manages queueing and retries. NERSC recommends FireWorks for many similar jobs and warns against bundling more than ~10 jobs into a single scheduler submission.
- **Storage:** Output volume, not compute, is often the constraint. Use S3 with lifecycle policies to Glacier for completed campaigns.
- **Key Discriminator:** If a user asks about "running VASP", the architecture depends entirely on whether they mean *one large calculation* (EFA, On-Demand, tightly coupled) or *a screening campaign* (Spot, no EFA, embarrassingly parallel). Ask which.
- **Source:** NERSC VASP documentation, "Running multiple VASP jobs simultaneously"; atomate2/FireWorks documentation.

## Interconnect and Coupling Requirements (cross-cutting)
- **Why This Matters:** Materials simulation is dominated by tightly-coupled MPI, a workload class with no analogue in genomics or single-cell pipelines. Recommending a general-purpose instance family for DFT is a common and expensive mistake.
- **Tightly Coupled (needs EFA + cluster placement group):** VASP, Quantum ESPRESSO within a pool, CP2K, large LAMMPS runs with long-range electrostatics, FEM and phase-field solvers.
- **Loosely Coupled (EFA unnecessary):** High-throughput screening, QE `-nimage` runs, parameter sweeps, MLIP inference, post-processing and analysis.
- **EFA Availability:** Hpc7a offers 300 Gbps, Hpc6a 100 Gbps. Hpc7a delivers roughly 2× the cores, 1.5× the memory channels, and 3× the network bandwidth of Hpc6a, giving up to 2.7× better per-instance and about 29% better per-core performance across AWS's tested CFD, MD, and weather workloads.
- **Cost Rule:** Tightly-coupled work should run On-Demand or Reserved; loosely-coupled work should run on Spot. Applying the genomics default (Spot everywhere, because Nextflow retries) to a DFT job makes it fail repeatedly and cost more, not less.
- **Source:** AWS HPC Blog, "Deep dive into Hpc7a"; Quantum ESPRESSO User Guide (coupling characteristics).
