# Tool Classification Reference: Computational Materials Science

Provenance: author-written guidance without cited sources. Prefer the sourced files (aws_ec2_catalog_*.md, sourced_tool_requirements.md) when they disagree.

## Plane-Wave DFT Codes (VASP, Quantum ESPRESSO, ABINIT, CASTEP, GPAW)
- **Tools:** VASP, Quantum ESPRESSO (PWscf), ABINIT, CASTEP, GPAW, Elk, Exciting.
- **Compute Profile:** CPU-bound, tightly coupled MPI. Dominated by 3D FFTs and iterative diagonalisation, both communication-heavy.
- **Hardware Recommendation:** AWS Hpc7a or Hpc6a with EFA and a cluster placement group. Shared POSIX scratch required (FSx for Lustre).
- **Spot Suitability:** POOR — losing one rank kills the job.
- **Sizing Note:** All members of this class share VASP's profile; apply the VASP guidance in `materials_science_hpc_requirements.md`.
- **Source:** Grouping by numerical method (plane-wave basis + iterative diagonalisation). Per-code confirmation: each project's own parallelisation documentation.

## All-Electron and Localised-Basis DFT (WIEN2k, SIESTA, OpenMX, FHI-aims, CP2K)
- **Tools:** WIEN2k (FLAPW, all-electron), SIESTA, OpenMX, FHI-aims, CP2K (Gaussian plane-wave hybrid).
- **Compute Profile:** CPU-bound. Localised-basis codes (SIESTA, OpenMX) communicate less than plane-wave codes and tolerate looser interconnects; all-electron FLAPW (WIEN2k) is more memory-intensive per atom. `[VERIFY]` Relative memory factors vary strongly by system and are not published as general figures.
- **Hardware Recommendation:** Hpc6a/Hpc7a. EFA still recommended for WIEN2k and CP2K; less critical for SIESTA/OpenMX at modest node counts.
- **Spot Suitability:** POOR for single large runs; acceptable for independent-calculation campaigns.
- **Source:** Grouping by basis-set type. `[VERIFY]` against each code's parallelisation guide before quoting specifics.

## Molecular Dynamics Engines (LAMMPS, GROMACS, NAMD, HOOMD-blue, OpenMM)
- **Tools:** LAMMPS, GROMACS, NAMD, HOOMD-blue, OpenMM, DL_POLY.
- **Compute Profile:** GPU-accelerable, moderately coupled. Long-range electrostatics (PPPM/PME) retains a communication-bound FFT step.
- **Hardware Recommendation:** G5 or P-family for GPU runs; Hpc7a/Hpc6a with EFA for large CPU runs.
- **Spot Suitability:** MODERATE — all checkpoint and restart cleanly; write restarts frequently to durable storage.
- **Sizing Note:** Expect sub-linear multi-node scaling; find the cost-optimal node count by benchmarking rather than assuming linear speedup.
- **Source:** LAMMPS `Speed_gpu` documentation; GROMACS scaling behaviour (see MD sections of `materials_science_hpc_requirements.md`).

## Quantum Chemistry Codes (Gaussian, ORCA, NWChem, Psi4, Molpro)
- **Tools:** Gaussian, ORCA, NWChem, Psi4, Molpro, TURBOMOLE.
- **Compute Profile:** Molecular rather than periodic. Frequently **memory- and scratch-disk-bound** rather than network-bound; many methods (notably post-Hartree-Fock: MP2, CCSD(T)) scale steeply with system size and write very large temporary files.
- **Hardware Recommendation:** Memory-optimised R-family or X-family with large, fast local NVMe scratch. Prioritise local disk throughput over interconnect.
- **Key Difference From DFT:** Do NOT default these to the tightly-coupled EFA recipe. Many quantum chemistry jobs run best on a single large node.
- **Spot Suitability:** POOR to MODERATE — depends on whether the method checkpoints. `[VERIFY]` per code.
- **Source:** `[VERIFY]` Classified by method family (local scratch and memory dominance in post-HF methods). Confirm against the specific code's hardware guide.

## Beyond-DFT and Quantum Monte Carlo (BerkeleyGW, Yambo, QMCPACK)
- **Tools:** BerkeleyGW, Yambo, ABINIT GW module, QMCPACK, CASINO.
- **Compute Profile:** Two opposite sub-classes. **GW/BSE** (BerkeleyGW, Yambo) is extremely memory-hungry and tightly coupled. **QMC** (QMCPACK, CASINO) is statistically independent across walkers and scales almost linearly — closer to an embarrassingly parallel workload.
- **Hardware Recommendation:** GW/BSE — largest available memory per node (Hpc7a 768 GiB, or X-family) with EFA. QMC — many nodes, EFA less critical.
- **Spot Suitability:** GW/BSE POOR; QMC GOOD (walkers are independent and restartable).
- **Source:** `[VERIFY]` Classified by algorithm structure. Confirm memory figures against each code's documentation.

## Semi-Empirical and Tight-Binding (DFTB+, xTB, LAMMPS ReaxFF)
- **Tools:** DFTB+, xTB / GFN-xTB, ReaxFF (via LAMMPS), MOPAC.
- **Compute Profile:** Orders of magnitude cheaper than DFT. Usually single-node, often single-core to modest thread counts.
- **Hardware Recommendation:** General-purpose C-family or M-family. EFA unnecessary. Do not provision HPC instances for these.
- **Spot Suitability:** EXCELLENT — short, cheap, restartable.
- **Common Error:** Treating these as DFT-class workloads massively over-provisions the cluster.
- **Source:** `[VERIFY]` Classified by method cost. Confirm against each code's documentation.

## Analysis, Visualisation, and Post-Processing (OVITO, VESTA, phonopy, ASE, pymatgen, matminer)
- **Tools:** OVITO, VESTA, phonopy, ASE (Atomic Simulation Environment), pymatgen, matminer, py4DSTEM, ParaView.
- **Compute Profile:** Light, single-node, mostly serial Python or desktop GUI. Memory scales with trajectory or dataset size, not with core count.
- **Hardware Recommendation:** A single M-family or R-family instance. **Never** provision an HPC cluster for these. For interactive visualisation (OVITO, VESTA, ParaView), consider a G4dn/G5 instance with a remote desktop rather than a batch cluster.
- **Spot Suitability:** EXCELLENT for batch post-processing.
- **Storage Note:** These tools read simulation output, so co-locate them with the FSx for Lustre file system or S3 bucket holding the trajectories.
- **Source:** `[VERIFY]` Classified by workload type (post-processing rather than simulation).

## Workflow and High-Throughput Orchestration (AiiDA, FireWorks, atomate2, Snakemake)
- **Tools:** AiiDA, FireWorks, atomate2, custodian, Snakemake, Nextflow.
- **Compute Profile:** The orchestrator itself is negligible; it submits and tracks many independent jobs. Architecture is driven by the *managed* workload, not the manager.
- **Hardware Recommendation:** Small persistent instance (t3/m6i) for the workflow server plus a database (RDS or self-hosted MongoDB for FireWorks). Worker capacity sized to the underlying calculations.
- **Spot Suitability:** EXCELLENT for workers; the orchestrator node itself should be On-Demand so campaign state is not lost.
- **Critical Distinction:** A query mentioning these tools is almost always about a *high-throughput campaign* — embarrassingly parallel, Spot-friendly, no EFA — even when the underlying code is VASP.
- **Source:** atomate2 / FireWorks / AiiDA project documentation; NERSC guidance recommending FireWorks for many similar jobs.

## Commercial Multiphysics and FEM (COMSOL, ABAQUS, ANSYS, MOOSE)
- **Tools:** COMSOL Multiphysics, ABAQUS, ANSYS Mechanical/Fluent, MOOSE, FEniCS.
- **Compute Profile:** CPU-bound, tightly coupled MPI for implicit solvers. Commercially licensed except MOOSE and FEniCS.
- **Hardware Recommendation:** Hpc7a or Hpc6a with EFA and a cluster placement group.
- **Licensing Constraint:** Per-core licence pricing usually favours fewer, faster cores. Hpc7a offers sizes with cores disabled at the same price specifically for this case — the licence cost, not the instance cost, typically dominates.
- **Spot Suitability:** POOR — tightly coupled, and a reclaimed instance can waste checked-out licence tokens.
- **Source:** AWS HPC Blog, "Deep dive into Hpc7a" (per-core licensing rationale and reduced-core sizes).
