# HPC Storage, Interconnect, and MPI Standards

## Shared File System for Multi-Node HPC (Amazon FSx for Lustre)
- **Why Required:** Multi-node MPI jobs need a shared POSIX file system every rank can read and write. S3 alone is insufficient — it is not POSIX and cannot back a running MPI job's scratch directory.
- **What It Provides:** FSx for Lustre is POSIX-compliant with read-after-write consistency and file locking, delivering sub-millisecond latencies, up to multiple TBps of throughput, and up to millions of IOPS.
- **Client Support:** The Lustre client ships with Amazon Linux 2023 and Amazon Linux 2. For RHEL, CentOS, and Ubuntu, AWS provides a Lustre client repository.
- **Access:** Mountable from EC2, EKS (via the FSx for Lustre CSI driver), and ECS. Instances in other Availability Zones in the same VPC can mount it if subnet routing allows, though same-AZ access is preferred for latency.
- **Security:** Encrypted at rest automatically using AWS KMS keys. Encryption in transit is automatic in certain Regions from supported EC2 instance types. FSx has been assessed for ISO, PCI-DSS, and SOC, and is **HIPAA eligible**.
- **Source:** AWS FSx for Lustre User Guide, "What is Amazon FSx for Lustre?" (docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html).

## FSx for Lustre Deployment Types and Storage Classes
- **Scratch vs Persistent:** *Scratch* file systems suit temporary, shorter-term processing — data is **not replicated and does not persist if a file server fails**. *Persistent* file systems replicate data and replace failed file servers, suiting longer-term and throughput-focused work.
- **Selection Rule:** Use Scratch for the working directory of a single simulation campaign whose inputs live in S3 and whose outputs are copied back. Use Persistent when the file system is the durable home of the data.
- **Storage Classes:** *SSD* — small random file operations, up to TBps, consistent sub-millisecond latency across the full dataset. *Intelligent-Tiering* — AWS's recommended default for most workloads, fully elastic, sub-millisecond access to frequently-used data with an optional SSD read cache. *HDD* — consistent single-digit-millisecond latency and up to tens of GBps, with an optional SSD read cache sized at 20% of HDD capacity.
- **Cost Note:** Do not default to SSD. Intelligent-Tiering is the documented recommendation unless the workload genuinely needs uniform low latency across the entire dataset.
- **Source:** AWS FSx for Lustre User Guide, "Deployment options and storage classes".

## Linking Compute Storage to S3 (Data Repository Association)
- **Mechanism:** An FSx for Lustre file system can be linked to an S3 bucket, transparently presenting S3 objects as files. FSx imports listings of existing objects at file system creation and can import listings of objects added later.
- **Write-Back:** The file system can write data back to S3. *Data repository tasks* handle transferring data and metadata between the file system and S3.
- **Recommended Pattern:** Keep the durable dataset in S3, create a Scratch FSx file system linked to it for the duration of a campaign, run the calculations, export results back to S3, then delete the file system. This avoids paying for high-performance storage between campaigns.
- **On-Premises Bursting:** Data can be imported from on-premises stores over Direct Connect or Site-to-Site VPN for hybrid bursting.
- **Source:** AWS FSx for Lustre User Guide, "FSx for Lustre and data repositories".

## Elastic Fabric Adapter (EFA) — Provisioning Requirements
- **Security Group:** EFA requires a security group that allows **all inbound and outbound traffic to and from the security group itself** (a self-referencing rule). This is the most common EFA misconfiguration — without it, ranks cannot communicate.
- **Network Interface:** The instance must be launched into an explicitly chosen subnet with interface type **EFA with ENA**. If no subnet is selected, EFA cannot be enabled.
- **Placement:** AWS recommends launching EFA-enabled instances into a **cluster placement group**, which places them in a low-latency group within a single Availability Zone. It is recommended rather than strictly required.
- **Capacity:** Pair a cluster placement group with a Capacity Reservation to guarantee capacity as the cluster scales.
- **Passwordless SSH:** MPI requires passwordless SSH from the leader node to all member nodes.
- **Recommended Workflow:** Configure one temporary instance, install the EFA software and the application, create an **EFA-enabled AMI**, then launch the cluster from that AMI.
- **Source:** AWS EC2 User Guide, "Get started with EFA and MPI for HPC workloads" (docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html).

## MPI Stack Selection (Open MPI vs Intel MPI on EFA)
- **What the EFA Installer Provides:** The EFA installer installs the EFA-enabled kernel module, EFA drivers, Libfabric, and an Open MPI stack. Libfabric is installed to `/opt/amazon/efa`, Open MPI 4.1 to `/opt/amazon/openmpi`, and Open MPI 5 to `/opt/amazon/openmpi5`.
- **Open MPI:** From EFA installer 1.30.0, both Open MPI 4.1 and Open MPI 5 are installed by default. Open MPI 5 must be activated with `module load openmpi5`. Install a single version with `--mpi=openmpi4` or `--mpi=openmpi5`.
- **Intel MPI:** Requires a separate installation from Intel and is not supported on every operating system. Install EFA with `./efa_installer.sh -y --minimal` if using Intel MPI's embedded Libfabric.
- **Preferring the EFA Libfabric:** The EFA installer typically ships a newer, sometimes more performant Libfabric than Intel MPI's embedded copy. Set `I_MPI_OFI_LIBRARY_INTERNAL=0` to use it.
- **Intra-Node Fabric:** Intel MPI uses shared memory (`shm`) intra-node and Libfabric (`ofi`) inter-node by default, which generally performs best. If an application hangs, force `I_MPI_FABRICS=ofi`; the EFA Libfabric provider uses shared memory internally, so performance is similar.
- **Source:** AWS EC2 User Guide, "Get started with EFA and MPI for HPC workloads", Steps 3–5.

## EFA Verification and Silent Failure Modes
- **The Critical Failure Mode:** If EFA is unavailable due to misconfiguration, **Intel MPI silently falls back to the TCP/IP network stack**, producing correct results at drastically reduced performance. A tightly-coupled DFT or CFD job can appear to work while losing most of the benefit of the HPC instance.
- **Preventing Silent Fallback:** Set `I_MPI_OFI_PROVIDER=efa`. The job then fails loudly with an `OFI fi_getinfo() failed` error instead of degrading quietly.
- **Verifying the Installation:** Run `fi_info -p efa -t FI_EP_RDM`. A working install returns the Libfabric EFA interfaces, showing `provider: efa` and `protocol: FI_PROTO_EFA`.
- **Verifying MPI:** `which mpicc` should return a path under `/opt/amazon/` for Open MPI or `/opt/intel/` for Intel MPI.
- **Runtime Diagnostics:** Set `I_MPI_DEBUG=1` or higher to print the Libfabric version and provider. A provider of `efa` confirms EFA is in use; `tcp;ofi_rxm` means it silently fell back. A version suffixed with `impi` indicates Intel MPI's internal Libfabric rather than the EFA one.
- **Ubuntu Requirement:** Libfabric uses Cross Memory Attach for intra-node shared memory, which is incompatible with ptrace protection. On distributions that enable it by default (such as Ubuntu), set `kernel.yama.ptrace_scope=0`.
- **Source:** AWS EC2 User Guide, "Get started with EFA and MPI for HPC workloads", Steps 5–7.

## Cluster Provisioning and Scientific Software Builds
- **AWS ParallelCluster:** An AWS-supported open-source cluster management tool for deploying and managing HPC clusters. It can automatically create FSx for Lustre file systems, or attach existing ones, during cluster creation. This is the recommended path for a Slurm-based HPC cluster rather than hand-rolling EC2 provisioning.
- **AWS Batch:** Integrates with FSx for Lustre through EC2 launch templates, and sizes instances dynamically from job resource requirements. Better suited to high-throughput independent jobs than to tightly-coupled MPI.
- **Container Formats:** Docker for ECS/EKS-based work; Singularity/Apptainer where an unprivileged runtime is required. MPI applications in containers must use an MPI build compatible with the host's Libfabric and EFA driver — version mismatches between container MPI and host EFA are a common source of fallback-to-TCP failures.
- **Building From Source:** `[VERIFY]` Spack and EasyBuild are the common package managers for building scientific codes such as VASP, Quantum ESPRESSO, and LAMMPS with EFA-aware MPI. This recommendation was not drawn from a retrieved AWS source — confirm against Spack documentation and the AWS HPC blog before relying on specifics.
- **Source:** AWS FSx for Lustre User Guide, "Integrations with AWS services"; AWS EC2 User Guide EFA tutorial (container/MPI compatibility inferred from the Libfabric version guidance).
