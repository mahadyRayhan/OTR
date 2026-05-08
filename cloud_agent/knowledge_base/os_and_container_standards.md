# Operating System and Container Standards

## Base Operating Systems
- **Ubuntu 22.04 LTS (Jammy Jellyfish):** The absolute standard for bioinformatics. It has the best support for apt-get bioinformatics packages, Python compilation, and CUDA drivers.
- **Amazon Linux 2023:** Highly optimized for AWS environments (ECS, Batch). Smaller footprint, but lacks some pre-compiled scientific libraries found in Ubuntu.

## Container Frameworks
- **Docker:** The standard for building images. Used extensively in AWS ECS and Batch.
- **Singularity/Apptainer:** Required for running containers on traditional university HPC clusters (Slurm, PBS) where users do not have root access (Docker requires root; Singularity does not).
- **Conda/Mamba:** Essential for creating reproducible virtual environments inside containers, especially for complex R and Python dependencies.