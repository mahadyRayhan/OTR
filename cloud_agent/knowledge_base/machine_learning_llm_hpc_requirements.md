# HPC Hardware Guidelines: Custom Machine Learning & LLM Training

Provenance: author-written guidance without cited sources. Prefer the sourced files (aws_ec2_catalog_*.md, sourced_tool_requirements.md) when they disagree.

## Training Custom Deep Learning Models (CNNs, Transformers)
- **Compute Profile:** Extremely GPU-bound. Matrix multiplications require high-bandwidth GPU memory (VRAM).
- **Hardware Recommendation:** AWS P-Family (e.g., `p4d` or `p5`) for large models, or G-Family (e.g., `g5`) for medium models. GCP A2/A3 VMs.
- **GPU Requirements:** 
  - For small models (e.g., standard CNNs, ResNets): NVIDIA T4 or A10G (16GB-24GB VRAM) is sufficient.
  - For large Transformer models (e.g., training DNABERT from scratch): Multiple NVIDIA A100 (40GB/80GB) or H100 GPUs are mandatory. Distributed training across multi-GPU nodes is often required.
- **Storage:** High-throughput NVMe SSDs are required to feed training data to the GPUs fast enough to prevent I/O bottlenecking.

## Fine-Tuning LLMs (LoRA, QLoRA, PEFT)
- **Compute Profile:** GPU-bound, but highly optimized to reduce VRAM requirements.
- **Hardware Recommendation:** A single AWS `g5.2xlarge` (NVIDIA A10G) or `p3.2xlarge` (NVIDIA V100).
- **Memory (VRAM) Considerations:** Using techniques like QLoRA (Quantized Low-Rank Adaptation) allows fine-tuning 7B-13B parameter models (like Llama 3 or Mistral) on a single 24GB VRAM GPU, making it highly cost-effective compared to full-parameter fine-tuning.

## MLOps & Frameworks
- **OS:** Ubuntu 22.04 LTS is the absolute standard.
- **Core Frameworks:** PyTorch or TensorFlow. PyTorch is heavily favored in modern bioinformatics (e.g., HuggingFace Transformers, PyTorch Geometric for graphs).
- **Drivers:** NVIDIA CUDA Toolkit, cuDNN, and NVIDIA Container Toolkit are mandatory for GPU utilization.
- **Distributed Training:** For multi-node training, frameworks like DeepSpeed, Ray, or PyTorch DDP (Distributed Data Parallel) are required, alongside high-bandwidth networking (AWS EFA/Elastic Fabric Adapter).

## Cloud Deployment (Inference / Serving)
- Once a custom model is trained, it does not require an A100 GPU for inference.
- **Hardware for Serving:** Deploy the model on cheaper instances with NVIDIA T4 GPUs (AWS `g4dn`), or use AWS Inferentia chips (`Inf2`) which are purpose-built for low-cost deep learning inference.
- **Infrastructure:** Deploy as a containerized microservice using AWS ECS (Fargate) or Amazon EKS (Kubernetes), orchestrated behind an Application Load Balancer.