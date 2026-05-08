# HPC Hardware Guidelines: AI-Driven Structural Biology

## Protein Structure Prediction (AlphaFold 3)
- **Compute Profile:** Heavily GPU-bound. AlphaFold relies on massive deep learning transformers (Evoformer) to predict 3D structures from amino acid sequences.
- **Hardware Recommendation:** AWS P-Family or G-Family (`p4d.24xlarge` or `g5.2xlarge`).
- **GPU Requirements:** NVIDIA GPUs with high VRAM are mandatory. A single A100 GPU (40GB/80GB VRAM) is the gold standard. For smaller proteins, a T4 or A10G GPU (24GB VRAM) may suffice.
- **Memory & Storage:** 128GB CPU RAM. The AlphaFold genetic database requires ~3TB of highly available NVMe SSD storage.
- **OS & Frameworks:** Ubuntu 20.04/22.04. Requires NVIDIA CUDA drivers, Docker, NVIDIA Container Toolkit, JAX, and TensorFlow/PyTorch.

## Physics-Based Folding (Rosetta)
- **Compute Profile:** CPU-bound Monte Carlo simulations. 
- **Hardware Recommendation:** AWS C-Family (Compute Optimized). No GPUs required.