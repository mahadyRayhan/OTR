class Config:
    # ---------------------------------------------------------
    # SYSTEM MODE
    # ---------------------------------------------------------
    # Options: "interactive" or "benchmark"
    RUN_MODE = "interactive"

    # ---------------------------------------------------------
    # ORCHESTRATION SETTINGS
    # ---------------------------------------------------------
    # Options: "plan_and_solve", "llm_compiler", "qdmr_rule", "coverage", "ultimate_neuro_symbolic", "auto"
    DEFAULT_DECOMPOSER = "ultimate_neuro_symbolic"
    
    # 🚀 NEW: THE DAG BUILDER
    # Options: "topological_dag", "parallel_dag", "got_dag"
    DAG_BUILDER = "got_dag"

    # ---------------------------------------------------------
    # AGENT SETTINGS
    # ---------------------------------------------------------
    USE_PROFILER = True

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------
    GENERATION_MODEL = "gemini-2.5-flash"
    JUDGE_MODEL = "gpt-5.4"

    # ---------------------------------------------------------
    # AGENT CAPABILITIES (Master Registry)
    # ---------------------------------------------------------
    AGENT_REGISTRY_PROMPT = """
    AGENT REGISTRY & CAPABILITIES:
    - 'rag' : General knowledge, biology concepts, definitions, theory, facts, pathway analysis, data formats.
    - 'scholar' : Finding actual human experts, researchers, real names, research papers, and citations.
    - 'code' : Writing Python scripts, pipeline code (Snakemake/Nextflow), code generation, debugging.
    - 'cloud' : AWS, cloud deployment, infrastructure design (Terraform/Docker/Kubernetes).
    """
    # ---------------------------------------------------------
    # CONTEXT ROUTING ALGORITHM
    # ---------------------------------------------------------
    # Options: "light" (Regex only - fast) or "heavy" (Dense+Sparse Semantic Embeddings - slow but accurate)
    CONTEXT_ROUTING_MODE = "heavy"