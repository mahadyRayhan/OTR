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
    DEFAULT_DECOMPOSER = "neuro_rule"
    
    # 🚀 NEW: THE DAG BUILDER
    # Options: "topological_dag", "parallel_dag", "got_dag"
    DAG_BUILDER = "got_dag"

    # ---------------------------------------------------------
    # AGENT SETTINGS
    # ---------------------------------------------------------
    USE_PROFILER = True
    # Options: "dirichlet" (Bayesian, math-heavy) or "naive_bayes" (fast sklearn NB)
    PROFILER_MODE = "naive_bayes"
    # Agent reasoning depth override.
    # None  — use whatever depth the GoT engine assigns per node (1=leaf, 2=mid, 3=root).
    # 1     — lock all agents to depth=1 (fast, direct generation, no extra reasoning).
    DEFAULT_DEPTH = 1

    # LLM-as-Judge ablation flags
    # USE_AGENT_JUDGE  — Gemini judge called after each GoT node; drives quality-gated
    #                    retries and expansion. False → heuristic judge (length/structure).
    # USE_FINAL_JUDGE  — GPT judge called once on the final synthesised answer.
    #                    False → skips the call entirely (score stored as N/A).
    USE_AGENT_JUDGE = True
    USE_FINAL_JUDGE = True

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------
    # Orchestrator model — used for decomposer (goal DAG) and all judges.
    # Flash is worth the extra latency here: a bad decomposition breaks the whole
    # run, and a miscalibrated judge triggers unnecessary Reflexion (+40s).
    # RAG uses Flash Lite (rag_config.py). CODE/CLOUD already use Flash (their own GEMINI_MODEL).
    # GENERATION_MODEL = "gemini-flash-latest"
    GENERATION_MODEL = "gemini-3.1-flash-lite-preview"
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

    # ---------------------------------------------------------
    # ORCHESTRATION MODE
    # ---------------------------------------------------------
    # Controls the execution strategy used by the engine.
    # Options:
    #   "got_dag"              — Adaptive GoT-DAG with quality gates and dynamic
    #                            node expansion (default, full system).
    #   "react_cot_context"    — Linear sequential execution; upstream agent output
    #                            is passed as raw context to each downstream agent.
    #                            No CUGCR compression, no GoT, no retry loop.
    #   "react_cot_no_context" — Linear sequential execution; no context propagation
    #                            between agents. Each agent only sees its sub-question.
    ORCHESTRATION_MODE = "react_cot_context"