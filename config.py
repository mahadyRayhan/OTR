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
    # ACTIVE RESEARCH DOMAIN
    # ---------------------------------------------------------
    # Which domain's knowledge the agents should retrieve from.
    # Options: "bioinformatics", "materials_science", "health_science"
    # (see scholar_agent/domains.py for the registry).
    #
    # Currently a global switch. Per-query domain routing replaces this — the
    # engine will resolve the domain once per question and set
    # self._active_domain, which _execute_agent already prefers over this value.
    ACTIVE_DOMAIN = "bioinformatics"

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

    # Per-node fitness threshold tau (paper, Section 4.2.1). A node whose judge
    # fitness score falls below tau gets a failure descriptor and is repaired.
    # Runs before Nov 2026 used a hard-coded 0.80.
    PER_NODE_TAU = 0.70

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------
    # Orchestrator model — used for decomposer (goal DAG) and all judges.
    # Flash is worth the extra latency here: a bad decomposition breaks the whole
    # run, and a miscalibrated judge triggers unnecessary Reflexion (+40s).
    # RAG uses Flash Lite (rag_config.py). CODE/CLOUD already use Flash (their own GEMINI_MODEL).
    # GENERATION_MODEL = "gemini-flash-latest"
    # Vertex AI serves the released "gemini-3.1-flash-lite"; the ICDCN runs used "-preview" (AI Studio).
    GENERATION_MODEL = "gemini-3.1-flash-lite"
    # JUDGE_MODEL is the EVALUATION judge used by scripts/eval/*.
    JUDGE_MODEL = "gpt-5.4"
    # IN_LOOP_JUDGE_MODEL is the system-level judge that drives global retries
    # inside a run. It must not be the evaluation judge, or the system is
    # optimised against the metric it is scored on. Runs before Nov 2026 used
    # JUDGE_MODEL here.
    IN_LOOP_JUDGE_MODEL = GENERATION_MODEL
    # Evaluation judges for the journal experiments (journal_fixes/). Two different
    # Gemini models (OpenAI access unavailable in Nov 2026); neither is the system model.
    EVAL_JUDGES = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
    # Strong single-model baseline (replaces the GPT-5 baseline); not used as a judge.
    STRONG_BASELINE_MODEL = "gemini-3.5-flash"

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