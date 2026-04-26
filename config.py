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