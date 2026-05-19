"""
rag_config.py — Centralised configuration via Pydantic Settings.

All values are loaded from environment variables (or the .env file).
Import the singleton `settings` anywhere in the project.
"""

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the rag_agent/ directory — used to anchor data paths
# so the index always lives inside the agent folder regardless of CWD.
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Active LLM Provider ──────────────────────────────────────────────────
    llm_provider: str = Field(
        "gemini",
        description="Active LLM provider: claude | openai | gemini | ollama",
    )
    llm_model: str = Field(
        "",
        description="Override model for the active provider. Empty = use provider default.",
    )

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", description="Anthropic API key")
    anthropic_model: str = Field(
        "claude-sonnet-4-6",
        description="Claude model (used when llm_provider=claude)",
    )

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", description="OpenAI API key")
    openai_model: str = Field(
        "gpt-4o-mini",
        description="OpenAI model (used when llm_provider=openai)",
    )
    openai_base_url: str = Field(
        "",
        description="Optional custom base URL for Azure OpenAI or local proxies",
    )

    # ── Google Gemini ─────────────────────────────────────────────────────────
    gemini_api_key: str = Field("", description="Google Gemini API key")
    gemini_model: str = Field(
        "gemini-3.1-flash-lite-preview",
        description="Gemini model (used when llm_provider=gemini)",
    )

    # ── Ollama (local) ────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        "http://localhost:11434",
        description="Ollama server base URL (used when llm_provider=ollama)",
    )
    ollama_model: str = Field(
        "llama3.2",
        description="Ollama model (used when llm_provider=ollama)",
    )

    # ── External APIs ────────────────────────────────────────────────────────
    semantic_scholar_api_key: str = Field("", description="Semantic Scholar API key (optional)")
    ncbi_api_key: str = Field("", description="NCBI Entrez API key")
    ncbi_email: str = Field("researcher@example.com", description="Required by NCBI TOS")

    # ── Embedding Model ──────────────────────────────────────────────────────
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="HuggingFace model ID used by ChromaDB and the cross-encoder device",
    )
    embedding_device: str = Field("cpu", description="torch device for cross-encoder: cpu | cuda | mps")

    # ── ChromaDB Vector Store ────────────────────────────────────────────────
    chroma_db_path: str = Field(
        default_factory=lambda: os.path.join(_AGENT_DIR, "chroma_db"),
        description="Path to ChromaDB persistent store for RAG papers",
    )
    chroma_collection_name: str = Field(
        "rag_paper_vault",
        description="ChromaDB collection name for indexed papers",
    )

    # ── Topic Modeling ───────────────────────────────────────────────────────
    bertopic_model_path: str = Field("./models/bertopic_model", description="Path to saved BERTopic model")
    topic_min_cluster_size: int = Field(10, description="HDBSCAN minimum cluster size")
    topic_n_gram_range: tuple[int, int] = Field((1, 3), description="N-gram range for c-TF-IDF")

    # ── Cross-Encoder Re-Ranking ─────────────────────────────────────────────
    cross_encoder_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for stage-2 re-ranking",
    )
    rerank_top_k: int = Field(20, description="Candidates passed to cross-encoder from ChromaDB")
    final_top_k: int = Field(5, description="Final publications returned to the user")
    confidence_threshold: float = Field(
        -6.0,
        description=(
            "Minimum cross-encoder logit score to include a result (ms-marco space). "
            "Typical ranges: >3 strong match, 0–3 relevant, -3–0 borderline, <-6 likely noise. "
            "Lowered from -2.0: biomedical paper chunks score -2 to -5 on ms-marco CE even when "
            "correctly matched (CE trained on web search, not academic text). "
            "Falls back to top-K if all papers score below this threshold."
        ),
    )

    # ── ReAct Agent ──────────────────────────────────────────────────────────
    max_react_steps: int = Field(8, description="Hard cap on ReAct chain depth")

    def build_llm_provider(self, provider_override=None):
        from .llm_provider import get_provider
        name = provider_override or self.llm_provider
        model_override = self.llm_model or None
        if name in ("claude", "anthropic"):
            return get_provider("claude", api_key=self.anthropic_api_key, model=model_override or self.anthropic_model)
        if name in ("openai", "gpt"):
            kw = dict(api_key=self.openai_api_key, model=model_override or self.openai_model)
            if self.openai_base_url: kw["base_url"] = self.openai_base_url
            return get_provider("openai", **kw)
        if name in ("gemini", "google"):
            return get_provider("gemini", api_key=self.gemini_api_key, model=model_override or self.gemini_model)
        if name in ("ollama", "local"):
            return get_provider("ollama", base_url=self.ollama_base_url, model=model_override or self.ollama_model)
        return get_provider(name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()


# Convenience alias
settings = get_settings()
