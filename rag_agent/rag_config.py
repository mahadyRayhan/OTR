"""
rag_config.py — Centralised configuration via Pydantic Settings.

All values are loaded from environment variables (or the .env file).
Import the singleton `settings` anywhere in the project.

Fix: FAISS / BERTopic paths are now resolved relative to this file's own
directory (OTR/rag_agent/) so they are correct regardless of the working
directory the orchestrator is launched from.
"""

import os as _os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the rag_agent/ package directory.
# __file__ is always OTR/rag_agent/rag_config.py, so dirname gives OTR/rag_agent/
_RAG_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _rag_path(*parts: str) -> str:
    """Return an absolute path anchored at the rag_agent/ directory."""
    return _os.path.join(_RAG_DIR, *parts)


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
        "gemini-2.5-flash",
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
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        description="HuggingFace model ID for PubMedBERT sentence embeddings",
    )
    embedding_batch_size: int = Field(32, description="Batch size for embedding inference")
    embedding_device: str = Field("cpu", description="torch device: cpu | cuda | mps")

    # ── FAISS Vector Store ───────────────────────────────────────────────────
    # Paths are anchored to rag_agent/ so they resolve correctly whether the
    # process is launched from OTR/rag_agent/ or OTR/scripts/ or anywhere else.
    faiss_index_path: str = Field(
        default_factory=lambda: _rag_path("data", "faiss_index"),
        description="Absolute path to persisted FAISS index",
    )
    faiss_metadata_path: str = Field(
        default_factory=lambda: _rag_path("data", "faiss_metadata.pkl"),
        description="Absolute path to pickled metadata store",
    )

    # ── Topic Modeling ───────────────────────────────────────────────────────
    bertopic_model_path: str = Field(
        default_factory=lambda: _rag_path("models", "bertopic_model"),
        description="Absolute path to saved BERTopic model",
    )
    topic_min_cluster_size: int = Field(10, description="HDBSCAN minimum cluster size")
    topic_n_gram_range: tuple[int, int] = Field((1, 3), description="N-gram range for c-TF-IDF")

    # ── Cross-Encoder Re-Ranking ─────────────────────────────────────────────
    cross_encoder_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for stage-2 re-ranking",
    )
    rerank_top_k: int = Field(20, description="Candidates passed to cross-encoder from FAISS")
    final_top_k: int = Field(5, description="Final publications returned to the user")

    # ── ReAct Agent ──────────────────────────────────────────────────────────
    max_react_steps: int = Field(8, description="Hard cap on ReAct chain depth")
    react_stop_threshold: float = Field(
        0.85, description="Stop the chain when top-result confidence exceeds this"
    )
    confidence_threshold: float = Field(0.50, description="Minimum score to include a result")

    # ── Application ──────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0")
    app_port: int = Field(8000)
    app_debug: bool = Field(False)
    log_level: str = Field("INFO")
    log_path: str = Field("./logs/app.log")

    # ── Safety ───────────────────────────────────────────────────────────────
    max_query_length: int = Field(512, description="Hard limit on incoming query characters")
    pii_detection_enabled: bool = Field(True, description="Strip PII before forwarding to LLM")

    @property
    def is_gpu(self) -> bool:
        return self.embedding_device in ("cuda", "mps")

    def build_llm_provider(self, provider_override=None):
        from .llm_provider import get_provider
        name = provider_override or self.llm_provider
        model_override = self.llm_model or None
        if name in ("claude", "anthropic"):
            return get_provider("claude", api_key=self.anthropic_api_key, model=model_override or self.anthropic_model)
        if name in ("openai", "gpt"):
            kw = dict(api_key=self.openai_api_key, model=model_override or self.openai_model)
            if self.openai_base_url:
                kw["base_url"] = self.openai_base_url
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