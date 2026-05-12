"""
llm_provider.py — Unified LLM provider abstraction.

Supports:
  • Anthropic Claude   (claude-opus-4-5, claude-sonnet-4-6, claude-haiku-4-5, …)
  • OpenAI GPT         (gpt-4o, gpt-4o-mini, gpt-4-turbo, …)
  • Google Gemini      (gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, …)
  • Ollama             (llama3, mistral, phi3, … — local inference)

All providers expose the same async interface:
    complete(messages, system, max_tokens) -> str

Usage
─────
    from llm_provider import get_provider

    provider = get_provider("claude")          # from registry
    provider = get_provider("gpt-4o-mini")    # auto-detects OpenAI
    response = await provider.complete(
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful assistant.",
        max_tokens=256,
    )
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    provider_name: str = "base"
    default_model: str = ""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        self.model = model or self.default_model
        self._kwargs = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a completion request and return a normalised LLMResponse."""

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1000,
    ) -> dict[str, Any]:
        """Complete and parse the response as JSON, stripping markdown fences."""
        response = await self.complete(messages, system=system, max_tokens=max_tokens)
        return _parse_json(response.text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Claude (Anthropic)
# ─────────────────────────────────────────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    """Anthropic Claude — uses the Messages API."""

    provider_name = "claude"
    default_model = "claude-sonnet-4-6"

    # Maps friendly short names → full model strings
    MODEL_ALIASES: dict[str, str] = {
        "opus":   "claude-opus-4-5",
        "sonnet": "claude-sonnet-4-6",
        "haiku":  "claude-haiku-4-5-20251001",
        "opus-4-5":    "claude-opus-4-5",
        "sonnet-4-6":  "claude-sonnet-4-6",
    }

    def __init__(self, model: str | None = None, api_key: str = "", **kwargs: Any) -> None:
        resolved = self.MODEL_ALIASES.get(model or "", model or self.default_model)
        super().__init__(resolved, **kwargs)
        self._api_key = api_key

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        api_key = self._api_key
        if not api_key:
            # from config import settings
            from .rag_config import settings
            api_key = settings.anthropic_api_key

        client = anthropic.AsyncAnthropic(api_key=api_key)
        t0 = time.perf_counter()

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = await client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        text = response.content[0].text if response.content else ""

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=round(latency, 3),
            raw=response,
        )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI GPT
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI GPT — uses the Chat Completions API."""

    provider_name = "openai"
    default_model = "gpt-4o-mini"

    MODEL_ALIASES: dict[str, str] = {
        "gpt4o":       "gpt-4o",
        "gpt4o-mini":  "gpt-4o-mini",
        "gpt4-turbo":  "gpt-4-turbo",
        "gpt4":        "gpt-4",
        "gpt3.5":      "gpt-3.5-turbo",
        "o1":          "o1",
        "o1-mini":     "o1-mini",
        "o3-mini":     "o3-mini",
    }

    def __init__(self, model: str | None = None, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        resolved = self.MODEL_ALIASES.get(model or "", model or self.default_model)
        super().__init__(resolved, **kwargs)
        self._api_key = api_key
        self._base_url = base_url  # For Azure OpenAI or custom endpoints

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        api_key = self._api_key
        if not api_key:
            # from config import settings
            from .rag_config import settings
            api_key = settings.openai_api_key

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = AsyncOpenAI(**client_kwargs)
        t0 = time.perf_counter()

        # System message prepended for OpenAI
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = await client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.perf_counter() - t0
        text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_s=round(latency, 3),
            raw=response,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini
# ─────────────────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    provider_name = "gemini"
    default_model = "gemini-2.0-flash-lite"

    MODEL_ALIASES: dict[str, str] = {
        "flash":      "gemini-2.0-flash-lite",
        "flash-lite": "gemini-2.0-flash-lite",
        "pro":        "gemini-1.5-pro",
    }

    def __init__(self, model=None, api_key="", **kwargs):
        resolved = self.MODEL_ALIASES.get(model or "", model or self.default_model)
        super().__init__(resolved, **kwargs)
        self._api_key = api_key

    async def complete(self, messages, system="", max_tokens=1000, temperature=0.0) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")

        api_key = self._api_key
        if not api_key:
            # from config import settings
            from .rag_config import settings
            api_key = settings.gemini_api_key

        client = genai.Client(api_key=api_key)
        t0 = time.perf_counter()

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        model_name = self.model.replace("models/", "")
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        latency = time.perf_counter() - t0
        text = response.text or ""
        usage = response.usage_metadata

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            latency_s=round(latency, 3),
            raw=response,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Ollama (local inference)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Ollama local inference — compatible with any pulled model."""

    provider_name = "ollama"
    default_model = "llama3.2"

    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434", **kwargs: Any) -> None:
        super().__init__(model or self.default_model, **kwargs)
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp not installed. Run: pip install aiohttp")

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        t0 = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                data = await r.json(content_type=None)
        latency = time.perf_counter() - t0

        text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            latency_s=round(latency, 3),
            raw=data,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry & factory
# ─────────────────────────────────────────────────────────────────────────────

# Maps provider name / model prefix → provider class
_PROVIDER_MAP: dict[str, type[LLMProvider]] = {
    # Explicit names
    "claude":   ClaudeProvider,
    "anthropic": ClaudeProvider,
    "openai":   OpenAIProvider,
    "gpt":      OpenAIProvider,
    "gemini":   GeminiProvider,
    "google":   GeminiProvider,
    "ollama":   OllamaProvider,
    "local":    OllamaProvider,
}

# Model prefix auto-detection
_PREFIX_MAP: list[tuple[str, type[LLMProvider]]] = [
    ("claude-",   ClaudeProvider),
    ("gpt-",      OpenAIProvider),
    ("o1",        OpenAIProvider),
    ("o3",        OpenAIProvider),
    ("gemini-",   GeminiProvider),
    ("llama",     OllamaProvider),
    ("mistral",   OllamaProvider),
    ("phi",       OllamaProvider),
    ("qwen",      OllamaProvider),
]


def get_provider(name_or_model: str, **kwargs: Any) -> LLMProvider:
    """
    Resolve a provider by name or model string.

    Examples
    ────────
        get_provider("claude")                 → ClaudeProvider(sonnet)
        get_provider("claude", model="haiku")  → ClaudeProvider(haiku)
        get_provider("gpt-4o-mini")            → OpenAIProvider(gpt-4o-mini)
        get_provider("gemini-2.0-flash")       → GeminiProvider(gemini-2.0-flash)
        get_provider("ollama", model="phi3")   → OllamaProvider(phi3)
    """
    key = name_or_model.lower().strip()

    # 1. Exact provider name match
    if key in _PROVIDER_MAP:
        return _PROVIDER_MAP[key](**kwargs)

    # 2. Model string prefix detection
    for prefix, cls in _PREFIX_MAP:
        if key.startswith(prefix):
            return cls(model=name_or_model, **kwargs)

    raise ValueError(
        f"Unknown provider or model: '{name_or_model}'. "
        f"Valid providers: {list(_PROVIDER_MAP.keys())}. "
        f"Or pass a full model string like 'gpt-4o', 'claude-sonnet-4-6', 'gemini-2.0-flash'."
    )


def list_providers() -> dict[str, dict[str, Any]]:
    """Return metadata about all available providers and their default models."""
    return {
        "claude": {
            "class": "ClaudeProvider",
            "default_model": ClaudeProvider.default_model,
            "aliases": list(ClaudeProvider.MODEL_ALIASES.keys()),
            "models": ["claude-opus-4-5", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            "env_key": "ANTHROPIC_API_KEY",
        },
        "openai": {
            "class": "OpenAIProvider",
            "default_model": OpenAIProvider.default_model,
            "aliases": list(OpenAIProvider.MODEL_ALIASES.keys()),
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini"],
            "env_key": "OPENAI_API_KEY",
        },
        "gemini": {
            "class": "GeminiProvider",
            "default_model": GeminiProvider.default_model,
            "aliases": list(GeminiProvider.MODEL_ALIASES.keys()),
            "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
            "env_key": "GEMINI_API_KEY",
        },
        "ollama": {
            "class": "OllamaProvider",
            "default_model": OllamaProvider.default_model,
            "aliases": [],
            "models": ["llama3.2", "llama3.1", "mistral", "phi3", "qwen2.5", "deepseek-r1"],
            "env_key": "OLLAMA_BASE_URL (optional)",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helper (shared across providers)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON from LLM output."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s — raw: %s", exc, clean[:120])
        return {}