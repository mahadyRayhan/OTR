"""
usage_meter.py — process-wide LLM call and token accounting.

Wraps every client the system uses so that tokens are counted no matter which
agent makes the call:
  - google.generativeai  GenerativeModel.generate_content   (legacy Gemini SDK)
  - google.genai         Models / AsyncModels.generate_content (new Gemini SDK)
  - openai               Completions / AsyncCompletions.create

Usage
-----
    from orchestrator import usage_meter
    usage_meter.install()          # idempotent; engine.py calls it on import
    usage_meter.reset()
    ... run one query ...
    usage = usage_meter.snapshot() # {"by_model": {...}, "calls": n, "input_tokens": n, "output_tokens": n}

Costs are deliberately NOT hard-coded: provider prices change. Pass a price
table to `estimate_cost()` when reporting dollars.
"""

from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_installed = False
_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})


def _record(model: str, input_tokens: int | None, output_tokens: int | None) -> None:
    with _lock:
        m = _by_model[model or "unknown"]
        m["calls"] += 1
        m["input_tokens"] += int(input_tokens or 0)
        m["output_tokens"] += int(output_tokens or 0)


def _gemini_usage(response) -> tuple[int, int]:
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return 0, 0
    inp = getattr(um, "prompt_token_count", 0) or 0
    out = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    return inp, out


def _openai_usage(response) -> tuple[int, int]:
    u = getattr(response, "usage", None)
    if u is None:
        return 0, 0
    return getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0


def install() -> None:
    """Patch the SDK entry points once. Safe to call repeatedly."""
    global _installed
    if _installed:
        return
    _installed = True

    try:
        import google.generativeai as legacy

        orig = legacy.GenerativeModel.generate_content

        def generate_content(self, *args, **kwargs):
            resp = orig(self, *args, **kwargs)
            _record(getattr(self, "model_name", "gemini"), *_gemini_usage(resp))
            return resp

        legacy.GenerativeModel.generate_content = generate_content
    except Exception:
        pass

    try:
        from google.genai import models as gmodels

        orig_sync = gmodels.Models.generate_content
        orig_async = gmodels.AsyncModels.generate_content

        def sync_gc(self, *args, **kwargs):
            resp = orig_sync(self, *args, **kwargs)
            _record(str(kwargs.get("model", "gemini")), *_gemini_usage(resp))
            return resp

        async def async_gc(self, *args, **kwargs):
            resp = await orig_async(self, *args, **kwargs)
            _record(str(kwargs.get("model", "gemini")), *_gemini_usage(resp))
            return resp

        gmodels.Models.generate_content = sync_gc
        gmodels.AsyncModels.generate_content = async_gc
    except Exception:
        pass

    try:
        from openai.resources.chat import completions as oc

        orig_osync = oc.Completions.create
        orig_oasync = oc.AsyncCompletions.create

        def osync(self, *args, **kwargs):
            resp = orig_osync(self, *args, **kwargs)
            _record(str(kwargs.get("model", "openai")), *_openai_usage(resp))
            return resp

        async def oasync(self, *args, **kwargs):
            resp = await orig_oasync(self, *args, **kwargs)
            _record(str(kwargs.get("model", "openai")), *_openai_usage(resp))
            return resp

        oc.Completions.create = osync
        oc.AsyncCompletions.create = oasync
    except Exception:
        pass


def reset() -> None:
    with _lock:
        _by_model.clear()


def snapshot() -> dict:
    with _lock:
        by_model = {k: dict(v) for k, v in _by_model.items()}
    return {
        "by_model": by_model,
        "calls": sum(v["calls"] for v in by_model.values()),
        "input_tokens": sum(v["input_tokens"] for v in by_model.values()),
        "output_tokens": sum(v["output_tokens"] for v in by_model.values()),
    }


def estimate_cost(snap: dict, prices_per_million: dict[str, tuple[float, float]]) -> float:
    """prices_per_million: {model_substring: (input_usd_per_M, output_usd_per_M)}."""
    total = 0.0
    for model, u in snap["by_model"].items():
        for key, (pin, pout) in prices_per_million.items():
            if key in model:
                total += u["input_tokens"] / 1e6 * pin + u["output_tokens"] / 1e6 * pout
                break
    return total
