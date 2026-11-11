"""
llm_cache.py — an opt-in response cache for every Gemini call (google-genai SDK, sync and async).

With the cache on, a call whose model, prompt and generation config were seen before returns the
stored response without contacting the API. Two uses:

  replay     rerun a recorded run with one input changed: nodes whose prompts are unchanged hit the
             cache, so only the part of the graph the change reaches costs anything (replay.py)
  repeats    rerunning an evaluation gives identical answers for free

Live web searches (PubMed, Semantic Scholar, Europe PMC) feed agent prompts, and their results drift,
so with the cache on they are cached too (memo / memo_async, keyed by source and query; empty results
are not stored, so a failed request is retried next time).

Turn it on with SCIWEAVER_LLM_CACHE=<sqlite path> (or =1 for ~/otr_local/llm_cache.sqlite) before the
engine is imported, or call install(path). It sits outside the usage meter: a cache hit is not
counted as tokens, and the run trace records it as {"cached": true}. Off by default, so normal runs
keep their sampling variation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import threading
from pathlib import Path

DEFAULT_PATH = Path.home() / "otr_local" / "llm_cache.sqlite"

_db: sqlite3.Connection | None = None
_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0}


def _key(kwargs: dict) -> str:
    def plain(v):
        if hasattr(v, "model_dump"):
            return v.model_dump(mode="json", exclude_none=True)
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v if isinstance(v, (str, int, float, bool, dict, type(None))) else repr(v)
    blob = json.dumps({k: plain(kwargs.get(k)) for k in ("model", "contents", "config")}, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()


def _get(key: str):
    with _lock:
        row = _db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    return row and row[0]


def _put(key: str, model: str, response) -> None:
    try:
        data = json.dumps(response.model_dump(mode="json", exclude_none=True))
    except Exception:
        return  # an unserialisable response is simply not cached
    with _lock:
        _db.execute("INSERT OR REPLACE INTO cache (key, model, response) VALUES (?, ?, ?)", (key, model, data))
        _db.commit()


def _revive(data: str):
    from google.genai import types
    return types.GenerateContentResponse.model_validate(json.loads(data))


def _note_hit(kwargs: dict, response) -> None:
    from orchestrator import trace
    _stats["hits"] += 1
    um = getattr(response, "usage_metadata", None)
    trace.record_llm_call(str(kwargs.get("model")), getattr(um, "prompt_token_count", 0) or 0,
                          getattr(um, "candidates_token_count", 0) or 0,
                          kwargs.get("contents") if isinstance(kwargs.get("contents"), str) else None)
    run = trace.current()
    if run is not None:  # mark the call just recorded as served from the cache
        calls = run.node(trace._node.get())["llm_calls"]
        if calls:
            calls[-1]["cached"] = True


def install(path: str | Path | None = None) -> None:
    """Open the cache and patch the SDK. Call after usage_meter.install(), so hits skip the meter."""
    global _db
    if _db is not None:
        return
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    _db = sqlite3.connect(str(path), check_same_thread=False)
    _db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, response TEXT)")
    _db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value BLOB)")

    from google.genai import models as gmodels
    orig_sync, orig_async = gmodels.Models.generate_content, gmodels.AsyncModels.generate_content

    def sync_gc(self, *args, **kwargs):
        key = _key(kwargs)
        hit = _get(key)
        if hit:
            resp = _revive(hit)
            _note_hit(kwargs, resp)
            return resp
        _stats["misses"] += 1
        resp = orig_sync(self, *args, **kwargs)
        _put(key, str(kwargs.get("model")), resp)
        return resp

    async def async_gc(self, *args, **kwargs):
        key = _key(kwargs)
        hit = _get(key)
        if hit:
            resp = _revive(hit)
            _note_hit(kwargs, resp)
            return resp
        _stats["misses"] += 1
        resp = await orig_async(self, *args, **kwargs)
        _put(key, str(kwargs.get("model")), resp)
        return resp

    gmodels.Models.generate_content = sync_gc
    gmodels.AsyncModels.generate_content = async_gc


def _call_key(namespace: str, key) -> str:
    return namespace + ":" + hashlib.sha256(json.dumps(key, sort_keys=True, default=repr).encode()).hexdigest()


def _call_get(k: str):
    with _lock:
        row = _db.execute("SELECT value FROM calls WHERE key = ?", (k,)).fetchone()
    return pickle.loads(row[0]) if row else None


def _call_put(k: str, value) -> None:
    if value:  # an empty result is often a failed request; do not freeze it
        with _lock:
            _db.execute("INSERT OR REPLACE INTO calls (key, value) VALUES (?, ?)", (k, pickle.dumps(value)))
            _db.commit()


def memo(namespace: str, key, fn, *args, **kwargs):
    """fn(*args, **kwargs), cached under (namespace, key) while the cache is on."""
    if _db is None:
        return fn(*args, **kwargs)
    k = _call_key(namespace, key)
    hit = _call_get(k)
    if hit is not None:
        return hit
    value = fn(*args, **kwargs)
    _call_put(k, value)
    return value


async def memo_async(namespace: str, key, fn, *args, **kwargs):
    """await fn(*args, **kwargs), cached under (namespace, key) while the cache is on."""
    if _db is None:
        return await fn(*args, **kwargs)
    k = _call_key(namespace, key)
    hit = _call_get(k)
    if hit is not None:
        return hit
    value = await fn(*args, **kwargs)
    _call_put(k, value)
    return value


def install_from_env() -> None:
    value = os.environ.get("SCIWEAVER_LLM_CACHE", "").strip()
    if value and value != "0":
        install(None if value == "1" else value)


def stats() -> dict:
    return dict(_stats, enabled=_db is not None)
