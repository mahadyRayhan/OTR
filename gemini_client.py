"""
gemini_client.py — the slice of the legacy `google.generativeai` API that this project uses
(configure, GenerativeModel(name).generate_content(prompt, generation_config=...)), served by the
unified `google-genai` SDK so the same code runs on either backend:

  Vertex AI (Cloud billing / credits)  GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT,
                                       GOOGLE_CLOUD_LOCATION, GOOGLE_APPLICATION_CREDENTIALS
  Gemini Developer API (AI Studio)     GOOGLE_API_KEY

Responses are google-genai responses; `.text` and `.usage_metadata.{prompt,candidates}_token_count`
match the legacy names, so callers need no other change. Calls go through
google.genai Models.generate_content, which orchestrator/usage_meter patches.

Usage: `import gemini_client as genai` in place of `import google.generativeai as genai`.
"""

from __future__ import annotations

import json
import os
import threading

from google import genai as _genai
from google.genai import types as _types

# Preview names that Vertex serves under their released name.
_VERTEX_ALIASES = {"gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite"}

_client = None
_lock = threading.Lock()


def use_vertex() -> bool:
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


def client() -> _genai.Client:
    """One shared client; Vertex when configured, never a silent fallback to the API key."""
    global _client
    with _lock:
        if _client is None:
            if use_vertex():
                _client = _genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
            else:
                _client = _genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        return _client


def resolve(model_name: str) -> str:
    name = model_name.replace("models/", "")
    return _VERTEX_ALIASES.get(name, name) if use_vertex() else name


def configure(api_key: str | None = None, **_) -> None:
    """Kept for call-site compatibility; credentials come from the environment."""


def parse_json(text: str):
    """The first JSON object or array in a model reply, ignoring code fences and anything after it.

    Models append text after the JSON when the user's query carries its own output instruction
    (e.g. "end with a line 'ANSWER: ...'"), which json.loads rejects as "Extra data".
    """
    text = (text or "").replace("```json", "").replace("```", "").strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    return json.JSONDecoder().raw_decode(text[min(starts) if starts else 0:])[0]


class GenerativeModel:
    def __init__(self, model_name: str):
        self.model_name = resolve(model_name)

    def generate_content(self, contents, generation_config: dict | None = None, **_):
        cfg = _types.GenerateContentConfig(
            **(generation_config or {}),
            automatic_function_calling=_types.AutomaticFunctionCallingConfig(disable=True))
        return client().models.generate_content(model=self.model_name, contents=contents, config=cfg)
