"""
llm.py — one async entry point for the journal experiments.

Routes by model name (gpt* -> OpenAI, everything else -> Gemini), retries transient
errors (rate limits, 5xx) with backoff, and stops at once when the account is out of
credit so a long run does not burn through retries. Calls go through the SDKs that
orchestrator/usage_meter patches, so tokens are counted.
"""

from __future__ import annotations

import asyncio
import os
import random


class CreditsExhausted(RuntimeError):
    pass


async def generate(model: str, prompt: str, temperature: float = 0.0, retries: int = 6) -> str:
    for attempt in range(retries):
        try:
            if model.startswith("gpt"):
                from openai import AsyncOpenAI
                r = await AsyncOpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            import gemini_client as genai
            r = await asyncio.to_thread(genai.GenerativeModel(model).generate_content, prompt,
                                        generation_config={"temperature": temperature})
            if r.text is None:
                raise RuntimeError(f"empty response ({model})")
            return r.text
        except Exception as exc:
            msg = str(exc)
            if "credits are depleted" in msg or ("billing" in msg.lower() and "429" in msg):
                raise CreditsExhausted(msg[:200]) from exc
            transient = any(s in msg for s in ("429", "500", "502", "503", "504", "deadline", "Deadline", "timed out"))
            if not transient or attempt == retries - 1:
                raise
            await asyncio.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError("unreachable")
