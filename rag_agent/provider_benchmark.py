"""
provider_benchmark.py — Head-to-head provider comparison for the Knowledge Discovery Agent.

Runs the same queries through multiple LLM providers and reports:
  - Latency per call
  - Token usage and estimated cost
  - JSON parse success rate
  - Topic / publication count
  - Side-by-side diff of responses

Run:
    python provider_benchmark.py                             # default providers
    python provider_benchmark.py --providers claude openai   # specific subset
    python provider_benchmark.py --query "CRISPR off-target" --providers claude gemini
    python provider_benchmark.py --component planner         # benchmark only the ReAct planner
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from llm_provider import get_provider, list_providers, LLMResponse, LLMProvider


# ─────────────────────────────────────────────────────────────────────────────
# Pricing table (USD per 1M tokens, as of March 2026 — update as needed)
# ─────────────────────────────────────────────────────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    # provider:model → {input, output} $/1M tokens
    "claude:claude-opus-4-5":           {"input": 15.0,  "output": 75.0},
    "claude:claude-sonnet-4-6":         {"input": 3.0,   "output": 15.0},
    "claude:claude-haiku-4-5-20251001": {"input": 0.8,   "output": 4.0},
    "openai:gpt-4o":                    {"input": 2.5,   "output": 10.0},
    "openai:gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "openai:gpt-4-turbo":               {"input": 10.0,  "output": 30.0},
    "openai:o1":                        {"input": 15.0,  "output": 60.0},
    "openai:o1-mini":                   {"input": 3.0,   "output": 12.0},
    "openai:o3-mini":                   {"input": 1.1,   "output": 4.4},
    "gemini:gemini-2.0-flash":          {"input": 0.10,  "output": 0.40},
    "gemini:gemini-2.0-flash-lite":     {"input": 0.075, "output": 0.30},
    "gemini:gemini-1.5-pro":            {"input": 1.25,  "output": 5.0},
    "gemini:gemini-1.5-flash":          {"input": 0.075, "output": 0.30},
    "ollama:*":                         {"input": 0.0,   "output": 0.0},
}


def estimate_cost(response: LLMResponse) -> float:
    """Return estimated USD cost for a single response."""
    key = f"{response.provider}:{response.model}"
    pricing = PRICING.get(key) or PRICING.get(f"{response.provider}:*") or {"input": 0, "output": 0}
    cost = (response.input_tokens * pricing["input"] + response.output_tokens * pricing["output"]) / 1_000_000
    return round(cost, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Test prompts (representative of real agent calls)
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a ReAct agent planner for scientific literature discovery.
Return ONLY valid JSON with fields: thought, action, action_input, stop.
action must be one of: get_topic_landscape, search_publications, get_tools."""

EXPLANATION_SYSTEM = """You are a scientific literature assistant.
Return ONLY valid JSON with field: relevance (one sentence, max 20 words)."""

TEST_PROMPTS: dict[str, dict[str, Any]] = {
    "planner": {
        "system": PLANNER_SYSTEM,
        "messages": [
            {"role": "user", "content": (
                "Query: synaptic plasticity in hippocampal neurons\n"
                "Steps taken so far: None yet.\n"
                "Decide the next action."
            )}
        ],
        "max_tokens": 256,
        "expect_json": True,
        "expect_keys": ["thought", "action", "stop"],
    },
    "topic_explanation": {
        "system": EXPLANATION_SYSTEM,
        "messages": [
            {"role": "user", "content": (
                "Query: synaptic plasticity in hippocampal neurons\n"
                "Complexity level: introductory\n"
                "Topic: Spike-Timing Dependent Plasticity (STDP)\n"
                "Representative keywords: STDP, hebbian, spike, calcium, LTP\n"
                "Write a single-sentence explanation of why this topic is relevant.\n"
                'Return: {"relevance": "..."}'
            )}
        ],
        "max_tokens": 80,
        "expect_json": True,
        "expect_keys": ["relevance"],
    },
    "paper_explanation": {
        "system": EXPLANATION_SYSTEM,
        "messages": [
            {"role": "user", "content": (
                "Query: synaptic plasticity hippocampal neurons\n"
                "Complexity level: introductory\n"
                "User background: graduate student new to neuroscience\n"
                "Paper title: Spike-timing dependent plasticity: a comprehensive overview\n"
                "Authors: Caporale & Dan\n"
                "Journal: Neuron, 2008\n"
                "Abstract: This review covers the mechanisms of STDP...\n"
                "Write a single-sentence explanation of why this paper is useful.\n"
                'Return: {"relevance": "..."}'
            )}
        ],
        "max_tokens": 80,
        "expect_json": True,
        "expect_keys": ["relevance"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProviderResult:
    provider_label: str
    model: str
    prompt_name: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    json_valid: bool
    missing_keys: list[str]
    response_snippet: str
    error: str = ""


@dataclass
class BenchmarkSummary:
    providers_tested: list[str]
    prompts_tested: list[str]
    results: list[ProviderResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class ProviderBenchmark:
    def __init__(self, provider_names: list[str]) -> None:
        self.providers: list[tuple[str, LLMProvider]] = []
        for name in provider_names:
            try:
                provider = get_provider(name)
                label = f"{provider.provider_name}:{provider.model}"
                self.providers.append((label, provider))
            except Exception as exc:
                print(f"  ⚠  Could not load provider '{name}': {exc}")

    async def run(self, component: str = "all") -> BenchmarkSummary:
        prompts = (
            {k: v for k, v in TEST_PROMPTS.items() if k == component}
            if component != "all"
            else TEST_PROMPTS
        )

        summary = BenchmarkSummary(
            providers_tested=[label for label, _ in self.providers],
            prompts_tested=list(prompts.keys()),
        )

        for prompt_name, prompt_cfg in prompts.items():
            print(f"\n  Prompt: {prompt_name}")
            print(f"  {'Provider':<40} {'Latency':>8} {'Tokens':>8} {'Cost':>10} {'JSON':>6}")
            print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")

            # Run all providers concurrently for the same prompt
            tasks = [
                self._run_single(label, provider, prompt_name, prompt_cfg)
                for label, provider in self.providers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, ProviderResult):
                    summary.results.append(result)
                    status = "✓" if result.json_valid and not result.error else "✗"
                    cost_str = f"${result.cost_usd:.6f}" if result.cost_usd > 0 else "free"
                    tokens_str = f"{result.input_tokens}+{result.output_tokens}"
                    print(f"  {result.provider_label:<40} {result.latency_s:>7.2f}s {tokens_str:>8} {cost_str:>10} {status:>6}")
                    if result.error:
                        print(f"    ✗ Error: {result.error}")
                    if result.missing_keys:
                        print(f"    ✗ Missing JSON keys: {result.missing_keys}")
                    print(f"    → {result.response_snippet}")

        return summary

    async def _run_single(
        self,
        label: str,
        provider: LLMProvider,
        prompt_name: str,
        cfg: dict[str, Any],
    ) -> ProviderResult:
        try:
            t0 = time.perf_counter()
            response: LLMResponse = await provider.complete(
                messages=cfg["messages"],
                system=cfg.get("system", ""),
                max_tokens=cfg.get("max_tokens", 256),
            )
            latency = time.perf_counter() - t0
            cost = estimate_cost(response)

            # Validate JSON
            json_valid = False
            missing_keys: list[str] = []
            if cfg.get("expect_json"):
                try:
                    import re
                    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE).strip()
                    parsed = json.loads(clean)
                    json_valid = True
                    expected_keys = cfg.get("expect_keys", [])
                    missing_keys = [k for k in expected_keys if k not in parsed]
                except Exception:
                    json_valid = False
            else:
                json_valid = bool(response.text.strip())

            snippet = response.text.strip().replace("\n", " ")[:100]

            return ProviderResult(
                provider_label=label,
                model=response.model,
                prompt_name=prompt_name,
                latency_s=round(latency, 3),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost,
                json_valid=json_valid,
                missing_keys=missing_keys,
                response_snippet=snippet,
            )
        except Exception as exc:
            return ProviderResult(
                provider_label=label,
                model=getattr(provider, "model", "?"),
                prompt_name=prompt_name,
                latency_s=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                json_valid=False,
                missing_keys=[],
                response_snippet="",
                error=str(exc)[:120],
            )

    def print_summary_table(self, summary: BenchmarkSummary) -> None:
        """Print aggregated stats across all prompts per provider."""
        from collections import defaultdict

        agg: dict[str, list[ProviderResult]] = defaultdict(list)
        for r in summary.results:
            agg[r.provider_label].append(r)

        print(f"\n{'─'*72}")
        print(f"  {'AGGREGATE SUMMARY':^68}")
        print(f"{'─'*72}")
        print(f"  {'Provider':<40} {'Avg lat':>8} {'JSON%':>6} {'Avg cost':>12}")
        print(f"  {'─'*40} {'─'*8} {'─'*6} {'─'*12}")

        for label, results in agg.items():
            good = [r for r in results if not r.error]
            if not good:
                print(f"  {label:<40} {'—':>8} {'0%':>6} {'—':>12}")
                continue
            avg_latency = statistics.mean(r.latency_s for r in good)
            json_pct = int(100 * sum(1 for r in good if r.json_valid) / len(good))
            avg_cost = statistics.mean(r.cost_usd for r in good)
            cost_str = f"${avg_cost:.6f}" if avg_cost > 0 else "free"
            print(f"  {label:<40} {avg_latency:>7.2f}s {json_pct:>5}% {cost_str:>12}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROVIDERS = ["claude", "openai", "gemini"]


async def _main(providers: list[str], component: str) -> None:
    print(f"\n  Knowledge Discovery Agent — Provider Benchmark")
    print(f"  Providers: {', '.join(providers)}")
    print(f"  Component: {component}")
    print(f"  {'─'*50}\n")

    bench = ProviderBenchmark(providers)
    if not bench.providers:
        print("  No providers could be loaded. Check API keys in .env")
        return

    summary = await bench.run(component=component)
    bench.print_summary_table(summary)

    # Save results
    Path("logs").mkdir(exist_ok=True)
    out = Path(f"logs/benchmark_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Results saved to {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Provider Benchmark")
    parser.add_argument(
        "--providers", nargs="+", default=DEFAULT_PROVIDERS,
        help="Providers to test: claude openai gemini ollama (or full model strings)"
    )
    parser.add_argument(
        "--component", default="all",
        choices=["all", "planner", "topic_explanation", "paper_explanation"],
        help="Which agent component to benchmark"
    )
    args = parser.parse_args()
    asyncio.run(_main(args.providers, args.component))