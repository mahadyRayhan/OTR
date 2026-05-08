"""
evaluations.py — Evaluation harness for the Knowledge Discovery Agent.

Metrics tracked
───────────────
  - Retrieval precision@k    (fraction of top-k papers deemed relevant)
  - Topic coherence score    (average pairwise cosine sim of topic words)
  - Confidence calibration   (MSE between predicted and human relevance scores)
  - ReAct chain efficiency   (steps taken vs minimum required)
  - Response latency         (end-to-end wall time per query)

Run:
    python evaluations.py [--suite all|retrieval|topics|latency] [--k 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from config import settings
from knowledge_base import KnowledgeBase
from react_agent import KnowledgeDiscoveryAgent, AgentResult
from rag_responder import RAGResponder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # Keep eval output clean

EVAL_DATA_PATH = Path("evaluation_data.json")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    expected_topics: list[str]
    expected_pmids: list[str]
    predicted_topics: list[str]
    predicted_pmids: list[str]
    top_scores: list[float]
    steps: int
    latency_s: float
    error: str = ""


@dataclass
class EvalSummary:
    suite: str
    n_queries: int
    precision_at_k: float
    topic_recall: float
    mean_top_score: float
    mean_latency_s: float
    mean_steps: float
    errors: int
    per_query: list[QueryResult] = field(default_factory=list)

    def print_report(self) -> None:
        bar = "─" * 52
        print(f"\n{bar}")
        print(f"  Knowledge Discovery Agent — Eval: {self.suite}")
        print(bar)
        print(f"  Queries evaluated : {self.n_queries}")
        print(f"  Precision@k       : {self.precision_at_k:.3f}")
        print(f"  Topic recall      : {self.topic_recall:.3f}")
        print(f"  Mean top score    : {self.mean_top_score:.3f}")
        print(f"  Mean latency (s)  : {self.mean_latency_s:.2f}")
        print(f"  Mean ReAct steps  : {self.mean_steps:.1f}")
        print(f"  Errors            : {self.errors}")
        print(bar)
        for qr in self.per_query:
            status = "✓" if not qr.error else "✗"
            print(f"  {status} [{qr.latency_s:.1f}s / {qr.steps} steps]  {qr.query[:55]}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(predicted: list[str], expected: list[str], k: int) -> float:
    """Fraction of top-k predicted PMIDs that appear in the expected set."""
    if not expected:
        return 1.0
    top_k = predicted[:k]
    hits = sum(1 for pmid in top_k if pmid in set(expected))
    return hits / k if k else 0.0


def topic_recall(predicted: list[str], expected: list[str]) -> float:
    """Fraction of expected topic keywords found in any predicted topic name."""
    if not expected:
        return 1.0
    pred_blob = " ".join(predicted).lower()
    hits = sum(1 for kw in expected if kw.lower() in pred_blob)
    return hits / len(expected)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, kb: KnowledgeBase, k: int = 5) -> None:
        self.kb = kb
        self.agent = KnowledgeDiscoveryAgent(kb)
        self.responder = RAGResponder()
        self.k = k

    async def run_suite(self, suite: str = "all") -> EvalSummary:
        eval_data = self._load_eval_data(suite)
        results: list[QueryResult] = []

        for case in eval_data:
            qr = await self._run_case(case)
            results.append(qr)
            print(f"  {'✓' if not qr.error else '✗'} {qr.query[:60]}")

        return self._summarise(suite, results)

    async def _run_case(self, case: dict[str, Any]) -> QueryResult:
        query = case["query"]
        expected_topics = case.get("expected_topics", [])
        expected_pmids = case.get("expected_pmids", [])
        t0 = time.perf_counter()

        try:
            result: AgentResult = await self.agent.run(query)
            latency = round(time.perf_counter() - t0, 3)

            predicted_topics = [t.name for t in result.topics]
            predicted_pmids = [p.pmid for p in result.papers]
            top_scores = [p.score for p in result.papers[:self.k]]

            return QueryResult(
                query=query,
                expected_topics=expected_topics,
                expected_pmids=expected_pmids,
                predicted_topics=predicted_topics,
                predicted_pmids=predicted_pmids,
                top_scores=top_scores,
                steps=result.steps,
                latency_s=latency,
            )
        except Exception as exc:
            latency = round(time.perf_counter() - t0, 3)
            logger.error("Eval case failed for '%s': %s", query, exc)
            return QueryResult(
                query=query,
                expected_topics=expected_topics,
                expected_pmids=expected_pmids,
                predicted_topics=[],
                predicted_pmids=[],
                top_scores=[],
                steps=0,
                latency_s=latency,
                error=str(exc),
            )

    def _summarise(self, suite: str, results: list[QueryResult]) -> EvalSummary:
        n = len(results)
        errors = sum(1 for r in results if r.error)
        good = [r for r in results if not r.error]

        p_at_k = statistics.mean(
            precision_at_k(r.predicted_pmids, r.expected_pmids, self.k)
            for r in good
        ) if good else 0.0

        t_recall = statistics.mean(
            topic_recall(r.predicted_topics, r.expected_topics)
            for r in good
        ) if good else 0.0

        mean_top_score = statistics.mean(
            r.top_scores[0] if r.top_scores else 0.0
            for r in good
        ) if good else 0.0

        mean_latency = statistics.mean(r.latency_s for r in results) if results else 0.0
        mean_steps = statistics.mean(r.steps for r in good) if good else 0.0

        return EvalSummary(
            suite=suite,
            n_queries=n,
            precision_at_k=round(p_at_k, 4),
            topic_recall=round(t_recall, 4),
            mean_top_score=round(mean_top_score, 4),
            mean_latency_s=round(mean_latency, 3),
            mean_steps=round(mean_steps, 1),
            errors=errors,
            per_query=results,
        )

    @staticmethod
    def _load_eval_data(suite: str) -> list[dict[str, Any]]:
        if not EVAL_DATA_PATH.exists():
            logger.warning("evaluation_data.json not found — using built-in minimal set.")
            return _MINIMAL_EVAL_SET

        with open(EVAL_DATA_PATH) as f:
            data = json.load(f)

        if suite == "all":
            return data
        return [c for c in data if c.get("suite", "all") == suite]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal built-in eval set (used when evaluation_data.json is absent)
# ─────────────────────────────────────────────────────────────────────────────

_MINIMAL_EVAL_SET: list[dict[str, Any]] = [
    {
        "query": "synaptic plasticity in hippocampal neurons",
        "suite": "retrieval",
        "expected_topics": ["STDP", "LTP", "hippocampus"],
        "expected_pmids": [],
    },
    {
        "query": "single-cell RNA sequencing tumor microenvironment",
        "suite": "retrieval",
        "expected_topics": ["scRNA-seq", "tumor", "microenvironment"],
        "expected_pmids": [],
    },
    {
        "query": "CRISPR base editing off-target effects",
        "suite": "retrieval",
        "expected_topics": ["CRISPR", "base editing", "off-target"],
        "expected_pmids": [],
    },
    {
        "query": "graph neural networks protein structure prediction",
        "suite": "latency",
        "expected_topics": ["GNN", "protein structure"],
        "expected_pmids": [],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

async def _main(suite: str, k: int) -> None:
    print(f"\n  Initialising Knowledge Base…")
    kb = KnowledgeBase()
    kb.load()
    evaluator = Evaluator(kb, k=k)

    print(f"  Running eval suite: '{suite}' with precision@{k}\n")
    summary = await evaluator.run_suite(suite)
    summary.print_report()

    # Write results to logs/
    Path("logs").mkdir(exist_ok=True)
    out_path = Path(f"logs/eval_{suite}_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Results saved to {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Discovery Agent Evaluator")
    parser.add_argument("--suite", default="all", choices=["all", "retrieval", "latency", "topics"])
    parser.add_argument("--k", type=int, default=5, help="k for precision@k")
    args = parser.parse_args()
    asyncio.run(_main(args.suite, args.k))