"""
rag_responder.py — LLM explanation layer (the final synthesis step).

After the ReAct agent has retrieved topics and papers, this module:
1. Constructs a rich context document from the retrieved knowledge.
2. Calls Claude to generate personalised "Why relevant / Why for you" explanations
   tailored to the inferred user expertise level.
3. Returns a fully annotated AgentResult ready for the API response.

The explanation layer is the module that converts raw retrieval scores into
natural language that helps a researcher understand *why* a paper matters to them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from config import settings
from llm_provider import LLMProvider
from knowledge_base import Paper, Topic
from react_agent import AgentResult, TraceStep

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

EXPLANATION_SYSTEM = """You are a scientific literature assistant helping researchers discover
relevant papers and topics. Your role is to personalise retrieval results with clear, concise
explanations that connect each result to the user's background and goals.

Return ONLY valid JSON — no markdown fences, no extra text."""

TOPIC_EXPLANATION_PROMPT = """Query: {query}
Complexity level: {complexity}

Topic: {topic_name}
Representative keywords: {keywords}

Write a single-sentence explanation (max 20 words) of why this topic is relevant
to a researcher at the {complexity} level asking this query.

Return: {{"relevance": "..."}}"""

PAPER_EXPLANATION_PROMPT = """Query: {query}
Complexity level: {complexity}
User background inference: {background}

Paper title: {title}
Authors: {authors}
Journal: {journal}, {year}
Abstract snippet: {abstract}

Write a single-sentence explanation (max 25 words) of why this paper is useful
for someone at the {complexity} level asking about: "{query}".

Return: {{"relevance": "..."}}"""

BACKGROUND_INFERENCE_PROMPT = """Query: "{query}"

In one short phrase (5-8 words), infer the user's likely academic background
from the phrasing of their query.
Examples: "graduate student in computational neuroscience", "wet-lab biologist new to bioinformatics"

Return: {{"background": "..."}}"""


# ─────────────────────────────────────────────────────────────────────────────
# RAGResponder
# ─────────────────────────────────────────────────────────────────────────────

class RAGResponder:
    """
    Post-processes AgentResult by enriching topics and papers with
    LLM-generated personalised explanations.

    Usage
    ─────
        responder = RAGResponder()
        enriched = await responder.enrich(agent_result)
        payload = responder.to_api_payload(enriched)
    """

    def __init__(self, provider=None) -> None:
        self._provider = provider or settings.build_llm_provider()

    def set_provider(self, provider_name: str) -> None:
        """Hot-swap the LLM provider without recreating the responder."""
        self._provider = settings.build_llm_provider(provider_name)

    async def enrich(self, result: AgentResult) -> AgentResult:
        """
        Add personalised relevance explanations to topics and papers.
        Runs explanation calls concurrently for speed.
        """
        import asyncio

        # 1. Infer user background once
        background = await self._infer_background(result.query)
        logger.debug("Inferred background: %s", background)

        # 2. Enrich topics & papers concurrently
        topic_tasks = [
            self._explain_topic(result.query, result.complexity, t)
            for t in result.topics
        ]
        paper_tasks = [
            self._explain_paper(result.query, result.complexity, background, p)
            for p in result.papers
        ]

        topic_explanations, paper_explanations = await asyncio.gather(
            asyncio.gather(*topic_tasks, return_exceptions=True),
            asyncio.gather(*paper_tasks, return_exceptions=True),
        )

        # 3. Apply explanations (fall back to existing if LLM call fails)
        for topic, explanation in zip(result.topics, topic_explanations):
            if isinstance(explanation, str) and explanation:
                topic.relevance = explanation

        for paper, explanation in zip(result.papers, paper_explanations):
            if isinstance(explanation, str) and explanation:
                # Store on paper — we add this field dynamically
                paper.__dict__["relevance"] = explanation

        return result

    def to_api_payload(self, result: AgentResult) -> dict[str, Any]:
        """Serialise AgentResult into the JSON schema consumed by the frontend."""
        return {
            "react_trace": [
                {"type": s.type, "content": s.content}
                for s in result.react_trace
            ],
            "steps": result.steps,
            "duration_s": result.duration_s,
            "topics": [
                {
                    "name": t.name,
                    "score": t.score,
                    "relevance": t.relevance,
                    "tools": t.tools,
                }
                for t in result.topics
            ],
            "publications": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "journal": p.journal,
                    "year": p.year,
                    "pmid": p.pmid,
                    "score": p.score,
                    "url": p.url,
                    "relevance": p.__dict__.get("relevance", ""),
                }
                for p in result.papers
            ],
            "handoff": result.handoff,
        }

    # ── LLM explanation helpers ───────────────────────────────────────────────

    async def _infer_background(self, query: str) -> str:
        try:
            prompt = BACKGROUND_INFERENCE_PROMPT.format(query=query)
            response = await self._provider.complete(
                max_tokens=256,
                system=EXPLANATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            import json, re
            raw = response.text.strip()
            raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw).get("background", "researcher")
        except Exception as exc:
            logger.debug("Background inference failed: %s", exc)
            return "researcher"

    async def _explain_topic(self, query: str, complexity: str, topic: Topic) -> str:
        try:
            prompt = TOPIC_EXPLANATION_PROMPT.format(
                query=query,
                complexity=complexity,
                topic_name=topic.name,
                keywords=", ".join(topic.representative_words[:5] or topic.tools[:3]),
            )
            response = await self._provider.complete(
                max_tokens=256,
                system=EXPLANATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            import json, re
            raw = response.text.strip()
            raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw).get("relevance", topic.relevance)
        except Exception as exc:
            logger.debug("Topic explanation failed: %s", exc)
            return topic.relevance

    async def _explain_paper(
        self, query: str, complexity: str, background: str, paper: Paper
    ) -> str:
        try:
            prompt = PAPER_EXPLANATION_PROMPT.format(
                query=query,
                complexity=complexity,
                background=background,
                title=paper.title,
                authors=paper.authors,
                journal=paper.journal,
                year=paper.year,
                abstract=paper.abstract[:400],
            )
            response = await self._provider.complete(
                max_tokens=256,
                system=EXPLANATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            import json, re
            raw = response.text.strip()
            raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw).get("relevance", "")
        except Exception as exc:
            logger.debug("Paper explanation failed: %s", exc)
            return ""