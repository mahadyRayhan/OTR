"""
react_agent.py — ReAct (Reason + Act) orchestration loop.

The agent follows the classic Thought → Action → Observation cycle.
On each step it either:
  1. Calls get_topic_landscape  — maps the conceptual territory
  2. Calls search_publications  — retrieves and re-ranks papers
  3. Calls get_tools            — looks up relevant tools & datasets
  4. Decides to STOP            — when the planner says stop=true

The planner uses Claude to decide which tool to call next, then the
KnowledgeBase executes the real retrieval. Results are collected and
returned as a structured AgentResult.

Fixes applied
─────────────
  #1  Early stopping removed from inside the agent loop. DOME's Self-Evaluator
      is responsible for deciding when output is good enough. The agent always
      completes its full ReAct loop and returns everything it found.

  #2  Handoff dict now carries the full structured output — all topics, all
      authors extracted from papers, and all tools — instead of a truncated
      summary string. The [:20] topic truncation is gone.

  #7  Off-by-one fixed: loop condition changed from
          step_num >= settings.max_react_steps - 1
      to
          step_num >= settings.max_react_steps
      so that max_react_steps=8 actually runs 8 steps (0–7).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from llm_provider import LLMProvider
from knowledge_base import KnowledgeBase, KnowledgeLandscape, Paper, Topic

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    type: str           # THOUGHT | ACTION | OBS
    content: str


@dataclass
class AgentResult:
    query: str
    react_trace: list[TraceStep]
    topics: list[Topic]
    papers: list[Paper]
    handoff: dict[str, Any]
    steps: int
    duration_s: float
    complexity: str = "introductory"


# ─────────────────────────────────────────────────────────────────────────────
# Planner prompt
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are the planner inside a Knowledge Discovery ReAct agent.

Available tools:
  get_topic_landscape(query: str) → maps research topics, tools, datasets
  search_publications(query: str, complexity: str) → finds ranked papers
  get_tools(topic: str) → lists relevant tools/datasets

Respond with ONLY a JSON object (no markdown) with these fields:
{
  "thought": "Reasoning about what to do next and why.",
  "action": "tool_name",
  "action_input": {"arg1": "value1"},
  "stop": false
}

Set stop=true when coverage is sufficient (you have topics AND publications).
complexity must be one of: introductory | intermediate | advanced.
Note: complexity is passed in externally and should NOT be inferred here."""

PLANNER_USER_TEMPLATE = """Query: {query}
Complexity: {complexity}

Steps taken so far:
{history}

Decide the next action."""


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDiscoveryAgent:
    """
    Orchestrates the ReAct loop for a single user query.

    Usage
    ─────
        agent = KnowledgeDiscoveryAgent(knowledge_base)
        result = await agent.run(
            "synaptic plasticity in hippocampal neurons",
            complexity="intermediate",   # passed from user profile / DOME
        )
    """

    def __init__(self, kb: KnowledgeBase, provider=None) -> None:
        self.kb = kb
        self._provider = provider or settings.build_llm_provider()

    def set_provider(self, provider_name: str) -> None:
        """Hot-swap the LLM provider without recreating the agent."""
        self._provider = settings.build_llm_provider(provider_name)

    async def run(self, query: str, complexity: str = "introductory") -> AgentResult:
        """
        Run the full ReAct loop for *query*.

        Parameters
        ──────────
        query      : the user's research question
        complexity : introductory | intermediate | advanced
                     Must be set by the caller (e.g. from the user profile
                     quadrant). The planner no longer re-infers this value.
        """
        t0 = time.perf_counter()
        trace: list[TraceStep] = []
        topics: list[Topic] = []
        papers: list[Paper] = []
        history_lines: list[str] = []

        # Fix #7 — use < max_react_steps so the loop runs exactly
        # max_react_steps iterations (0 … max_react_steps-1).
        for step_num in range(settings.max_react_steps):
            # ── Ask planner what to do next ──────────────────────────────────
            plan = await self._plan(query, history_lines, complexity)

            # Record THOUGHT
            thought_step = TraceStep(type="THOUGHT", content=plan.get("thought", ""))
            trace.append(thought_step)
            history_lines.append(f"THOUGHT: {thought_step.content}")

            # Stop when the planner says so OR when we've exhausted the budget.
            # Fix #1 — no score-based early stopping here; DOME's Self-Evaluator
            # is responsible for deciding whether the output is good enough.
            if plan.get("stop"):
                break

            action = plan.get("action", "")
            action_input = plan.get("action_input", {})

            # Record ACTION
            action_str = self._format_action(action, action_input)
            trace.append(TraceStep(type="ACTION", content=action_str))
            history_lines.append(f"ACTION: {action_str}")

            # ── Execute tool ─────────────────────────────────────────────────
            obs_text, new_topics, new_papers = await self._execute(
                action, action_input, query, complexity
            )
            if new_topics:
                topics = new_topics
            if new_papers:
                papers = new_papers

            # Record OBS
            trace.append(TraceStep(type="OBS", content=obs_text))
            history_lines.append(f"OBS: {obs_text}")

        # ── Build enriched handoff context (Fix #2) ───────────────────────────
        # Pass the full structured output so downstream agents (e.g. Scholar)
        # have everything they need: all topics, all authors, all tools.
        all_topics: list[str] = [t.name for t in topics]
        all_tools: list[str] = list({
            tool
            for t in topics
            for tool in t.tools
        })
        all_authors: list[str] = list({
            author.strip()
            for p in papers
            for author in p.authors.split(",")
            if author.strip()
        })

        handoff: dict[str, Any] = {
            # Scalar conveniences kept for backward-compat (un-truncated now)
            "primary_topic": topics[0].name if topics else query,
            "primary_tool": topics[0].tools[0] if topics and topics[0].tools else "",
            "complexity": complexity,
            # Full arrays for downstream field-mapping
            "topics": all_topics,
            "tools": all_tools,
            "authors": all_authors,
        }

        return AgentResult(
            query=query,
            react_trace=trace,
            topics=topics,
            papers=papers,
            handoff=handoff,
            steps=len(trace),
            duration_s=round(time.perf_counter() - t0, 2),
            complexity=complexity,
        )

    # ── Planner (LLM call) ────────────────────────────────────────────────────

    async def _plan(
        self, query: str, history: list[str], complexity: str
    ) -> dict[str, Any]:
        """
        Ask the LLM what to do next.

        Fix #4 — complexity is now passed in as a parameter and injected into
        the prompt rather than being re-inferred by the planner from query
        phrasing. The planner system prompt no longer contains an instruction
        to infer complexity.
        """
        history_text = "\n".join(history) if history else "None yet."
        user_msg = PLANNER_USER_TEMPLATE.format(
            query=query,
            complexity=complexity,
            history=history_text,
        )
        try:
            response = await self._provider.complete(
                messages=[{"role": "user", "content": user_msg}],
                system=PLANNER_SYSTEM,
                max_tokens=1024,
            )
            raw = response.text.strip()
            raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Planner call failed: %s — using fallback plan.", exc)
            # Deterministic fallback so the chain can still run
            if not history:
                return {
                    "thought": "Starting by mapping the topic landscape for this query.",
                    "action": "get_topic_landscape",
                    "action_input": {"query": query},
                    "stop": False,
                }
            if len(history) < 4:
                return {
                    "thought": "Topics retrieved. Now searching for relevant publications.",
                    "action": "search_publications",
                    "action_input": {"query": query, "complexity": complexity},
                    "stop": False,
                }
            return {
                "thought": "Sufficient coverage obtained.",
                "action": "",
                "action_input": {},
                "stop": True,
            }

    # ── Tool executor ─────────────────────────────────────────────────────────

    async def _execute(
        self,
        action: str,
        action_input: dict[str, Any],
        original_query: str,
        complexity: str,
    ) -> tuple[str, list[Topic], list[Paper]]:
        """Execute the chosen tool and return (obs_text, topics, papers)."""

        if action == "get_topic_landscape":
            q = action_input.get("query", original_query)
            topics = await self.kb.get_topic_landscape(q)
            topic_names = ", ".join(t.name for t in topics[:4])
            tools_seen = list({tool for t in topics for tool in t.tools[:2]})[:4]
            obs = (
                f"Top topics: {topic_names}. "
                f"Tools: {', '.join(tools_seen)}."
            )
            return obs, topics, []

        elif action == "search_publications":
            q = action_input.get("query", original_query)
            # Use the externally-set complexity rather than whatever the
            # planner might have inferred (Fix #4).
            papers = await self.kb.search_publications(q, complexity=complexity)
            if papers:
                score_str = ", ".join(f"{p.score:.2f}" for p in papers[:3])
                obs = (
                    f"{len(papers)} papers retrieved "
                    f"(top scores: {score_str}). "
                    "Returning all results for downstream evaluation."
                )
            else:
                obs = "No papers above threshold. Broadening query on next step."
            return obs, [], papers

        elif action == "get_tools":
            topic = action_input.get("topic", original_query)
            tools = self.kb.get_tools_for_topic(topic)
            datasets = self.kb.get_datasets_for_topic(topic)
            obs = f"Tools: {', '.join(tools)}. Datasets: {', '.join(datasets)}."
            return obs, [], []

        else:
            return f"Unknown action '{action}' — skipping.", [], []

    # ── Formatting helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_action(action: str, action_input: dict[str, Any]) -> str:
        if action == "get_topic_landscape":
            q = action_input.get("query", "")
            return f'KnowledgeDiscoveryAgent.get_topic_landscape("{q}")'
        elif action == "search_publications":
            q = action_input.get("query", "")
            c = action_input.get("complexity", "introductory")
            return f'KnowledgeDiscoveryAgent.search_publications("{q}", complexity={c})'
        elif action == "get_tools":
            t = action_input.get("topic", "")
            return f'KnowledgeDiscoveryAgent.get_tools("{t}")'
        return f"{action}({json.dumps(action_input)})"