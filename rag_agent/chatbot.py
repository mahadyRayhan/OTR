"""
chatbot.py — Multi-turn conversational interface over the Knowledge Discovery Agent.

Maintains per-session conversation history and context so follow-up queries
like "tell me more about STDP" correctly resolve to the prior search context.
Each session stores the last AgentResult and injects it into subsequent turns.

Fix #4 — complexity is now threaded from the caller (user profile / request)
into agent.run() rather than being re-inferred inside the planner prompt.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from config import settings
from llm_provider import LLMProvider
from knowledge_base import KnowledgeBase
from react_agent import AgentResult, KnowledgeDiscoveryAgent
from rag_responder import RAGResponder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str               # user | assistant
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    last_result: AgentResult | None = None
    last_topics: list[str] = field(default_factory=list)
    last_tools: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.touch()

    def anthropic_history(self, max_turns: int = 10) -> list[dict[str, str]]:
        """Return the last N message pairs in Anthropic message format."""
        recent = self.messages[-(max_turns * 2):]
        return [{"role": m.role, "content": m.content} for m in recent]


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────────────────────────────────────

CHAT_SYSTEM = """You are a helpful scientific literature assistant for the Knowledge Discovery Agent.

Your job:
- Help researchers find papers, understand topics, and navigate research areas.
- Use the provided research context (topics, papers, tools) to give grounded answers.
- For new research requests, signal that a fresh agent search should be triggered.
- For clarification questions, answer directly using the session context.
- Keep responses concise (2-4 sentences). Cite paper titles when relevant.

If the user asks about a completely new research area not in the session context,
respond with exactly: [TRIGGER_SEARCH: <refined query>]

Session context will be injected below."""


class Chatbot:
    """
    Manages multi-turn conversations, resolving follow-ups against session context
    and triggering fresh agent runs for new research requests.

    Usage
    ─────
        bot = Chatbot(knowledge_base)
        # First turn — triggers a full agent run
        reply, payload = await bot.chat(session_id, "synaptic plasticity STDP")
        # Follow-up — resolved from session context
        reply, payload = await bot.chat(session_id, "tell me more about the Caporale paper")
    """

    SESSION_TTL = 3600  # 1 hour

    def __init__(self, kb: KnowledgeBase, provider=None) -> None:
        self.kb = kb
        self.agent = KnowledgeDiscoveryAgent(kb)
        self.responder = RAGResponder()
        self._provider = provider or settings.build_llm_provider()
        self._sessions: dict[str, Session] = {}

    def set_provider(self, provider_name: str) -> None:
        """Hot-swap the LLM provider and propagate to agent + responder."""
        provider = settings.build_llm_provider(provider_name)
        self._provider = provider
        self.agent.set_provider(provider_name)
        self.responder.set_provider(provider_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(session_id=sid)
        logger.info("New session: %s", sid)
        return sid

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def chat(
        self,
        session_id: str,
        user_message: str,
        complexity: str = "introductory",
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Process a user message.

        Parameters
        ──────────
        session_id   : active session identifier
        user_message : raw message text from the user
        complexity   : introductory | intermediate | advanced
                       Should be supplied by the caller from the user profile
                       (DOME quadrant Q1–Q4). Defaults to introductory.
                       Fix #4 — no longer re-inferred inside the planner.

        Returns
        ───────
        (reply_text, agent_payload | None)
            agent_payload is set when a fresh search was triggered, None for follow-ups.
        """
        # Ensure session exists
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)

        session = self._sessions[session_id]
        self._evict_stale_sessions()

        session.add("user", user_message)
        clean_query = user_message.strip()

        # Determine intent: new search vs follow-up
        intent = await self._classify_intent(clean_query, session)

        if intent.startswith("[TRIGGER_SEARCH:"):
            # Extract refined query from the intent string
            refined = intent.removeprefix("[TRIGGER_SEARCH:").rstrip("]").strip()
            reply, payload = await self._run_search(
                refined or clean_query, session, complexity
            )
        else:
            # Direct conversational response using session context
            reply = intent
            payload = None

        session.add("assistant", reply)
        return reply, payload

    # ── Intent classifier ─────────────────────────────────────────────────────

    async def _classify_intent(self, query: str, session: Session) -> str:
        """
        Ask the LLM whether this is a follow-up question or a new search request.
        Returns either a direct reply string or [TRIGGER_SEARCH: refined_query].
        """
        context_block = self._build_context_block(session)
        system = f"{CHAT_SYSTEM}\n\n{context_block}"

        try:
            response = await self._provider.complete(
                messages=session.anthropic_history(max_turns=10),
                system=system,
                max_tokens=1024,   #256
            )
            return response.text.strip()
        except Exception as exc:
            logger.warning("Intent classification failed: %s", exc)
            # Fallback: treat every message as a new search
            return f"[TRIGGER_SEARCH: {query}]"

    # ── Search runner ─────────────────────────────────────────────────────────

    async def _run_search(
        self,
        query: str,
        session: Session,
        complexity: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Run the full ReAct → RAG pipeline and update session context.

        Fix #4 — complexity is passed through to agent.run() rather than
        being re-inferred inside the planner.
        """
        result = await self.agent.run(query, complexity=complexity)
        enriched = await self.responder.enrich(result)
        payload = self.responder.to_api_payload(enriched)

        # Update session context
        session.last_result = enriched
        session.last_topics = [t.name for t in enriched.topics]
        # Fix #2 — handoff now contains a "tools" list, not a bare string
        session.last_tools = enriched.handoff.get("tools", [])

        topic_names = ", ".join(session.last_topics[:3])
        reply = (
            f"I've mapped the research landscape for your query. "
            f"Key topics identified: **{topic_names}**. "
            f"I found {len(enriched.papers)} relevant papers — see the results panel."
        )
        return reply, payload

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context_block(session: Session) -> str:
        if not session.last_result:
            return "No prior search context in this session."
        r = session.last_result
        topic_str = ", ".join(t.name for t in r.topics[:3])
        paper_str = " | ".join(
            f"{p.title[:50]} ({p.year})" for p in r.papers[:3]
        )
        return (
            f"Current session context:\n"
            f"  Last query: {r.query}\n"
            f"  Topics: {topic_str}\n"
            f"  Papers: {paper_str}\n"
            f"  Handoff: {r.handoff}\n"
        )

    def _evict_stale_sessions(self) -> None:
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.SESSION_TTL
        ]
        for sid in stale:
            del self._sessions[sid]
            logger.debug("Evicted stale session: %s", sid)