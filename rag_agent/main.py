"""
main.py — Knowledge Discovery Agent: FastAPI app + Uvicorn entry point.

Fix #4 — complexity from SearchRequest is now threaded into agent.run()
and chatbot.chat() instead of being re-inferred by the planner prompt.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from knowledge_base import KnowledgeBase
from react_agent import KnowledgeDiscoveryAgent
from rag_responder import RAGResponder
from chatbot import Chatbot

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    Path(settings.log_path).parent.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    console_fmt = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
    )
    file_fmt = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setFormatter(console_fmt)
    file_h = logging.FileHandler(settings.log_path)
    file_h.setFormatter(file_fmt)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(console_h)
    root.addHandler(file_h)

    for noisy in ("httpx", "httpcore", "faiss", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# App state (singletons initialised once at startup)
# ---------------------------------------------------------------------------

class _State:
    kb: KnowledgeBase
    agent: KnowledgeDiscoveryAgent
    responder: RAGResponder
    chatbot: Chatbot
    ready: bool = False


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    log = structlog.get_logger("startup")
    log.info("Loading Knowledge Discovery Agent...", provider=settings.llm_provider)

    state.kb = KnowledgeBase()
    state.kb.load()
    state.agent = KnowledgeDiscoveryAgent(state.kb)
    state.responder = RAGResponder()
    state.chatbot = Chatbot(state.kb)
    state.ready = True
    log.info("Ready.", provider=settings.llm_provider, model=settings.llm_model or "(default)")
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Knowledge Discovery Agent",
    description="RAG-powered scientific literature discovery with ReAct reasoning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _error_handler(request, exc):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


def _ready():
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service not ready.")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=512)
    complexity: str = Field("introductory", description="introductory | intermediate | advanced")


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=512)
    complexity: str = Field(
        "introductory",
        description=(
            "introductory | intermediate | advanced. "
            "Should be derived from the user profile quadrant (DOME Q1–Q4) "
            "by the caller. Fix #4 — no longer re-inferred by the planner."
        ),
    )

class RAGRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=512)
    strategy: str = Field("cot", description="cot | tot | got")
    depth: int = Field(1, description="Reasoning depth 1-3")
    complexity: str = Field("introductory", description="introductory | intermediate | advanced")
    agent_registry: str = Field("", description="Optional multi-agent context")

class TopicsRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=512)


class SwitchProviderRequest(BaseModel):
    provider: str = Field(..., description="claude | openai | gemini | ollama | or full model string")
    model: str = Field("", description="Optional model override e.g. haiku, gpt-4o-mini")


class BenchmarkRequest(BaseModel):
    providers: list[str] = Field(default=["claude", "openai", "gemini"])
    component: str = Field("planner", description="planner | topic_explanation | paper_explanation | all")


# ---------------------------------------------------------------------------
# Endpoints — core
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/readiness")
async def readiness():
    _ready()
    return {"status": "ready", "provider": settings.llm_provider}


@app.post("/api/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    """Full single-turn search: ReAct agent -> cross-encoder re-ranking -> LLM explanation.

    Fix #4 — complexity is taken from the request and passed directly into
    agent.run(); the planner no longer re-infers it from query phrasing.
    """
    _ready()
    complexity = (
        req.complexity
        if req.complexity in ("introductory", "intermediate", "advanced")
        else "introductory"
    )
    # Pass complexity explicitly so the planner never has to guess it.
    result = await state.agent.run(req.query, complexity=complexity)
    enriched = await state.responder.enrich(result)
    return state.responder.to_api_payload(enriched)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    """Multi-turn chat with session memory.

    Fix #4 — complexity is threaded from the request into chatbot.chat().
    """
    _ready()
    if state.chatbot.get_session(req.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found. Call GET /api/session/new first.")
    reply, agent_payload = await state.chatbot.chat(
        req.session_id,
        req.message,
        complexity=req.complexity,
    )
    return {"reply": reply, "agent_payload": agent_payload, "session_id": req.session_id}


@app.post("/api/rag")
async def rag_search(req: RAGRequest) -> dict[str, Any]:
    """
    RAG agent with pluggable reasoning strategies (CoT / ToT / GoT).
 
    Uses the same FAISS + PubMed + Semantic Scholar retrieval pipeline
    as /api/search, but replaces the ReAct loop with a configurable
    reasoning strategy.
 
    Strategies:
      cot — Chain of Thought: linear step-by-step reasoning
      tot — Tree of Thought: parallel branches, picks the best
      got — Graph of Thought: decomposes into sub-questions, merges answers
 
    Depth 1-3 controls how many reasoning passes are made.
    """
    _ready()
    from .agent import call_rag_agent
    result = await call_rag_agent(
        user_input=req.query,
        strategy=req.strategy,
        depth=req.depth,
        agent_registry=req.agent_registry,
    )
    return result
 

@app.get("/api/session/new")
async def new_session() -> dict[str, str]:
    _ready()
    return {"session_id": state.chatbot.new_session()}


@app.post("/api/topics")
async def get_topics(req: TopicsRequest) -> dict[str, Any]:
    """Fast topic landscape — no publication retrieval."""
    _ready()
    topics = await state.kb.get_topic_landscape(req.query)
    return {
        "query": req.query,
        "topics": [{"name": t.name, "score": t.score, "relevance": t.relevance, "tools": t.tools} for t in topics],
    }


@app.get("/api/tools/{topic}")
async def get_tools(topic: str) -> dict[str, Any]:
    """Tools and datasets for a topic from the curated registry."""
    _ready()
    return {
        "topic": topic,
        "tools": state.kb.get_tools_for_topic(topic),
        "datasets": state.kb.get_datasets_for_topic(topic),
    }


# ---------------------------------------------------------------------------
# Endpoints — provider management
# ---------------------------------------------------------------------------

@app.get("/api/providers")
async def list_providers_endpoint() -> dict:
    """List all supported LLM providers and the currently active one."""
    from llm_provider import list_providers
    return {
        "active": {"provider": settings.llm_provider, "model": settings.llm_model or "(default)"},
        "available": list_providers(),
    }


@app.post("/api/providers/switch")
async def switch_provider(req: SwitchProviderRequest) -> dict:
    """Hot-swap the active LLM provider across the whole agent stack — no restart needed."""
    _ready()
    try:
        provider = settings.build_llm_provider(req.provider)
        if req.model:
            provider.model = req.model
        state.chatbot.set_provider(req.provider)
        return {
            "switched_to": {"provider": provider.provider_name, "model": provider.model},
            "message": f"Switched to {provider.provider_name}:{provider.model}",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/providers/benchmark")
async def benchmark_providers(req: BenchmarkRequest) -> dict:
    """Head-to-head latency, cost, and JSON-validity comparison across providers."""
    _ready()
    from provider_benchmark import ProviderBenchmark
    bench = ProviderBenchmark(req.providers)
    summary = await bench.run(component=req.component)
    return dataclasses.asdict(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge Discovery Agent")
    parser.add_argument("--host", default=settings.app_host)
    parser.add_argument("--port", type=int, default=settings.app_port)
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload")
    parser.add_argument("--log-level", default=settings.log_level,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    configure_logging(args.log_level)
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
        access_log=True,
    )