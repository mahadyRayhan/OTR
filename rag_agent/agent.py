"""
rag_agent.py — RAG agent with pluggable reasoning strategies (CoT / ToT / GoT).

Uses the same KnowledgeBase (FAISS + PubMed + Semantic Scholar + cross-encoder)
as react_agent.py for retrieval, but replaces the ReAct loop with a configurable
reasoning strategy that synthesises retrieved papers into a direct answer.

Returns:
    {
        "answer":      str,   # synthesised answer
        "quality":     str,   # LLM-as-judge score e.g. "Score: 8/10 - ..."
        "strategy":    str,   # "COT" | "TOT" | "GOT"
        "source_used": str,   # titles of papers used as context
    }
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import os
from typing import Any

from rag_agent.knowledge_base import KnowledgeBase, Paper
from rag_agent.rag_config import settings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🚀 ADD THIS TO FORCE LOAD THE API KEYS
from dotenv import load_dotenv
load_dotenv()
 
logger = logging.getLogger(__name__)

# Per-call token accumulator — set in call_rag_agent, updated by every _llm() call
_rag_token_acc: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_rag_token_acc", default=None
)
 
 
def _build_context(papers: list[Paper], max_papers: int = 5) -> tuple[str, str]:
    top = papers[:max_papers]
    chunks = []
    for i, p in enumerate(top):
        abstract = (p.abstract or "")[:2000]
        chunks.append(
            f"[{i+1}] {p.title} ({p.year})\n"
            f"Authors: {p.authors}\n"
            f"Abstract: {abstract}"
        )
    context = "\n\n".join(chunks)
    source_titles = " | ".join(p.title[:60] for p in top)
    return context, source_titles
 
 
async def _llm(provider, prompt: str, max_tokens: int = 8192) -> str:
    try:
        response = await provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        acc = _rag_token_acc.get()
        if acc is not None:
            acc["input"]  += response.input_tokens
            acc["output"] += response.output_tokens
        return response.text.strip()
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return f"[LLM error: {e}]"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CoT
# ─────────────────────────────────────────────────────────────────────────────
 
async def _cot(query: str, context: str, provider, depth: int) -> str:

    answer = await _llm(
        provider,
        f"You are a scientific expert. Based on the research papers below, "
        f"answer this question thoroughly: {query}\n\n"
        f"Research papers:\n{context}\n\n"
        f"Provide a detailed answer grounded in the papers above."
    )

    if depth < 2:
        return answer

    answer = await _llm(
        provider,
        f"You are improving a scientific answer. Question: {query}\n\n"
        f"Current answer:\n{answer}\n\n"
        f"Research papers:\n{context}\n\n"
        f"Identify specifically what is missing or incorrect, then write "
        f"a complete improved answer that fills those gaps. "
        f"Ground every claim in the papers above."
    )

    if depth < 3:
        return answer

    return await _llm(
        provider,
        f"Write a final, polished answer to: {query}\n\n"
        f"Research papers:\n{context}\n\n"
        f"Previous reasoning:\n{answer}\n\n"
        f"Final answer:"
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ToT
# ─────────────────────────────────────────────────────────────────────────────
 
async def _tot(query: str, context: str, provider, depth: int) -> str:
    num_branches = min(depth + 1, 4)
    angles = [
        "focus on the biological mechanisms involved",
        "compare different approaches and methods",
        "focus on practical implications and applications",
        "take a critical view of current limitations",
    ]

    branches = await asyncio.gather(*[
        _llm(provider,
            f"You are a scientific expert. Answer this question by focusing on "
            f"the following angle: {angles[i]}\n\n"
            f"Question: {query}\n\n"
            f"Research papers:\n{context}\n\n"
            f"Answer:"
        )
        for i in range(num_branches)
    ])

    # At depth=0 only 1 branch is generated — synthesis adds a redundant LLM
    # call with no new information. Skip it and return the single answer directly.
    if num_branches == 1:
        return branches[0]

    branch_text = "\n\n---\n\n".join(
        f"Perspective {i+1} ({angles[i]}):\n{b}"
        for i, b in enumerate(branches)
    )

    best = await _llm(provider,
        f"You have {num_branches} different expert perspectives on this question: {query}\n\n"
        f"{branch_text}\n\n"
        f"Synthesise these perspectives into one comprehensive, well-rounded answer:"
    )

    if depth < 3:
        return best

    return await _llm(provider,
        f"Polish and finalise this answer to: {query}\n\n"
        f"Draft answer:\n{best}\n\n"
        f"Write a clear, concise final version:"
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GoT
# ─────────────────────────────────────────────────────────────────────────────
 
async def _got(query: str, context: str, provider, depth: int) -> str:
    num_nodes = min(depth + 1, 4)
 
    decomposition = await _llm(provider,
        f"Break this question into exactly {num_nodes} specific sub-questions "
        f"that together fully answer it.\n\n"
        f"Question: {query}\n\n"
        f"Output only a numbered list of sub-questions:"
    )
 
    sub_questions = []
    for line in decomposition.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            q = re.sub(r"^[\d\.\-\)]+\s*", "", line).strip()
            if q:
                sub_questions.append(q)
 
    if not sub_questions:
        sub_questions = [query]
    sub_questions = sub_questions[:num_nodes]
 
    node_answers = await asyncio.gather(*[
        _llm(provider,
            f"Using these research papers as reference:\n{context}\n\n"
            f"Answer this specific question: {sq}"
        )
        for sq in sub_questions
    ])
 
    nodes_text = "\n\n".join(
        f"Q{i+1}: {sq}\nA{i+1}: {ans}"
        for i, (sq, ans) in enumerate(zip(sub_questions, node_answers))
    )
 
    if depth >= 2:
        nodes_text = await _llm(provider,
            f"These are answers to sub-questions about: {query}\n\n"
            f"{nodes_text}\n\n"
            f"Identify the key connections and write an integrated summary:"
        )
 
    return await _llm(provider,
        f"Based on all the research and reasoning below, write a single "
        f"comprehensive answer to: {query}\n\n"
        f"{nodes_text}\n\n"
        f"Research papers:\n{context}\n\n"
        f"Final answer:"
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────────────────────
 
async def _judge(query: str, answer: str, context: str, provider) -> str:
    try:
        result = await _llm(provider,
            f"Rate this answer to the question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Reference papers: {context[:600]}\n\n"
            f"Give a score out of 10 and one sentence of justification. "
            f"Format: 'Score: X/10 - [reason]'",
            max_tokens=512,
        )
        return result
    except Exception as e:
        return f"Judge error: {e}"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
_STRATEGIES = {"cot": _cot, "tot": _tot, "got": _got}
 
 
async def call_rag_agent(
    kb: KnowledgeBase,
    user_input: str,
    strategy: str = "cot",
    depth: int = 1,
    complexity: str = "introductory",
    agent_registry: str = "",
) -> dict[str, Any]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_STRATEGIES.keys())}")

    _acc = {"input": 0, "output": 0}
    _tok = _rag_token_acc.set(_acc)

    # Use the pre-loaded kb passed by the caller (e.g. OrchestratorEngine).
    # The original code overwrote this with a fresh KnowledgeBase().load() every
    # call, reloading the embedding model + FAISS + cross-encoder each time (~30s).
    if kb is None or not hasattr(kb, 'chroma'):
        kb = KnowledgeBase()
        kb.load()
    provider = settings.build_llm_provider()
    logger.warning("-------------------------------------------------------------")
    logger.warning("user_input : '%s'", user_input)
    logger.warning("-------------------------------------------------------------")
    papers = await kb.search_publications(user_input, complexity=complexity)
 
    # if not papers:
    #     logger.warning("RAG agent: no papers retrieved for '%s'", user_input)
    #     answer = await _llm(provider,
    #         f"You are a scientific expert. Answer comprehensively: {user_input}"
    #     )
    #     return {
    #         "answer": answer,
    #         "quality": "Score: N/A - No papers retrieved, answer from general knowledge.",
    #         "strategy": strategy.upper(),
    #         "source_used": "General LLM Knowledge",
    #     }
    
    if not papers:
        print("❌ FAISS returned 0 papers above the confidence threshold.")
        print("💡 FIX: Try lowering 'confidence_threshold' in your config.py or .env file.")
    else:
        print(f"✅ Found {len(papers)} papers!")
        for i, p in enumerate(papers):
            print(f"  [{i+1}] Score {p.score:.2f} | {p.title[:60]}...")
 
    context, source_titles = _build_context(papers)
    if agent_registry:
        context = f"Agent context: {agent_registry}\n\n{context}"
 
    answer = await _STRATEGIES[strategy](user_input, context, provider, depth)
    # Internal judge removed — the GoT external judge evaluates every node already.
    # Calling _judge() here was a redundant second LLM call per RAG invocation.

    if not papers and len(answer.strip()) < 80:
        answer = f"[STRUCTURAL_FAIL: no papers retrieved and answer is too short]\n{answer}"

    _rag_token_acc.reset(_tok)
    return {
        "answer":     answer,
        "quality":    "N/A (judged externally by GoT)",
        "strategy":   strategy.upper(),
        "source_used": source_titles,
        "paper_count": len(papers),
        "tokens":     _acc,
    }
 