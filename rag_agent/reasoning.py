"""
reasoning.py — Pluggable reasoning strategies for the RAG agent.

Supported strategies:
    cot  — Chain of Thought: linear step-by-step reasoning
    tot  — Tree of Thought: branches multiple reasoning paths, picks the best
    got  — Graph of Thought: decomposes into sub-questions, merges answers

Each plugin exposes:
    name: str
    execute(query, context, model, depth) -> str
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningPlugin(ABC):
    name: str = "base"

    @abstractmethod
    async def execute(self, query: str, context: str, model: Any, depth: int = 1) -> str:
        pass

    async def _call(self, prompt: str, model: Any) -> str:
        """Safe LLM call — returns empty string on failure."""
        if model is None:
            return "[No LLM available]"
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return f"[LLM error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Chain of Thought (CoT)
# ─────────────────────────────────────────────────────────────────────────────

class ChainOfThought(ReasoningPlugin):
    """
    Linear step-by-step reasoning.

    depth=1 → single-pass answer with reasoning shown
    depth=2 → answer, then a self-critique and refined answer
    depth=3 → answer, critique, refinement, then a final synthesis pass
    """
    name = "cot"

    async def execute(self, query: str, context: str, model: Any, depth: int = 1) -> str:
        # Step 1 — initial answer with explicit reasoning
        prompt = f"""You are an expert assistant. Use the context below to answer the query.
Think step by step, showing your reasoning explicitly before giving a final answer.

Context:
{context}

Query: {query}

Format your response as:
Step 1: [first reasoning step]
Step 2: [next reasoning step]
...
Final Answer: [concise answer]"""

        answer = await self._call(prompt, model)

        if depth < 2:
            return answer

        # Step 2 — self-critique
        critique_prompt = f"""Review this answer to the query "{query}":

{answer}

Identify any gaps, errors, or missing nuance. Then write an improved answer."""
        answer = await self._call(critique_prompt, model)

        if depth < 3:
            return answer

        # Step 3 — final synthesis
        synthesis_prompt = f"""Given everything reasoned so far about "{query}", write a clean, 
comprehensive final answer. Remove any meta-commentary about the reasoning process itself.

Context:
{context}

Previous reasoning:
{answer}

Final synthesised answer:"""
        return await self._call(synthesis_prompt, model)


# ─────────────────────────────────────────────────────────────────────────────
# Tree of Thought (ToT)
# ─────────────────────────────────────────────────────────────────────────────

class TreeOfThought(ReasoningPlugin):
    """
    Explores multiple reasoning branches concurrently, then picks the best.

    depth=1 → 2 branches
    depth=2 → 3 branches + evaluation pass
    depth=3 → 4 branches + evaluation + synthesis of top-2
    """
    name = "tot"

    async def execute(self, query: str, context: str, model: Any, depth: int = 1) -> str:
        num_branches = min(depth + 1, 4)

        # Generate branches concurrently
        branch_prompts = [
            f"""You are reasoning about: "{query}"
Context: {context}

Approach #{i+1}: Think about this from a {'mechanistic' if i==0 else 'comparative' if i==1 else 'practical' if i==2 else 'critical'} angle.
Provide a concise but complete answer using this approach."""
            for i in range(num_branches)
        ]

        branches = await asyncio.gather(*[self._call(p, model) for p in branch_prompts])

        # Evaluate and pick the best branch
        branch_text = "\n\n".join(
            f"Branch {i+1}:\n{b}" for i, b in enumerate(branches)
        )

        eval_prompt = f"""You have {num_branches} different reasoning branches for the query:
"{query}"

{branch_text}

Evaluate each branch for accuracy, completeness, and clarity.
Then write a final best answer, incorporating the strongest elements from each branch.

Final Answer:"""

        best = await self._call(eval_prompt, model)

        if depth < 3:
            return best

        # depth=3: synthesise top-2 branches with the evaluation
        synthesis_prompt = f"""Building on this evaluation:
{best}

Write a definitive, polished answer to: "{query}"
Draw from the context:
{context}"""
        return await self._call(synthesis_prompt, model)


# ─────────────────────────────────────────────────────────────────────────────
# Graph of Thought (GoT)
# ─────────────────────────────────────────────────────────────────────────────

class GraphOfThought(ReasoningPlugin):
    """
    Decomposes the query into sub-questions (nodes), answers each, then
    merges the answers at a final aggregation node.

    depth=1 → 2 sub-questions
    depth=2 → 3 sub-questions + cross-link pass
    depth=3 → 4 sub-questions + cross-link + final synthesis
    """
    name = "got"

    async def execute(self, query: str, context: str, model: Any, depth: int = 1) -> str:
        num_nodes = min(depth + 1, 4)

        # Node 0 — decompose into sub-questions
        decompose_prompt = f"""Break this query into exactly {num_nodes} distinct sub-questions 
that together fully answer the original.

Query: {query}
Context available: {context[:500]}

Output ONLY a numbered list of sub-questions, nothing else:"""

        decomposition = await self._call(decompose_prompt, model)

        # Parse sub-questions
        sub_questions = []
        for line in decomposition.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                q = line.lstrip("0123456789.-) ").strip()
                if q:
                    sub_questions.append(q)

        # Fallback if parsing failed
        if not sub_questions:
            sub_questions = [query]

        sub_questions = sub_questions[:num_nodes]

        # Answer each sub-question concurrently (graph nodes)
        node_prompts = [
            f"""Context: {context}

Sub-question: {sq}

Answer this sub-question concisely and precisely:"""
            for sq in sub_questions
        ]

        node_answers = await asyncio.gather(*[self._call(p, model) for p in node_prompts])

        # Build node summary
        nodes_text = "\n\n".join(
            f"Node {i+1} — {sq}\nAnswer: {ans}"
            for i, (sq, ans) in enumerate(zip(sub_questions, node_answers))
        )

        if depth >= 2:
            # Cross-link pass — find connections between nodes
            crosslink_prompt = f"""These sub-answers relate to the query: "{query}"

{nodes_text}

Identify connections and dependencies between these answers, 
then write an integrated intermediate answer:"""
            nodes_text = await self._call(crosslink_prompt, model)

        # Final aggregation node — merge all into one answer
        merge_prompt = f"""You have answered all sub-questions for the query: "{query}"

Sub-question answers:
{nodes_text}

Context:
{context}

Now write a single, coherent, complete final answer that integrates all the above:"""

        final = await self._call(merge_prompt, model)

        if depth < 3:
            return final

        # depth=3 — polish pass
        polish_prompt = f"""Polish this answer to "{query}" — make it clear, concise, and well-structured.
Remove any redundancy. Keep all key information.

Draft:
{final}"""
        return await self._call(polish_prompt, model)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGIES: dict[str, ReasoningPlugin] = {
    "cot": ChainOfThought(),
    "tot": TreeOfThought(),
    "got": GraphOfThought(),
}


class ReasoningEngine:
    """Registry that maps strategy names to plugin instances."""

    def get_strategy(self, name: str) -> ReasoningPlugin:
        key = name.lower().strip()
        if key not in _STRATEGIES:
            available = ", ".join(_STRATEGIES.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return _STRATEGIES[key]

    @staticmethod
    def available() -> list[str]:
        return list(_STRATEGIES.keys())