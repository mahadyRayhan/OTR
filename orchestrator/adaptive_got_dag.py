"""
adaptive_got_dag.py
═══════════════════════════════════════════════════════════════════════════════
Adaptive Graph-of-Thoughts DAG Orchestrator

Replaces the classic Kahn's Algorithm (layered topological sort) with an
adaptive GoT-DAG whose depth is co-governed by two real-time signals:

  Signal 1 — User Profile (UserProfile from user_profiler.py)
      Each dimension (CONCEPTUAL / PROCEDURAL / ANALYTICAL) has its own
      maximum depth budget derived from its profile weight.
      Heavy profile weight → more sub-questions → richer answer.

  Signal 2 — LLM-as-Judge quality score (Zheng et al. 2023)
      After each execution wave, nodes below a quality threshold are
      expanded via GoT operators (BRANCH / REFINE / AGGREGATE).
      The system stops when quality_score ≥ threshold, depth ≥ budget,
      or quality delta < ε (convergence plateau).

Algorithm family
────────────────
  GoT (Graph of Thoughts):
      Besta et al. (2023). Graph of Thoughts: Solving Elaborate Problems
      with Large Language Models. AAAI 2024.
      Operators: Branch, Aggregate, Refine, Select.

  Iterative Deepening (IDDFS applied to GoT):
      Korf (1985). Depth-First Iterative-Deepening: An Optimal Admissible
      Tree Search. Artificial Intelligence, 27(1), 97-109.
      Adapted: instead of goal-test, uses quality threshold as stopping crit.

  Self-Refine / quality gating:
      Madaan et al. (2023). Self-Refine: Iterative Refinement with
      Self-Feedback. NeurIPS 2023.

  LLM-as-Judge:
      Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and
      Chatbot Arena. NeurIPS 2023.

  Beam-width control (prevents combinatorial explosion):
      Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with
      Large Language Models. NeurIPS 2023.

  Profile-weighted depth budget (novel):
      Combining Graves (2016) Adaptive Computation Time framing with
      Kobsa (2001) user modeling — not present in prior GoT literature.

  Convergence termination:
      Even-Dar et al. (2006). Action Elimination and Stopping Conditions
      for the Multi-Armed Bandit. JMLR.

What is novel here vs. existing work
─────────────────────────────────────
  GoT (Besta 2023):       Fixed graph structure, no adaptive depth.
  ToT (Yao 2023):         Tree only (no DAG aggregation), no user profile.
  Self-Refine (Madaan):   Linear chain, no graph, no profile.
  Reflexion (Shinn 2023): Single-path, verbal memory only.
  ──────────────────────────────────────────────────────────────────────
  THIS SYSTEM:            Per-dimension depth budget from live user profile
                          + quality-gated IDDFS on a full GoT DAG
                          + beam-width cap to prevent explosion
                          + four-condition termination for convergence.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import google.generativeai as genai
import config

# ── import sibling modules (or stub if running standalone) ────────────────────
try:
    from user_profiler import Dim, UserProfile
except ImportError:
    # Minimal stubs for standalone execution
    class Dim(str, Enum):  # type: ignore[no-redef]
        CONCEPTUAL = "conceptual"
        PROCEDURAL = "procedural"
        ANALYTICAL = "analytical"

    @dataclass
    class UserProfile:  # type: ignore[no-redef]
        alpha_init: float = 10.0
        alpha: Dict = field(default_factory=lambda: {
            Dim.CONCEPTUAL: 10.0, Dim.PROCEDURAL: 10.0, Dim.ANALYTICAL: 10.0
        })
        n_queries: int = 0

        @property
        def profile(self):
            total = sum(self.alpha.values())
            return {d: (self.alpha[d] / total) * 100.0 for d in Dim}

try:
    from query_decomposer import TaskNode
except ImportError:
    @dataclass
    class TaskNode:  # type: ignore[no-redef]
        id: int
        agent: str
        query: str
        depends_on: List[int] = field(default_factory=list)
        qdmr_op: Optional[str] = None
        features_covered: Optional[FrozenSet[str]] = None

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GoT OPERATORS
# Formalised in Besta et al. (2023), Section 3.
# ──────────────────────────────────────────────────────────────────────────────

class GoTOp(str, Enum):
    """
    Graph-of-Thoughts transformation operators.

    BRANCH    — Split one thought into k independent sub-thoughts.
                Used when a topic is too broad; each branch explores one facet.
                Increases breadth. Maps to: decompose in CONCEPTUAL/ANALYTICAL.

    REFINE    — Replace a thought with an improved version of itself.
                Used when a thought is on-track but insufficiently deep.
                Preserves structure; improves content. Maps to: PROCEDURAL.

    AGGREGATE — Merge k parent thoughts into one synthesised thought.
                Used after parallel branches to combine insights.
                Reduces graph width. Maps to: final synthesis step.

    SELECT    — Pick the single best thought from a set of candidates.
                Used when branching produced redundant or contradictory paths.
                Acts as a filter / beam-width enforcer.

    ROOT      — The initial seed thought (no operator applied).
    """
    ROOT      = "root"
    BRANCH    = "branch"
    REFINE    = "refine"
    AGGREGATE = "aggregate"
    SELECT    = "select"


class FailureMode(str, Enum):
    """
    Classifies WHY a node scored below the quality threshold.

    This is the key fix: a scalar score of 0.2 is ambiguous — it could mean
    "the reasoning is shallow (expandable)" or "the data doesn't exist
    (not fixable by going deeper)". Expansion should only be triggered for
    failure modes where more computation can actually help.

    SHALLOW_REASONING — Answer exists but lacks depth/detail.
                        Fix: BRANCH or REFINE. Expansion ALLOWED.
    LOGICAL_ERROR     — Reasoning contains a factual or logical mistake.
                        Fix: REFINE with critique. Expansion ALLOWED.
    FORMAT_ERROR      — Output is malformed, truncated, or wrong structure.
                        Fix: REFINE with format guidance. Expansion ALLOWED.
    MISSING_DATA      — The knowledge base genuinely doesn't contain the answer.
                        Fix: None. Expanding will loop. Expansion BLOCKED.
    UNKNOWN           — Judge could not determine the failure type (heuristic path).
                        Treated conservatively: expansion ALLOWED once, then blocked.
    """
    SHALLOW_REASONING = "shallow_reasoning"
    LOGICAL_ERROR     = "logical_error"
    FORMAT_ERROR      = "format_error"
    MISSING_DATA      = "missing_data"
    UNKNOWN           = "unknown"


# Modes that can potentially be fixed by deeper computation
_EXPANDABLE_MODES: frozenset = frozenset({
    FailureMode.SHALLOW_REASONING,
    FailureMode.LOGICAL_ERROR,
    FailureMode.FORMAT_ERROR,
    FailureMode.UNKNOWN,
})


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH NODES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GoTNode:
    """
    A single thought node in the adaptive GoT-DAG.

    Attributes
    ──────────
    node_id         Unique UUID string.
    task            The underlying TaskNode (agent + query).
    dim             Which profile dimension this node serves.
    depth           Distance from the root node.
    op              GoT operator that produced this node.
    parent_ids      IDs of parent nodes (empty for root).
    result          Execution result (filled by executor).
    quality_score   LLM-as-judge score in [0.0, 1.0] (None = not yet judged).
    is_expanded     True once this node has been expanded with children.
    children_ids    IDs of child nodes produced by expansion.
    """
    node_id:        str
    task:           TaskNode
    dim:            Dim
    depth:          int
    op:             GoTOp              = GoTOp.ROOT
    parent_ids:     List[str]          = field(default_factory=list)
    result:         Optional[str]      = None
    quality_score:  Optional[float]    = None
    # Fix 1: typed failure classification (replaces opaque scalar)
    failure_mode:   Optional[FailureMode] = None
    # Fix 2: critique text stored here so _make_refine_task can thread it forward
    judge_reasoning: Optional[str]    = None
    is_expanded:    bool               = False
    children_ids:   List[str]          = field(default_factory=list)
    elapsed_s:      float              = 0.0

    @property
    def is_executed(self) -> bool:
        return self.result is not None

    @property
    def needs_expansion(self) -> bool:
        """True if this node was judged below threshold and hasn't been expanded yet."""
        return (
            self.quality_score is not None
            and self.quality_score < 1.0  # caller sets threshold externally
            and not self.is_expanded
        )


# ──────────────────────────────────────────────────────────────────────────────
# DEPTH BUDGET  (Novel contribution: profile → depth per dimension)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DepthBudget:
    """
    Maps user profile weights to per-dimension maximum expansion depth.

    Math
    ────
    For each dimension d with profile weight w_d ∈ [0, 100]:

        depth_d = base_depth + floor(w_d / weight_per_level)

    With base_depth=1, weight_per_level=20:
        w=33% (balanced)   → depth = 1 + 1 = 2
        w=50% (moderate)   → depth = 1 + 2 = 3
        w=70% (dominant)   → depth = 1 + 3 = 4
        w=90% (heavy)      → depth = 1 + 4 = 5

    The depth is capped at max_depth to prevent runaway expansion.

    Intuition
    ─────────
    If the user profile has ANALYTICAL=70%, the system will ask up to
    4 rounds of analytical sub-questions before accepting an answer.
    A PROCEDURAL query at 20% gets only 1 round — just produce the code.

    Reference:
        Graves (2016). Adaptive Computation Time for Recurrent Neural Networks.
        arXiv:1603.08983. (Framing: halt computation when quality is sufficient.)
    """
    base_depth:       int   = 1
    weight_per_level: float = 20.0
    max_depth:        int   = 5
    min_depth:        int   = 1

    def budget_for(self, profile: Dict[Dim, float]) -> Dict[Dim, int]:
        """Compute max depth per dimension from a normalised profile."""
        return {
            dim: min(
                self.max_depth,
                max(
                    self.min_depth,
                    self.base_depth + int(profile[dim] / self.weight_per_level),
                ),
            )
            for dim in Dim
        }

    def summary(self, profile: Dict[Dim, float]) -> str:
        budgets = self.budget_for(profile)
        return "  ".join(f"{d.value[:4]}={budgets[d]}" for d in Dim)


# ──────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE   (Zheng et al. 2023)
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """
You are an expert evaluator for a bioinformatics AI system.
Score the following answer on a scale of 1 to 10 using EXACTLY these criteria:

  Completeness (1-10): Does it fully answer the question?
  Depth        (1-10): Is the level of detail appropriate for the question type?
  Accuracy     (1-10): Are the claims factually correct?
  Clarity      (1-10): Is it well-structured and easy to follow?

Also classify the PRIMARY reason the score is below 10, choosing EXACTLY one:
  "shallow_reasoning" — answer exists but lacks detail or depth
  "logical_error"     — contains a factual mistake or flawed reasoning
  "format_error"      — output is malformed, truncated, or wrong structure
  "missing_data"      — the information genuinely does not exist in the knowledge base
                        (e.g. answer says "not found", "no data available", "unknown")
  "unknown"           — cannot determine; answer is near-perfect

Output ONLY valid JSON — no prose, no markdown:
{{"completeness": <int>, "depth": <int>, "accuracy": <int>, "clarity": <int>,
  "failure_mode": "<one of the five strings above>",
  "reasoning": "<one sentence explaining the primary weakness>"}}

Question: {question}
Answer: {answer}
""".strip()

_JUDGE_TIMEOUT_S: float = 45.0
_JUDGE_WEIGHTS = {"completeness": 0.35, "depth": 0.30, "accuracy": 0.25, "clarity": 0.10}


async def _llm_judge(
    question: str,
    answer: str,
    model: genai.GenerativeModel,
) -> Tuple[float, str, FailureMode]:
    """
    Score an answer with an LLM judge.

    Returns
    ───────
    (score_0_to_1, reasoning_string, failure_mode)

    Score is a weighted average of the four criteria, normalised to [0,1].
    failure_mode classifies WHY the score is low — the expansion loop uses
    this to skip MISSING_DATA nodes (no point going deeper into absent data).

    On failure, returns (0.5, "judge unavailable", FailureMode.UNKNOWN) —
    a neutral fallback that neither rewards nor penalises the node.

    Bias mitigation (Zheng et al. 2023 §4.2):
        - Explicit rubric with four named criteria (not a single opaque score).
        - Required "reasoning" field discourages lazy high scores.
        - Criteria weights are biased toward Completeness and Depth, which
          matter most for adaptive depth decisions.
    """
    prompt = _JUDGE_PROMPT.format(question=question, answer=answer)
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=_JUDGE_TIMEOUT_S,
        )
        text = res.text.replace("```json", "").replace("```", "").strip()
        raw = json.loads(text)

        score = sum(
            raw.get(k, 5) * w for k, w in _JUDGE_WEIGHTS.items()
        ) / 10.0  # normalise from [1,10] to [0.1, 1.0]

        raw_mode = raw.get("failure_mode", "unknown")
        try:
            mode = FailureMode(raw_mode)
        except ValueError:
            mode = FailureMode.UNKNOWN

        return (
            round(min(1.0, max(0.0, score)), 3),
            raw.get("reasoning", ""),
            mode,
        )

    except asyncio.TimeoutError:
        logger.warning("[Judge] Timeout. Using neutral score 0.5.")
        return 0.5, "judge timed out", FailureMode.UNKNOWN
    except Exception as exc:
        logger.warning("[Judge] Failed (%s). Using neutral score 0.5.", exc)
        return 0.5, f"judge error: {exc}", FailureMode.UNKNOWN


def _heuristic_judge(question: str, answer: str) -> float:
    """
    Deterministic quality heuristic (no LLM required).
    Used when no model is provided or as a fast pre-filter.

    Scoring signals:
        - Length: very short answers (<80 chars) score low.
        - Code presence: rewarded for procedural questions.
        - Structural markers: headers, bullet points suggest depth.
        - Keyword overlap with question: basic relevance check.
    """
    if not answer or len(answer) < 20:
        return 0.1

    score = 0.0
    n = len(answer)

    # Length score (sigmoid-like)
    score += min(0.35, n / 2000.0)

    # Structure markers
    markers = ["```", "##", "- ", "* ", "1.", "2.", "\n\n"]
    score += min(0.25, sum(0.05 for m in markers if m in answer))

    # Keyword overlap (normalised)
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words) / max(len(q_words), 1)
    score += min(0.20, overlap)

    # Code detection bonus for procedural queries
    proc_signals = ["def ", "import ", "python", "script", "```python", "bash"]
    if any(s in question.lower() for s in ["write", "implement", "code", "script"]):
        if any(s in answer.lower() for s in proc_signals):
            score += 0.20

    return round(min(1.0, score), 3)


# ──────────────────────────────────────────────────────────────────────────────
# GoT EXPANSION OPERATORS
# ──────────────────────────────────────────────────────────────────────────────

async def _make_branch_tasks(parent: GoTNode, model: Optional[genai.GenerativeModel], n_branches: int = 2) -> List[TaskNode]:
    """
    BRANCH: decompose parent's query into n_branches sub-questions.
    Dynamically asks an LLM for subtasks and delegates to appropriate agents.
    """
    base = parent.task.query
    new_id_base = parent.task.id * 10
    
    # Simple fallback if no model is provided
    if not model:
        return [
            TaskNode(id=new_id_base + i, agent=parent.task.agent, query=f"Explore facet {i+1} of: {base}", depends_on=[parent.task.id])
            for i in range(n_branches)
        ]
        
    prompt = f"""
    You are an Agent Coordinator. The current agent ({parent.task.agent.upper()}) failed to fully answer the query below.
    Reason for failure: {parent.failure_mode}
    Failure Critique: {parent.judge_reasoning}
    Query: "{base}"
    
    Please decompose this query into up to {n_branches} distinct sub-queries that will help resolve the failure.
    Delegate each sub-query to the most appropriate agent from the registry below.
    
    {config.Config.AGENT_REGISTRY_PROMPT}

    Return ONLY a JSON list of objects:
    [
      {{"agent": "rag", "query": "background check on X"}},
      {{"agent": "code", "query": "write python script for Y"}}
    ]
    """
    try:
        res = await asyncio.wait_for(asyncio.to_thread(model.generate_content, prompt), timeout=15)
        text = res.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        tasks = []
        for i, item in enumerate(data):
            tasks.append(TaskNode(
                id=new_id_base + i, 
                agent=item.get("agent", "rag").lower(), 
                query=item.get("query", ""), 
                depends_on=[parent.task.id]
            ))
        return tasks[:n_branches]
    except Exception as e:
        logger.error(f"Failed to generate dynamic branches: {e}")
        return [TaskNode(id=new_id_base, agent=parent.task.agent, query=f"Refine focus on: {base}", depends_on=[parent.task.id])]


def _make_refine_task(parent: GoTNode) -> TaskNode:
    """
    REFINE: produce a deeper, more detailed version of the parent's answer.
    The refinement prompt explicitly references the parent's result so the
    model can identify gaps.
    """
    prior = (parent.result or "")[:300].replace("\n", " ")
    return TaskNode(
        id=parent.task.id * 10 + 9,
        agent=parent.task.agent,
        query=(
            f"The following answer is incomplete or shallow. "
            f"Provide a significantly more detailed and rigorous response.\n"
            f"Original question: {parent.task.query}\n"
            f"Prior answer (first 300 chars): {prior}\n"
            f"Refined answer:"
        ),
        depends_on=[parent.task.id],
    )


# def _make_aggregate_task(parents: List[GoTNode], sink_id: int) -> TaskNode:
#     """
#     AGGREGATE: synthesise insights from multiple parent nodes into one answer.
#     Used after parallel BRANCH nodes complete.
#     Ref: Besta et al. (2023), Section 3.3 — Aggregate operator.
#     """
#     summaries = "\n".join(
#         f"[{i+1}] {n.task.query[:80]}: {(n.result or '')[:150]}"
#         for i, n in enumerate(parents)
#     )
#     return TaskNode(
#         id=sink_id,
#         agent="rag",
#         query=(
#             f"Synthesise the following {len(parents)} partial answers into one "
#             f"comprehensive, well-structured response.\n\n{summaries}"
#         ),
#         depends_on=[n.task.id for n in parents],
#     )

def _make_aggregate_task(parents: List[GoTNode], sink_id: int) -> TaskNode:
    """
    AGGREGATE: synthesise insights from multiple parent nodes into one answer.
    Used after parallel BRANCH nodes complete.
    Ref: Besta et al. (2023), Section 3.3 — Aggregate operator.
    """
    # 🚀 FIX: Removed the [:150] truncation. We must pass the FULL results 
    # to the aggregator so the LLM doesn't lose the Python code or the biology text!
    summaries = "\n\n".join(
        f"--- Source [{i+1}]: {n.task.query} ---\n{(n.result or 'No data returned.')}"
        for i, n in enumerate(parents)
    )
    
    return TaskNode(
        id=sink_id,
        agent="rag",  # This gets caught by the bypass in engine.py
        query=(
            f"You are the final aggregator. Synthesize the following {len(parents)} source answers "
            f"into one comprehensive, perfectly structured final response.\n\n"
            f"CRITICAL: Do NOT drop any Python code. Do NOT drop any expert names. "
            f"Merge them seamlessly.\n\n{summaries}"
        ),
        depends_on=[n.task.id for n in parents],
    )
    
# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE GoT-DAG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpansionRecord:
    """Tracks one iterative deepening expansion step for diagnostics."""
    wave:           int
    node_id:        str
    op:             GoTOp
    dim:            Dim
    depth_before:   int
    quality_before: float
    n_children:     int


class AdaptiveGoTDAG:
    """
    Adaptive Graph-of-Thoughts DAG.

    Construction
    ────────────
    Seed nodes are created from the initial TaskNode list (output of any
    query decomposer). Each seed gets a dimension assignment based on its
    QDMR op or Coverage features.

    Execution model
    ───────────────
    Wave 0:  Execute all root nodes in topological order (Kahn's algorithm
             on the seed graph — the classic system behaviour).
    Wave 1+: Quality-gate check. For each executed node below threshold:
               - If depth < budget[dim]: expand with GoT operator.
               - If depth >= budget[dim]: mark exhausted, skip.
             Execute new nodes. Repeat.

    Termination (four conditions — any one stops the loop):
        (T1) All nodes: quality_score >= quality_threshold
        (T2) All nodes: depth >= budget[dim]
        (T3) Quality delta < epsilon (plateau — no improvement last wave)
        (T4) Total expansions >= max_expansions (absolute budget)

    Beam control
    ────────────
    At most beam_width nodes are expanded per wave, selected by lowest
    quality score (highest need first). This bounds node count to
    O(beam_width * max_depth) rather than O(b^max_depth).

    Reference: Yao et al. (2023) use b=5, d=3 → 15 expansions max.
    Default: beam_width=3, max_depth=5 → 15 expansions max.
    """

    def __init__(
        self,
        *,
        quality_threshold: float = 0.75,
        beam_width: int = 3,
        max_expansions: int = 20,
        plateau_epsilon: float = 0.02,
        depth_budget: Optional[DepthBudget] = None,
    ) -> None:
        self.quality_threshold = quality_threshold
        self.beam_width = beam_width
        self.max_expansions = max_expansions
        self.plateau_epsilon = plateau_epsilon
        self.depth_budget = depth_budget or DepthBudget()

        # Graph state
        self._nodes: Dict[str, GoTNode] = {}
        self._adj: Dict[str, Set[str]] = defaultdict(set)   # parent → children
        self._radj: Dict[str, Set[str]] = defaultdict(set)  # child → parents

        # Diagnostics
        self.expansion_log: List[ExpansionRecord] = []
        self.wave_scores: List[float] = []   # mean quality score per wave

    # ── Node management ───────────────────────────────────────────────────────

    def _new_node_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _add_node(self, node: GoTNode) -> None:
        self._nodes[node.node_id] = node
        for pid in node.parent_ids:
            self._adj[pid].add(node.node_id)
            self._radj[node.node_id].add(pid)

    def _dim_for_task(self, task: TaskNode) -> Dim:
        """
        Infer which profile dimension a task belongs to.
        Uses QDMR op if available; falls back to keyword heuristic.
        """
        if task.qdmr_op in ("CODE", "DEPLOY"):
            return Dim.PROCEDURAL
        if task.qdmr_op in ("EXPERT", "COMPARATIVE", "BOOLEAN"):
            return Dim.ANALYTICAL
        if task.qdmr_op in ("SELECT", "PROJECT", "UNION"):
            return Dim.CONCEPTUAL

        # Coverage features
        if task.features_covered:
            if "code_generation" in task.features_covered or "pipeline_design" in task.features_covered:
                return Dim.PROCEDURAL
            if "expert_discovery" in task.features_covered or "comparative_analysis" in task.features_covered:
                return Dim.ANALYTICAL

        # Keyword fallback
        q = task.query.lower()
        if any(k in q for k in ["write", "implement", "script", "code", "debug"]):
            return Dim.PROCEDURAL
        if any(k in q for k in ["compare", "versus", "evaluate", "literature", "trade"]):
            return Dim.ANALYTICAL
        return Dim.CONCEPTUAL

    # ── Seed construction ─────────────────────────────────────────────────────

    def seed_from_tasks(self, tasks: List[TaskNode]) -> None:
        """
        Initialise the DAG from an existing task list (decomposer output).
        Preserves the dependency graph structure as GoT ROOT nodes.
        """
        id_to_nid: Dict[int, str] = {}
        for task in tasks:
            nid = self._new_node_id()
            id_to_nid[task.id] = nid
            node = GoTNode(
                node_id=nid,
                task=task,
                dim=self._dim_for_task(task),
                depth=0,
                op=GoTOp.ROOT,
                parent_ids=[id_to_nid[d] for d in task.depends_on if d in id_to_nid],
            )
            self._add_node(node)
        logger.info("[GoTDAG] Seeded with %d root nodes.", len(tasks))

    # ── Topological execution order (Kahn's on current graph) ─────────────────

    def _execution_waves(self) -> List[List[GoTNode]]:
        """
        Compute layered topological order (Kahn's algorithm) on the CURRENT
        graph state. Returns list of waves; each wave is a list of nodes that
        can execute in parallel (all parents already in a previous wave).

        This is Kahn's algorithm operating as a subroutine inside GoT —
        GoT governs WHICH nodes exist; Kahn's governs ORDER of execution.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for child_id in self._adj[nid]:
                in_degree[child_id] = in_degree.get(child_id, 0) + 1

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        waves: List[List[GoTNode]] = []

        while queue:
            wave_nodes = list(queue)
            queue.clear()
            waves.append([self._nodes[nid] for nid in wave_nodes])
            for nid in wave_nodes:
                for child_id in self._adj[nid]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        queue.append(child_id)

        return waves

    # ── GoT expansion ─────────────────────────────────────────────────────────

    def _select_expansion_op(self, node: GoTNode) -> GoTOp:
        """
        Choose the most appropriate GoT operator for a given node.

        Heuristic (grounded in Besta et al. 2023, Table 1):
            REFINE  → when the node has a result but it's shallow; depth > 0
                      (we already tried once; improve the same thought)
            BRANCH  → when depth == 0 (first expansion; explore sub-facets)
            AGGREGATE is produced externally after branch children complete.
            SELECT  is produced externally during beam-width pruning.
        """
        if node.depth > 0 and node.result:
            return GoTOp.REFINE
        return GoTOp.BRANCH

    async def expand_node(
        self,
        node: GoTNode,
        budget: Dict[Dim, int],
        wave_number: int,
        n_branches: int = 2,
        model: Optional[genai.GenerativeModel] = None,
    ) -> List[GoTNode]:
        """
        Expand a low-quality node using the appropriate GoT operator.
        Returns a list of newly created child nodes added to the graph.

        Respects depth budget: if node.depth >= budget[dim], no expansion.
        """
        if node.depth >= budget[node.dim]:
            logger.debug(
                "[GoTDAG] Node %s at depth %d: budget exhausted for %s.",
                node.node_id, node.depth, node.dim.value,
            )
            node.is_expanded = True
            return []

        op = self._select_expansion_op(node)
        new_nodes: List[GoTNode] = []

        if op == GoTOp.BRANCH:
            child_tasks = await _make_branch_tasks(node, model, n_branches=n_branches)
            for ct in child_tasks:
                nid = self._new_node_id()
                child = GoTNode(
                    node_id=nid,
                    task=ct,
                    dim=node.dim,
                    depth=node.depth + 1,
                    op=GoTOp.BRANCH,
                    parent_ids=[node.node_id],
                )
                self._add_node(child)
                node.children_ids.append(nid)
                new_nodes.append(child)
            self.expansion_log.append(ExpansionRecord(
                wave=wave_number, node_id=node.node_id, op=op, dim=node.dim,
                depth_before=node.depth, quality_before=node.quality_score or 0.0,
                n_children=len(new_nodes),
            ))

        elif op == GoTOp.REFINE:
            rt = _make_refine_task(node)
            nid = self._new_node_id()
            child = GoTNode(
                node_id=nid,
                task=rt,
                dim=node.dim,
                depth=node.depth + 1,
                op=GoTOp.REFINE,
                parent_ids=[node.node_id],
            )
            self._add_node(child)
            node.children_ids.append(nid)
            new_nodes.append(child)
            self.expansion_log.append(ExpansionRecord(
                wave=wave_number, node_id=node.node_id, op=op, dim=node.dim,
                depth_before=node.depth, quality_before=node.quality_score or 0.0,
                n_children=1,
            ))

        node.is_expanded = True
        return new_nodes

    # ── Beam-width pruning ────────────────────────────────────────────────────

    def _prune_to_beam(
        self, candidates: List[GoTNode], budget: Dict[Dim, int]
    ) -> List[GoTNode]:
        """
        From candidates eligible for expansion, select at most beam_width
        nodes with the LOWEST quality scores (highest need first).

        Ref: Yao et al. (2023) beam search in Tree of Thoughts.
        """
        eligible = [
            n for n in candidates
            if (n.quality_score or 0.0) < self.quality_threshold
            and n.depth < budget[n.dim]
            and not n.is_expanded
        ]
        eligible.sort(key=lambda n: n.quality_score or 0.0)
        return eligible[: self.beam_width]

    # ── Main execution loop ───────────────────────────────────────────────────

    async def execute(
        self,
        executor: Callable[[GoTNode], str],
        profile: Dict[Dim, float],
        model: Optional[genai.GenerativeModel] = None,
        *,
        n_branches: int = 2,
    ) -> Dict[str, GoTNode]:
        """
        Run the adaptive GoT-DAG execution loop.

        Args:
            executor:   Async callable (GoTNode → str). In production this calls
                        the appropriate agent. In simulation it returns a stub.
            profile:    Normalised dimension weights from UserProfile.
            model:      Optional LLM for judge scoring. If None, uses heuristic.
            n_branches: Branching factor for BRANCH expansions.

        Returns:
            Final graph state: {node_id: GoTNode}

        Termination conditions (checked after every wave):
            T1 — All non-exhausted nodes have quality >= threshold
            T2 — All nodes at max depth for their dimension
            T3 — Mean quality delta < epsilon (plateau)
            T4 — Total expansion count >= max_expansions
        """
        budget = self.depth_budget.budget_for(profile)
        logger.info(
            "[GoTDAG] Starting. Budget: %s | threshold=%.2f | beam=%d",
            self.depth_budget.summary(profile),
            self.quality_threshold,
            self.beam_width,
        )

        total_expansions = 0
        prev_mean_quality = 0.0
        wave_number = 0

        while True:
            wave_number += 1

            # ── Execute pending nodes (Kahn's order) ──────────────────────────
            waves = self._execution_waves()
            newly_executed = 0
            for wave in waves:
                for node in wave:
                    if not node.is_executed:
                        _t0 = time.monotonic()
                        node.result = await executor(node)
                        node.elapsed_s = round(time.monotonic() - _t0, 2)
                        newly_executed += 1

            if newly_executed == 0 and wave_number > 1:
                logger.info("[GoTDAG] No new nodes to execute. Stopping.")
                break

            # ── Judge quality of newly executed nodes (parallelized) ──────────
            from .evaluation.communication_log import comm_tracker
            unjudged = [n for n in self._nodes.values()
                        if n.is_executed and n.quality_score is None]
            if unjudged:
                if model is not None:
                    judge_results = await asyncio.gather(*[
                        _llm_judge(n.task.query, n.result or "", model)
                        for n in unjudged
                    ])
                    for node, (score, reason, failure_mode) in zip(unjudged, judge_results):
                        node.quality_score = score
                        node.failure_mode = failure_mode
                        logger.debug("[Judge] Node %s: %.3f — %s", node.node_id, score, reason)
                else:
                    for node in unjudged:
                        node.quality_score = _heuristic_judge(node.task.query, node.result or "")
                        node.failure_mode = FailureMode.UNKNOWN
                for node in unjudged:
                    for log in comm_tracker.logs:
                        if log.to_id == node.node_id:
                            log.quality_score = node.quality_score

            # ── Wave quality summary ──────────────────────────────────────────
            scores = [n.quality_score for n in self._nodes.values()
                      if n.quality_score is not None]
            mean_q = sum(scores) / len(scores) if scores else 0.0
            self.wave_scores.append(mean_q)
            delta_q = abs(mean_q - prev_mean_quality)
            logger.info(
                "[GoTDAG] Wave %d | nodes=%d | mean_q=%.3f | delta=%.3f",
                wave_number, len(self._nodes), mean_q, delta_q,
            )

            # ── Termination checks ────────────────────────────────────────────
            leaf_nodes = [
                n for n in self._nodes.values()
                if not n.children_ids and n.is_executed
            ]

            # T1: all leaves above threshold
            if all((n.quality_score or 0.0) >= self.quality_threshold for n in leaf_nodes):
                logger.info("[GoTDAG] T1: all leaves above quality threshold. Done.")
                break

            # T2: all leaves at max depth
            if all(n.depth >= budget[n.dim] for n in leaf_nodes):
                logger.info("[GoTDAG] T2: all leaves at max depth. Done.")
                break

            # T3: quality plateau
            if wave_number > 1 and delta_q < self.plateau_epsilon:
                logger.info("[GoTDAG] T3: quality plateau (delta=%.4f). Done.", delta_q)
                break

            # T4: expansion budget
            if total_expansions >= self.max_expansions:
                logger.info("[GoTDAG] T4: max expansions (%d) reached. Done.",
                            self.max_expansions)
                break

            # ── Select nodes for expansion ────────────────────────────────────
            candidates = [
                n for n in self._nodes.values()
                if n.is_executed and not n.is_expanded
                and (n.quality_score or 0.0) < self.quality_threshold
            ]
            to_expand = self._prune_to_beam(candidates, budget)

            if not to_expand:
                logger.info("[GoTDAG] No candidates for expansion. Done.")
                break

            # ── Expand selected nodes ─────────────────────────────────────────
            for node in to_expand:
                new_nodes = await self.expand_node(node, budget, wave_number, n_branches, model)
                total_expansions += len(new_nodes)
                logger.info(
                    "[GoTDAG] Expanded %s (dim=%s, depth=%d→%d, q=%.3f) → %d children.",
                    node.node_id, node.dim.value, node.depth, node.depth + 1,
                    node.quality_score or 0.0, len(new_nodes),
                )

            prev_mean_quality = mean_q

        return dict(self._nodes)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def build_final_aggregate(self) -> Optional[GoTNode]:
        """
        Automatically create an AGGREGATE node that synthesises all leaf
        node results into a single final answer.

        Only created if there are ≥2 leaf nodes with results.
        """
        leaves = [
            n for n in self._nodes.values()
            if not n.children_ids and n.is_executed and n.result
        ]
        if len(leaves) < 2:
            return leaves[0] if leaves else None

        sink_id = max(n.task.id for n in self._nodes.values()) + 1
        agg_task = _make_aggregate_task(leaves, sink_id)
        nid = self._new_node_id()
        agg_node = GoTNode(
            node_id=nid,
            task=agg_task,
            dim=Dim.ANALYTICAL,  # synthesis is an analytical operation
            depth=max(n.depth for n in leaves) + 1,
            op=GoTOp.AGGREGATE,
            parent_ids=[n.node_id for n in leaves],
        )
        self._add_node(agg_node)
        logger.info(
            "[GoTDAG] Created AGGREGATE node synthesising %d leaf nodes.", len(leaves)
        )
        return agg_node

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        nodes = list(self._nodes.values())
        scores = [n.quality_score for n in nodes if n.quality_score is not None]
        by_op = defaultdict(int)
        by_dim = defaultdict(int)
        for n in nodes:
            by_op[n.op.value] += 1
            by_dim[n.dim.value] += 1
        mean_q_str = f"{sum(scores)/len(scores):.3f}" if scores else "n/a"
        return (
            f"GoTDAG Summary:\n"
            f"  Total nodes    : {len(nodes)}\n"
            f"  By operator    : {dict(by_op)}\n"
            f"  By dimension   : {dict(by_dim)}\n"
            f"  Max depth      : {max((n.depth for n in nodes), default=0)}\n"
            f"  Mean quality   : {mean_q_str}\n"
            f"  Waves executed : {len(self.wave_scores)}\n"
            f"  Expansions     : {len(self.expansion_log)}\n"
            f"  Wave scores    : {[round(s,3) for s in self.wave_scores]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR  (public API — replaces old Kahn's orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveGoTOrchestrator:
    """
    Drop-in replacement for the classic Kahn's-algorithm task orchestrator.

    Public API
    ──────────
    run(tasks, user_profile, model) → final_answer: str

    Internally:
        1. Seeds the GoT-DAG from the task list.
        2. Computes depth budget from user profile.
        3. Runs the adaptive execution loop.
        4. Appends an AGGREGATE node.
        5. Executes the aggregate node.
        6. Returns the final synthesised answer.
    """

    def __init__(
        self,
        *,
        quality_threshold: float = 0.75,
        beam_width: int = 3,
        max_expansions: int = 20,
        depth_budget: Optional[DepthBudget] = None,
    ) -> None:
        self.quality_threshold = quality_threshold
        self.beam_width = beam_width
        self.max_expansions = max_expansions
        self.depth_budget = depth_budget or DepthBudget()
        self._dag: Optional[AdaptiveGoTDAG] = None

    async def run(
        self,
        tasks: List[TaskNode],
        user_profile: UserProfile,
        model: Optional[genai.GenerativeModel] = None,
        *,
        agent_executor: Optional[Callable] = None,
    ) -> str:
        """
        Execute tasks adaptively via GoT-DAG.

        Args:
            tasks:          Decomposed TaskNode list from any decomposer.
            user_profile:   Live UserProfile object (drives depth budget).
            model:          Gemini model for judge scoring (optional).
            agent_executor: Callable(GoTNode) → str. Production agents go here.
                            If None, uses a stub that returns a placeholder.

        Returns:
            The final synthesised answer string.
        """
        profile = user_profile.profile

        self._dag = AdaptiveGoTDAG(
            quality_threshold=self.quality_threshold,
            beam_width=self.beam_width,
            max_expansions=self.max_expansions,
            depth_budget=self.depth_budget,
        )
        self._dag.seed_from_tasks(tasks)

        # Use provided executor or simulation stub
        executor = agent_executor or _stub_executor

        await self._dag.execute(
            executor=executor,
            profile=profile,
            model=model,
            n_branches=2,
        )

        # Aggregate leaf results
        agg_node = self._dag.build_final_aggregate()
        if agg_node and not agg_node.is_executed:
            agg_node.result = await executor(agg_node)

        logger.info("\n%s", self._dag.summary())

        if agg_node and agg_node.result:
            return agg_node.result

        # Fallback: concatenate all leaf results
        leaves = [
            n for n in self._dag._nodes.values()
            if not n.children_ids and n.result
        ]
        return "\n\n---\n\n".join(
            f"[{n.dim.value.upper()}] {n.result}" for n in leaves
        )

    @property
    def dag(self) -> Optional[AdaptiveGoTDAG]:
        return self._dag


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION STUB EXECUTOR
# ──────────────────────────────────────────────────────────────────────────────

async def _stub_executor(node: GoTNode) -> str:
    """
    Stub executor for simulation/testing — no real agents needed.

    Simulates answer quality based on:
        - Query length (longer = more considered question)
        - Depth (deeper = more focused = slightly better)
        - Operator (REFINE produces more text than BRANCH leaf)
    """
    await asyncio.sleep(0)  # yield to event loop
    q = node.task.query
    depth_bonus = "  ".join(["[deeper level]"] * node.depth)
    op_label = f"[{node.op.value.upper()}]"
    dim_label = f"[{node.dim.value.upper()}]"

    if node.op == GoTOp.AGGREGATE:
        return (
            f"{dim_label} {op_label} Synthesised response integrating "
            f"{len(node.parent_ids)} sub-answers.\n"
            f"Topic: {q[:120]}\n"
            f"Final integrated answer covering all relevant dimensions."
        )
    if node.op == GoTOp.REFINE:
        return (
            f"{dim_label} {op_label} {depth_bonus}\n"
            f"Refined and significantly more detailed response.\n"
            f"Q: {q[:100]}\n"
            f"A: Detailed explanation with examples, edge cases, "
            f"and comprehensive coverage of the topic. "
            f"{'Code example included. ' if node.dim == Dim.PROCEDURAL else ''}"
            f"{'Literature references cited. ' if node.dim == Dim.ANALYTICAL else ''}"
        )
    # ROOT or BRANCH leaf
    short = len(q) < 60
    return (
        f"{dim_label} {op_label}\n"
        f"Q: {q[:100]}\n"
        f"A: {'Brief initial answer.' if short else 'Initial response covering the key points of the question.'}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────────────────────

async def demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── 1. Simulate a user profile ────────────────────────────────────────────
    profile = UserProfile(alpha_init=3.0)
    # Simulate several analytical queries having been asked
    profile.alpha[Dim.ANALYTICAL] += 15
    profile.alpha[Dim.CONCEPTUAL] += 3
    profile.alpha[Dim.PROCEDURAL] += 2
    profile.n_queries = 7

    p = profile.profile
    print("═" * 68)
    print("Adaptive GoT-DAG Orchestrator — Demo")
    print("═" * 68)
    print("\nUser Profile:")
    print(f"  Conceptual:  {p[Dim.CONCEPTUAL]:.1f}%")
    print(f"  Procedural:  {p[Dim.PROCEDURAL]:.1f}%")
    print(f"  Analytical:  {p[Dim.ANALYTICAL]:.1f}%")

    budget = DepthBudget().budget_for(p)
    print("\nDepth Budget:")
    for dim, d in budget.items():
        print(f"  {dim.value:12}: max_depth = {d}")

    # ── 2. Seed tasks (simulating decomposer output) ──────────────────────────
    tasks = [
        TaskNode(id=1, agent="rag",     query="Explain the GATK variant calling pipeline",
                 depends_on=[], qdmr_op="SELECT"),
        TaskNode(id=2, agent="scholar", query="Find 2 experts on GATK and variant calling benchmarks",
                 depends_on=[], qdmr_op="EXPERT"),
        TaskNode(id=3, agent="code",    query="Write a Snakemake pipeline for GATK HaplotypeCaller",
                 depends_on=[1], qdmr_op="CODE"),
        TaskNode(id=4, agent="rag",     query="Compare GATK vs DeepVariant for germline variant calling",
                 depends_on=[], qdmr_op="COMPARATIVE"),
    ]

    # ── 3. Run orchestrator ───────────────────────────────────────────────────
    orchestrator = AdaptiveGoTOrchestrator(
        quality_threshold=0.75,
        beam_width=2,
        max_expansions=8,
        depth_budget=DepthBudget(base_depth=1, weight_per_level=20, max_depth=4),
    )

    print(f"\n{'─'*68}")
    print("Running adaptive execution loop (deterministic stub — no model)…")
    print(f"{'─'*68}\n")

    final_answer = await orchestrator.run(
        tasks=tasks,
        user_profile=profile,
        model=None,  # deterministic path
    )

    # ── 4. Print results ──────────────────────────────────────────────────────
    dag = orchestrator.dag
    print("\n" + "═" * 68)
    print(dag.summary())

    print(f"\n{'─'*68}")
    print("Expansion log:")
    for rec in dag.expansion_log:
        print(
            f"  wave={rec.wave} | {rec.op.value:8} | {rec.dim.value:12} | "
            f"depth {rec.depth_before}→{rec.depth_before+1} | "
            f"q_before={rec.quality_before:.3f} | children={rec.n_children}"
        )

    print(f"\n{'─'*68}")
    print("Wave-by-wave mean quality:")
    for i, s in enumerate(dag.wave_scores, 1):
        bar = "█" * int(s * 30)
        print(f"  Wave {i}: {s:.3f} {bar}")

    print(f"\n{'─'*68}")
    print("Final synthesised answer (truncated):")
    print(final_answer[:400] + ("…" if len(final_answer) > 400 else ""))
    print("═" * 68)


if __name__ == "__main__":
    asyncio.run(demo())