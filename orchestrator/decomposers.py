"""
query_decomposer.py
═══════════════════════════════════════════════════════════════════════════════
Multi-strategy Query Decomposition for Bioinformatics AI Agents

Four decomposition strategies:

  Strategy 1 — LLM-based (probabilistic)
    ├── PlanAndSolveDecomposer  [Wang et al., 2023 — Plan-and-Solve Prompting]
    └── LLMCompilerDecomposer   [Kim et al., 2023 — LLMCompiler]

  Strategy 2 — QDMR Rule-based (fully deterministic)
    └── QDMRRuleDecomposer      [Wolfson et al., 2020; Khot et al., 2021]
        • Clause segmentation via RST heuristics (Marcu 1997)
        • QDMR op classification via ordered regex rules
        • Static agent routing table
        • Dependency inference via QDMR @@REF@@ convention

  Strategy 3 — Coverage-based (novel, deterministic)
    └── CoverageDecomposer
        • Represents the query as a bit-vector of information-need features
        • Frames decomposition as Minimum Set Cover (Chvatal 1979)
        • Greedy approximation guarantees all needs are addressed
        • Inspired by MMR (Carbonell & Goldstein 1998) and VisProg (Suris 2023)

  Strategy 4 — Neural+Rule hybrid (LLM for DAG, deterministic for routing)
    └── NeuroRuleDecomposer
        • Phase 1 (neural): LLM performs coreference resolution and produces
          a clean, dependency-annotated goal DAG — one intent per goal.
        • Phase 2 (deterministic): QDMR regex rules route each LLM goal to
          an agent and inject required strategy/count annotations.
        • Timeout-guarded LLM call with graceful QDMR fallback.

References
──────────
  Wolfson et al. (2020). Break It Down: A Question Understanding Benchmark. TACL.
  Khot et al. (2021). Text Modular Networks. EMNLP.
  Perez et al. (2020). Unsupervised Question Decomposition for QA. EMNLP.
  Marcu (1997). The Rhetorical Parsing of Natural Language.
  Wang et al. (2023). Plan-and-Solve Prompting. ACL.
  Kim et al. (2023). An LLM Compiler for Parallel Function Calling. ICML.
  Chvatal (1979). A Greedy Heuristic for the Set-Covering Problem. Math OR.
  Carbonell & Goldstein (1998). The Use of MMR for Reranking Documents. SIGIR.
  Suris et al. (2023). VisProg: Compositional Visual Programming. CVPR.
  Garcez & Lamb (2023). Neurosymbolic AI: The 3rd Wave. AI Review.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SHARED: agent knowledge base + task node
# ──────────────────────────────────────────────────────────────────────────────

AGENT_GUIDELINES = """
AVAILABLE AGENTS & CAPABILITIES:
1. "rag" - Use for general knowledge, definitions, tool explanations, and concepts in bioinformatics.
   - Feature: You MUST append a reasoning strategy to the query.
     - If query asks for a single fact -> append "--strategy=cot"
     - If query asks to verify a claim -> append "--strategy=react"
     - If query is open-ended or asks for multiple items -> append "--strategy=tot"
     - If query asks to compare distinct concepts -> append "--strategy=got"
2. "scholar" - Use ONLY for finding actual human experts, research papers, citations, and collaborators.
   - Feature: Always specify the number of experts in the sub_query (e.g., "find 2 experts on X").
3. "code" - Use for writing Python scripts, debugging, and coding tasks.
4. "cloud" - Use for AWS, deployment, and infrastructure.
"""


@dataclass
class TaskNode:
    id: int
    agent: str
    query: str
    depends_on: List[int] = field(default_factory=list)
    # Optional metadata populated by QDMR and Coverage decomposers
    qdmr_op: Optional[str] = None
    features_covered: Optional[FrozenSet[str]] = None


class BaseDecomposer:
    name: str = "base"

    async def decompose(
        self, query: str, model: Optional[genai.GenerativeModel] = None
    ) -> List[TaskNode]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — LLM-based decomposers (probabilistic)
# ──────────────────────────────────────────────────────────────────────────────


class PlanAndSolveDecomposer(BaseDecomposer):
    """
    Plan-and-Solve Prompting — generates flat, parallel tasks via an LLM.

    Reference:
        Wang et al. (2023). Plan-and-Solve Prompting: Improving Zero-Shot
        Chain-of-Thought Reasoning by Large Language Models. ACL 2023.

    Notes:
        - Probabilistic: same query may yield different decompositions.
        - No explicit dependency graph; all tasks are assumed independent.
        - Falls back to a single RAG task on JSON parse failure.
    """

    name = "plan_and_solve"

    async def decompose(
        self, query: str, model: genai.GenerativeModel
    ) -> List[TaskNode]:
        prompt = f"""
        Break down the user's query into sub-tasks.
        {AGENT_GUIDELINES}

        Return ONLY JSON in this format:
        {{"tasks":[{{"id": 1, "agent": "rag", "query": "..."}}]}}
        """
        res = await asyncio.to_thread(
            model.generate_content, f"{prompt}\nUser Query: {query}"
        )
        text = res.text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text)
            return [
                TaskNode(id=t["id"], agent=t["agent"], query=t["query"])
                for t in data["tasks"]
            ]
        except Exception:
            return [TaskNode(id=1, agent="rag", query=query)]


class LLMCompilerDecomposer(BaseDecomposer):
    """
    LLMCompiler — generates tasks with explicit inter-task dependencies via LLM.

    Reference:
        Kim et al. (2023). An LLM Compiler for Parallel Function Calling.
        ICML 2024 Workshop.

    Notes:
        - Probabilistic: dependency graph structure varies across runs.
        - Supports cross-task placeholder references ("@@REF@@"-style).
        - Falls back to a single RAG task on JSON parse failure.
    """

    name = "llm_compiler"

    async def decompose(
        self, query: str, model: genai.GenerativeModel
    ) -> List[TaskNode]:
        prompt = f"""
        You are an LLMCompiler Orchestrator. Break the complex query into a
        Dependency Graph of sub-tasks.
        {AGENT_GUIDELINES}

        INSTRUCTIONS:
        1. If a task requires information from a previous task, add the previous
           task's ID to "depends_on" and use a placeholder like
           "[tool from task 1]" in the query.
        2. Pay strict attention to the Agent Capabilities. Use "--strategy=" for
           rag and specify expert counts for scholar.

        Return exactly this JSON format:
        {{
          "tasks":[
            {{"id": 1, "agent": "rag",
              "query": "What are the key applications of bioinformatics? --strategy=tot",
              "depends_on": []}},
            {{"id": 2, "agent": "scholar",
              "query": "Find 2 experts on [application from task 1]",
              "depends_on": [1]}}
          ]
        }}
        """
        res = await asyncio.to_thread(
            model.generate_content, f"{prompt}\nUser Query: {query}"
        )
        text = res.text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text)
            return [
                TaskNode(
                    id=t["id"],
                    agent=t["agent"],
                    query=t["query"],
                    depends_on=t.get("depends_on", []),
                )
                for t in data["tasks"]
            ]
        except Exception as e:
            logger.error("[LLMCompilerDecomposer] JSON parse failed: %s", e)
            return [TaskNode(id=1, agent="rag", query=query)]


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — QDMR Rule-Based Decomposer (deterministic)
#
# Literature basis:
#   Wolfson et al. (2020). Break It Down: A Question Understanding Benchmark.
#       Transactions of the ACL.
#   Khot et al. (2021). Text Modular Networks: Learning to Decompose Tasks in
#       the Language of Existing Tools. EMNLP 2021.
#   Perez et al. (2020). Unsupervised Question Decomposition for QA. EMNLP.
#   Marcu (1997). The Rhetorical Parsing of Natural Language.
# ──────────────────────────────────────────────────────────────────────────────


class QDMROp(Enum):
    """
    QDMR high-level operations as defined in Wolfson et al. (2020), Table 2.
    Extended with CODE, DEPLOY, and EXPERT for agent routing.
    """

    SELECT = "SELECT"               # retrieve a set of entities or facts
    FILTER = "FILTER"               # restrict a set by a predicate
    PROJECT = "PROJECT"             # retrieve a property of entities
    AGGREGATE = "AGGREGATE"         # count / max / min / sum / avg
    GROUP = "GROUP"                 # partition a set by attribute
    SUPERLATIVE = "SUPERLATIVE"     # highest / lowest / most / least
    COMPARATIVE = "COMPARATIVE"     # compare two or more entities
    UNION = "UNION"                 # combine two sets
    INTERSECTION = "INTERSECTION"   # overlap between sets
    DISCARD = "DISCARD"             # set difference (A minus B)
    SORT = "SORT"                   # order a set
    BOOLEAN = "BOOLEAN"             # yes/no question
    ARITHMETIC = "ARITHMETIC"       # mathematical computation
    # ── Extensions for agent routing ──────────────────────────────────────────
    CODE = "CODE"                   # programmatic / scripting task
    DEPLOY = "DEPLOY"               # cloud / infrastructure task
    EXPERT = "EXPERT"               # find human experts or papers


# Static routing: QDMR operation → agent name
QDMR_OP_TO_AGENT: Dict[QDMROp, str] = {
    QDMROp.SELECT:        "rag",
    QDMROp.FILTER:        "rag",
    QDMROp.PROJECT:       "rag",
    QDMROp.AGGREGATE:     "rag",
    QDMROp.GROUP:         "rag",
    QDMROp.SUPERLATIVE:   "rag",
    QDMROp.COMPARATIVE:   "rag",
    QDMROp.UNION:         "rag",
    QDMROp.INTERSECTION:  "rag",
    QDMROp.DISCARD:       "rag",
    QDMROp.SORT:          "rag",
    QDMROp.BOOLEAN:       "rag",
    QDMROp.ARITHMETIC:    "rag",
    QDMROp.CODE:          "code",
    QDMROp.DEPLOY:        "cloud",
    QDMROp.EXPERT:        "scholar",
}

# Static routing: QDMR operation → RAG reasoning strategy
# (per AGENT_GUIDELINES; only applied when agent == "rag")
QDMR_OP_TO_STRATEGY: Dict[QDMROp, str] = {
    QDMROp.SELECT:        "tot",   # open-ended, multiple items
    QDMROp.FILTER:        "cot",   # single constrained fact
    QDMROp.PROJECT:       "cot",   # single property retrieval
    QDMROp.AGGREGATE:     "cot",   # single computed answer
    QDMROp.GROUP:         "tot",   # multiple groups to enumerate
    QDMROp.SUPERLATIVE:   "cot",   # single best/worst answer
    QDMROp.COMPARATIVE:   "got",   # compare distinct concepts
    QDMROp.UNION:         "tot",   # multiple items from two sets
    QDMROp.INTERSECTION:  "got",   # compare to find overlap
    QDMROp.DISCARD:       "cot",   # single set-difference result
    QDMROp.SORT:          "cot",   # single ordered list
    QDMROp.BOOLEAN:       "react", # verify a claim
    QDMROp.ARITHMETIC:    "cot",   # single numeric result
}

# Ordered pattern rules: first match wins.
# Patterns are ordered from most specific domain signal to most generic.
# Each tuple: (compiled regex, QDMROp)
QDMR_PATTERN_RULES: List[Tuple[re.Pattern, QDMROp]] = [
    # ── Infrastructure / cloud ──────────────────────────────────────────────
    (re.compile(
        r"\b(deploy|aws|s3|ec2|lambda|docker|kubernetes|k8s|"
        r"terraform|cloud infra|serverless|container|helm|ecs|fargate)\b",
        re.IGNORECASE,
    ), QDMROp.DEPLOY),

    # ── Code / scripting ────────────────────────────────────────────────────
    (re.compile(
        r"\b(write( a)?|code|script|implement|program|debug|"
        r"function|class|module|pip install|conda|biopython|"
        r"parse|automate|pipeline code|snakemake|nextflow|"
        r"pandas|numpy|matplotlib|seaborn|scikit)\b",
        re.IGNORECASE,
    ), QDMROp.CODE),

    # ── Expert / paper finding ──────────────────────────────────────────────
    (re.compile(
        r"\b(expert|researcher|paper|publication|citation|"
        r"author|journal|who works? on|find.*who|recent study|"
        r"literature review|bibliograph)\b",
        re.IGNORECASE,
    ), QDMROp.EXPERT),

    # ── Boolean / verification ──────────────────────────────────────────────
    (re.compile(
        r"\b(is it|does it|can it|verify|true or false|confirm|"
        r"check if|is there|are there|does .{1,40} exist|"
        r"is .{1,30} valid|validate)\b",
        re.IGNORECASE,
    ), QDMROp.BOOLEAN),

    # ── Comparative ─────────────────────────────────────────────────────────
    (re.compile(
        r"\b(compare|versus|vs\.?|difference between|similarities|"
        r"contrast|which is better|pros and cons|trade.?off|"
        r"advantages? of .{1,20} over)\b",
        re.IGNORECASE,
    ), QDMROp.COMPARATIVE),

    # ── Aggregate ───────────────────────────────────────────────────────────
    (re.compile(
        r"\b(how many|count(ing)?|total number|sum of|average|"
        r"mean|median|percentage|proportion|frequency of)\b",
        re.IGNORECASE,
    ), QDMROp.AGGREGATE),

    # ── Superlative ─────────────────────────────────────────────────────────
    (re.compile(
        r"\b(most|least|best|worst|highest|lowest|top \d+|"
        r"bottom \d+|largest|smallest|maximum|minimum|"
        r"primary|leading|predominant)\b",
        re.IGNORECASE,
    ), QDMROp.SUPERLATIVE),

    # ── Arithmetic / scoring ────────────────────────────────────────────────
    (re.compile(
        r"\b(calculate|compute|formula|equation|log|ratio|"
        r"score|e.?value|p.?value|fdr|bitscore|identity|"
        r"coverage|gc content|entropy|information content)\b",
        re.IGNORECASE,
    ), QDMROp.ARITHMETIC),

    # ── Filter / restriction ────────────────────────────────────────────────
    (re.compile(
        r"\b(only|where|filter|subset|restrict|with condition|"
        r"that (have|has|are|is)|given that|limited to|"
        r"specific(ally)?|excluding|within)\b",
        re.IGNORECASE,
    ), QDMROp.FILTER),

    # ── Project (get property of entity) ────────────────────────────────────
    (re.compile(
        r"\b(what (is|are) the .{0,30} of|property|attribute|"
        r"feature of|structure of|sequence of|function of|"
        r"role of|mechanism of)\b",
        re.IGNORECASE,
    ), QDMROp.PROJECT),

    # ── Union / combination ─────────────────────────────────────────────────
    (re.compile(
        r"\b(combine|merge|union|together with|along with|"
        r"as well as|and also|integrate|consolidate)\b",
        re.IGNORECASE,
    ), QDMROp.UNION),

    # ── Intersection ────────────────────────────────────────────────────────
    (re.compile(
        r"\b(both|overlap|shared|common between|in common|"
        r"intersection|co-expressed|co-occurring)\b",
        re.IGNORECASE,
    ), QDMROp.INTERSECTION),

    # ── Sort ────────────────────────────────────────────────────────────────
    (re.compile(
        r"\b(sort|rank|order|prioritize|arrange|list in order of)\b",
        re.IGNORECASE,
    ), QDMROp.SORT),

    # ── Default: open-ended retrieval ───────────────────────────────────────
    (re.compile(r".*", re.IGNORECASE), QDMROp.SELECT),
]


class ClauseExtractor:
    """
    Deterministic clause splitter based on RST (Rhetorical Structure Theory)
    segmentation heuristics.

    Reference:
        Marcu (1997). The Rhetorical Parsing of Natural Language.

    The extractor detects three types of clause boundaries:
      - Sentence-terminal punctuation followed by whitespace
      - Semicolons (strong boundary)
      - Commas immediately before an action verb (weak boundary)

    Then splits each candidate on discourse connectives that indicate a
    new information need (e.g. "additionally", "furthermore").

    Only clauses longer than MIN_CLAUSE_LEN characters are retained to
    filter out sentence fragments that carry no standalone query intent.
    """

    MIN_CLAUSE_LEN = 12

    # Boundaries that unambiguously start a new clause
    _STRONG_BOUNDARY = re.compile(
        r"(?<=[.?!])\s+|"
        r"\s*;\s*|"
        r"\s*,\s*(?=(?:find|get|write|compare|check|also|and|"
        r"calculate|deploy|implement)\b)",
        re.IGNORECASE,
    )

    # RST nucleus-satellite connectives that signal a new information need
    _DISCOURSE_CONNECTIVE = re.compile(
        r"\b(and also|furthermore|additionally|moreover|in addition|"
        r"as well as|along with|while also|and then|after that|"
        r"following that|subsequently|finally|lastly|"
        r"on top of that|beyond that)\b",
        re.IGNORECASE,
    )

    def split(self, query: str) -> List[str]:
        parts = self._STRONG_BOUNDARY.split(query)
        clauses: List[str] = []
        for part in parts:
            for sub in self._DISCOURSE_CONNECTIVE.split(part):
                cleaned = sub.strip()
                if len(cleaned) >= self.MIN_CLAUSE_LEN:
                    clauses.append(cleaned)
        return clauses if clauses else [query.strip()]


def _classify_clause(clause: str) -> QDMROp:
    """Apply QDMR pattern rules in order; return the first matching operation."""
    for pattern, op in QDMR_PATTERN_RULES:
        if pattern.search(clause):
            return op
    return QDMROp.SELECT  # unreachable — last rule is a catch-all


def _annotate_query(clause: str, op: QDMROp, agent: str) -> str:
    """
    Append agent-specific metadata to the clause text, following AGENT_GUIDELINES:
      - rag:     append --strategy=<strategy>
      - scholar: prepend "Find N experts on:" if no count is present
      - code/cloud: no modification needed
    """
    if agent == "rag":
        strategy = QDMR_OP_TO_STRATEGY.get(op, "tot")
        return f"{clause} --strategy={strategy}"
    if agent == "scholar":
        if not re.search(r"\bfind\s+\d+\b", clause, re.IGNORECASE):
            return f"Find 2 experts on: {clause}"
    return clause


# QDMR operations that typically depend on a prior SELECT/UNION task
# for their referent — maps to "@@REF@@" resolution in Wolfson et al.
_REFERENT_DEPENDENT_OPS: FrozenSet[QDMROp] = frozenset({
    QDMROp.FILTER,
    QDMROp.PROJECT,
    QDMROp.SORT,
    QDMROp.SUPERLATIVE,
    QDMROp.COMPARATIVE,
    QDMROp.INTERSECTION,
    QDMROp.DISCARD,
    QDMROp.AGGREGATE,
})


def _infer_dependencies(nodes: List[TaskNode]) -> List[TaskNode]:
    """
    Heuristic dependency resolution based on QDMR's @@REF@@ convention.

    For each node whose QDMR operation is in _REFERENT_DEPENDENT_OPS,
    we search backwards for the nearest node that itself has no dependencies
    (i.e. a root/SELECT task) and assign it as a prerequisite.

    This mirrors the reference-resolution step in Wolfson et al. where
    dependent steps refer to the output of an earlier step.
    """
    for i, node in enumerate(nodes):
        # Always make CODE and DEPLOY depend on the immediate previous node 
        # so they receive the biological context or the code to deploy!
        if node.qdmr_op in [QDMROp.CODE.value, QDMROp.DEPLOY.value]:
            if i > 0:
                node.depends_on = [nodes[i-1].id]
                continue
                
        # Standard QDMR dependency logic for other nodes
        if node.qdmr_op and QDMROp(node.qdmr_op) in _REFERENT_DEPENDENT_OPS:
            if i == 0:
                continue
            for j in range(i - 1, -1, -1):
                if not nodes[j].depends_on:
                    node.depends_on = [nodes[j].id]
                    break
    return nodes


class QDMRRuleDecomposer(BaseDecomposer):
    """
    Fully deterministic query decomposer based on QDMR.

    Given the same input, this decomposer always produces the same output —
    no LLM calls, no sampling, no randomness.

    Pipeline:
        1. Clause segmentation    — RST-inspired discourse splitting
        2. QDMR op classification — ordered regex pattern matching
        3. Agent routing          — static lookup table
        4. Query annotation       — strategy / count metadata
        5. Dependency inference   — QDMR @@REF@@ heuristic

    References:
        Wolfson et al. (2020). Break It Down: A Question Understanding
            Benchmark. Transactions of the ACL.
        Khot et al. (2021). Text Modular Networks. EMNLP.
        Perez et al. (2020). Unsupervised Question Decomposition for QA. EMNLP.
        Marcu (1997). The Rhetorical Parsing of Natural Language.
    """

    name = "qdmr_rule"

    def __init__(self) -> None:
        self._clause_extractor = ClauseExtractor()

    async def decompose(
        self, query: str, model: Optional[genai.GenerativeModel] = None
    ) -> List[TaskNode]:
        # model is intentionally ignored — this decomposer is LLM-free
        clauses = self._clause_extractor.split(query)

        nodes: List[TaskNode] = []
        for i, clause in enumerate(clauses, start=1):
            op = _classify_clause(clause)
            agent = QDMR_OP_TO_AGENT[op]
            annotated = _annotate_query(clause, op, agent)
            nodes.append(
                TaskNode(
                    id=i,
                    agent=agent,
                    query=annotated,
                    qdmr_op=op.value,
                )
            )

        nodes = _infer_dependencies(nodes)
        return nodes


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — Coverage-Based Decomposer (novel, deterministic)
#
# Framing:
#   A query is mapped to a bit-vector of *information-need features*.
#   Each agent is modelled as a set of features it can satisfy.
#   Decomposition = finding the minimum sub-set of agents whose
#   feature-sets cover all active query features (Set Cover, NP-hard
#   in general, ε-approximable greedily in O(n log n)).
#
# References:
#   Chvatal (1979). A Greedy Heuristic for the Set-Covering Problem.
#       Mathematics of Operations Research, 4(3).
#   Carbonell & Goldstein (1998). The Use of MMR, Diversity-Based Reranking
#       for Reordering Documents and Producing Summaries. SIGIR.
#   Suris et al. (2023). VisProg: Compositional Visual Programming. CVPR.
# ──────────────────────────────────────────────────────────────────────────────


class Feature(str, Enum):
    """
    Atomic information-need features that a bioinformatics query may exhibit.
    """

    # ── Retrieval / knowledge ─────────────────────────────────────────────────
    CONCEPTUAL_KNOWLEDGE  = "conceptual_knowledge"
    FACTUAL_LOOKUP        = "factual_lookup"
    COMPARATIVE_ANALYSIS  = "comparative_analysis"
    MULTI_HOP_REASONING   = "multi_hop_reasoning"
    VERIFICATION          = "verification"

    # ── Computational / code ──────────────────────────────────────────────────
    CODE_GENERATION       = "code_generation"
    CODE_DEBUGGING        = "code_debugging"
    PIPELINE_DESIGN       = "pipeline_design"

    # ── Literature / human expertise ─────────────────────────────────────────
    EXPERT_DISCOVERY      = "expert_discovery"
    PAPER_RETRIEVAL       = "paper_retrieval"
    CITATION_NEEDED       = "citation_needed"

    # ── Infrastructure / deployment ───────────────────────────────────────────
    CLOUD_DEPLOYMENT      = "cloud_deployment"
    INFRASTRUCTURE_DESIGN = "infrastructure_design"

    # ── Bioinformatics domain-specific ────────────────────────────────────────
    SEQUENCE_ANALYSIS     = "sequence_analysis"
    STRUCTURAL_BIOLOGY    = "structural_biology"
    PATHWAY_ANALYSIS      = "pathway_analysis"
    STATISTICS_BIOINFO    = "statistics_bioinfo"
    DATA_FORMAT           = "data_format"


_FEATURE_EXTRACTION_RULES: List[Tuple[re.Pattern, Set[Feature]]] = [
    (re.compile(r"\b(write|script|code|implement|function|class|debug|fix)\b", re.I),
     {Feature.CODE_GENERATION}),
    (re.compile(r"\b(debug|error|traceback|exception|bug|fix the code)\b", re.I),
     {Feature.CODE_DEBUGGING}),
    (re.compile(r"\b(pipeline|workflow|snakemake|nextflow|cwl|wdl)\b", re.I),
     {Feature.PIPELINE_DESIGN, Feature.CODE_GENERATION}),
    (re.compile(r"\b(aws|deploy|docker|kubernetes|cloud|s3|ec2|lambda)\b", re.I),
     {Feature.CLOUD_DEPLOYMENT}),
    (re.compile(r"\b(architect|infra|infrastructure|terraform|helm)\b", re.I),
     {Feature.INFRASTRUCTURE_DESIGN, Feature.CLOUD_DEPLOYMENT}),
    (re.compile(r"\b(expert|researcher|author|who works? on)\b", re.I),
     {Feature.EXPERT_DISCOVERY}),
    (re.compile(r"\b(paper|publication|journal|article|citation|reference)\b", re.I),
     {Feature.PAPER_RETRIEVAL, Feature.CITATION_NEEDED}),
    (re.compile(r"\b(literature review|state of the art|recent advances?)\b", re.I),
     {Feature.PAPER_RETRIEVAL, Feature.EXPERT_DISCOVERY}),
    (re.compile(r"\b(compare|versus|vs\.?|difference|trade.?off|pros and cons)\b", re.I),
     {Feature.COMPARATIVE_ANALYSIS}),
    (re.compile(r"\b(then|after|because|so that|in order to|therefore)\b", re.I),
     {Feature.MULTI_HOP_REASONING}),
    (re.compile(r"\b(is it true|verify|confirm|check if|does .{0,40} work)\b", re.I),
     {Feature.VERIFICATION}),
    (re.compile(
        r"\b(sequence|alignment|blast|fasta|fastq|genome|exome|"
        r"transcript|rna.?seq|dna|protein|amino acid|codon|"
        r"assembly|annotation|variant|snp|indel|vcf)\b", re.I),
     {Feature.SEQUENCE_ANALYSIS, Feature.CONCEPTUAL_KNOWLEDGE}),
    (re.compile(
        r"\b(structure|pdb|alphafold|rosetta|homology model|"
        r"cryo.?em|x.?ray|crystallography|fold|domain)\b", re.I),
     {Feature.STRUCTURAL_BIOLOGY, Feature.CONCEPTUAL_KNOWLEDGE}),
    (re.compile(
        r"\b(pathway|kegg|reactome|go term|enrichment|gsea|"
        r"network|interaction|signaling|metabol)\b", re.I),
     {Feature.PATHWAY_ANALYSIS, Feature.CONCEPTUAL_KNOWLEDGE}),
    (re.compile(
        r"\b(p.?value|fdr|multiple testing|statistical|"
        r"differential expression|fold.?change|normali[sz])\b", re.I),
     {Feature.STATISTICS_BIOINFO, Feature.FACTUAL_LOOKUP}),
    (re.compile(
        r"\b(bam|sam|vcf|gff|gtf|bed|fasta|fastq|hdf5|zarr|"
        r"parse|convert|format|read.?in)\b", re.I),
     {Feature.DATA_FORMAT, Feature.CODE_GENERATION}),
    (re.compile(r"\b(what|how|why|explain|define|describe|overview|concept)\b", re.I),
     {Feature.CONCEPTUAL_KNOWLEDGE}),
    (re.compile(r"\b(which|list|enumerate|give me|show me)\b", re.I),
     {Feature.FACTUAL_LOOKUP}),
]

AGENT_CAPABILITY_MATRIX: Dict[str, Set[Feature]] = {
    "rag": {
        Feature.CONCEPTUAL_KNOWLEDGE, Feature.FACTUAL_LOOKUP,
        Feature.COMPARATIVE_ANALYSIS, Feature.MULTI_HOP_REASONING,
        Feature.VERIFICATION, Feature.SEQUENCE_ANALYSIS,
        Feature.STRUCTURAL_BIOLOGY, Feature.PATHWAY_ANALYSIS,
        Feature.STATISTICS_BIOINFO, Feature.DATA_FORMAT,
    },
    "code": {
        Feature.CODE_GENERATION, Feature.CODE_DEBUGGING,
        Feature.PIPELINE_DESIGN, Feature.DATA_FORMAT,
        Feature.SEQUENCE_ANALYSIS, Feature.STATISTICS_BIOINFO,
    },
    "scholar": {
        Feature.EXPERT_DISCOVERY, Feature.PAPER_RETRIEVAL,
        Feature.CITATION_NEEDED, Feature.COMPARATIVE_ANALYSIS,
    },
    "cloud": {
        Feature.CLOUD_DEPLOYMENT, Feature.INFRASTRUCTURE_DESIGN,
        Feature.PIPELINE_DESIGN,
    },
}


def _extract_features(query: str) -> Set[Feature]:
    """Deterministically extract the set of information-need features."""
    active: Set[Feature] = set()
    for pattern, feats in _FEATURE_EXTRACTION_RULES:
        if pattern.search(query):
            active |= feats
    if not active:
        active.add(Feature.CONCEPTUAL_KNOWLEDGE)
    return active


def _greedy_set_cover(
    needs: Set[Feature],
    capabilities: Dict[str, Set[Feature]],
) -> List[Tuple[str, Set[Feature]]]:
    """
    Greedy approximation for the Minimum Set Cover problem.

    Reference:
        Chvatal (1979). A Greedy Heuristic for the Set-Covering Problem.
        Mathematics of Operations Research, 4(3), 233-235.
    """
    uncovered = set(needs)
    selected: List[Tuple[str, Set[Feature]]] = []
    agent_priority = {"rag": 0, "code": 1, "cloud": 2, "scholar": 3}

    while uncovered:
        best_agent: Optional[str] = None
        best_coverage: Set[Feature] = set()

        for agent, agent_caps in capabilities.items():
            covered = uncovered & agent_caps
            if len(covered) > len(best_coverage) or (
                len(covered) == len(best_coverage)
                and best_agent is not None
                and agent_priority.get(agent, 99) < agent_priority.get(best_agent, 99)
            ):
                best_agent = agent
                best_coverage = covered

        if not best_agent or not best_coverage:
            selected.append(("rag", uncovered.copy()))
            break

        selected.append((best_agent, best_coverage))
        uncovered -= best_coverage

    return selected


def _build_coverage_query(
    original_query: str,
    agent: str,
    features: Set[Feature],
) -> str:
    """Construct a focused sub-query for the agent, scoped to its feature set."""
    feature_hints: Dict[Feature, str] = {
        Feature.COMPARATIVE_ANALYSIS:  "Compare",
        Feature.VERIFICATION:          "Verify",
        Feature.EXPERT_DISCOVERY:      "Find 2 experts on",
        Feature.PAPER_RETRIEVAL:       "Find papers on",
        Feature.CODE_GENERATION:       "Write a Python script to",
        Feature.CODE_DEBUGGING:        "Debug the following code for",
        Feature.PIPELINE_DESIGN:       "Design a pipeline to",
        Feature.CLOUD_DEPLOYMENT:      "Deploy to cloud:",
        Feature.INFRASTRUCTURE_DESIGN: "Design infrastructure for",
    }
    hint = next((h for f, h in feature_hints.items() if f in features), None)
    base = f"{hint}: {original_query}" if hint else original_query

    if agent == "rag":
        if Feature.VERIFICATION in features:
            strategy = "react"
        elif Feature.COMPARATIVE_ANALYSIS in features:
            strategy = "got"
        elif len(features) == 1:
            strategy = "cot"
        else:
            strategy = "tot"
        return f"{base} --strategy={strategy}"

    if agent == "scholar":
        if not re.search(r"\bfind\s+\d+\b", base, re.IGNORECASE):
            base = f"Find 2 experts on: {original_query}"
    return base


class CoverageDecomposer(BaseDecomposer):
    """
    Coverage-Based Decomposer: frames query decomposition as Minimum Set Cover
    over an explicit information-need feature space.

    Fully deterministic — no LLM calls. Guarantees:
      • Completeness: every extracted need is assigned to at least one agent.
      • Minimality: greedy approximation minimises total number of tasks.
      • No redundancy: the same feature set is never covered twice.

    References:
        Chvatal (1979). Math OR.
        Carbonell & Goldstein (1998). SIGIR.
        Suris et al. (2023). VisProg. CVPR.
        Feige (1998). A Threshold of ln n for Approximating Set Cover. JACM.
    """

    name = "coverage"

    async def decompose(
        self, query: str, model: Optional[genai.GenerativeModel] = None
    ) -> List[TaskNode]:
        needs = _extract_features(query)
        cover_plan = _greedy_set_cover(needs, AGENT_CAPABILITY_MATRIX)

        agent_to_id: Dict[str, int] = {}
        nodes: List[TaskNode] = []

        for task_id, (agent, covered_features) in enumerate(cover_plan, start=1):
            sub_query = _build_coverage_query(query, agent, covered_features)
            depends: List[int] = []
            if agent in ("code", "cloud") and "rag" in agent_to_id:
                depends = [agent_to_id["rag"]]

            nodes.append(
                TaskNode(
                    id=task_id,
                    agent=agent,
                    query=sub_query,
                    depends_on=depends,
                    features_covered=frozenset(f.value for f in covered_features),
                )
            )
            agent_to_id.setdefault(agent, task_id)

        return nodes


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 4 — Neural+Rule Hybrid Decomposer
#
# Framing:
#   Phase 1 (neural):       LLM resolves coreferences and constructs an
#                           explicit, dependency-annotated goal DAG.
#                           Each goal is constrained to a SINGLE intent so
#                           Phase 2 classification is unambiguous.
#   Phase 2 (deterministic): QDMR regex rules route each LLM goal to an
#                           agent and inject strategy/count annotations
#                           exactly as in QDMRRuleDecomposer.
#
# Why not "neuro-symbolic"?
#   True neurosymbolic AI (Garcez & Lamb 2023) integrates neural computation
#   with formal symbolic logic (e.g. first-order logic, constraint solvers,
#   probabilistic programs). This decomposer uses deterministic regex rules —
#   a rule-based system, not a formal symbolic reasoner. The hybrid label
#   is therefore more accurate.
#
# References:
#   Garcez & Lamb (2023). Neurosymbolic AI: The 3rd Wave. AI Review.
#   Wolfson et al. (2020). Break It Down. TACL.
# ──────────────────────────────────────────────────────────────────────────────

# LLM timeout in seconds for the neural phase
_NEURAL_PHASE_TIMEOUT_S: float = 15.0


class NeuroRuleDecomposer(BaseDecomposer):
    """
    Neural+Rule Hybrid Decomposer.

    Phase 1 (neural): An LLM resolves pronouns, disambiguates referents, and
    produces a dependency-annotated goal DAG. The prompt enforces one intent
    per goal so that Phase 2 classification is unambiguous.

    Phase 2 (deterministic): Each LLM-produced goal is classified via the
    QDMR regex rules (_classify_clause) and annotated via _annotate_query —
    identical to QDMRRuleDecomposer's pipeline. No regex is applied to the
    raw user query; the LLM's clarified goal text is used instead.

    On LLM failure or timeout, falls back gracefully to QDMRRuleDecomposer.

    References:
        Garcez & Lamb (2023). Neurosymbolic AI: The 3rd Wave. AI Review.
        Wolfson et al. (2020). Break It Down. TACL.
    """

    name = "neuro_rule"

    # FIX (Issue 7): Prompt explicitly enforces one intent per goal, preventing
    # multi-intent goals that would confuse the single-op _classify_clause.
    _PHASE1_PROMPT = """
You are a Logical Query Architect. Break the user's query into a DAG of sub-goals.

STRICT RULES:
1. Resolve ALL pronouns and demonstratives — replace "this topic", "it", "them"
   with the actual noun from the query.
2. Each goal must have EXACTLY ONE intent (retrieve, code, find experts, deploy).
   Never combine two intents in one goal. Split them into separate goals.
3. If goal N must use the result of goal M, add M to "depends_on" for goal N
   and write a placeholder like "[result of goal M]" in the goal text.
4. DO NOT assign agents — just write the goal text clearly.

Return ONLY valid JSON:
{{
  "goals": [
    {{"id": 1, "text": "...", "depends_on": []}},
    {{"id": 2, "text": "...", "depends_on": [1]}}
  ]
}}

User Query: {query}
""".strip()

    async def decompose(
        self, query: str, model: Optional[genai.GenerativeModel] = None
    ) -> List[TaskNode]:
        if model is None:
            logger.warning(
                "[NeuroRuleDecomposer] No model provided; falling back to QDMRRuleDecomposer."
            )
            return await QDMRRuleDecomposer().decompose(query)

        # ── Phase 1: Neural DAG extraction ───────────────────────────────────
        logger.debug("[NeuroRuleDecomposer] Phase 1: neural goal DAG extraction.")
        try:
            prompt = self._PHASE1_PROMPT.format(query=query)
            # FIX (Issue 8): timeout guard — avoids indefinite block on API hang
            res = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=_NEURAL_PHASE_TIMEOUT_S,
            )
            text = res.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            goals = data.get("goals", [])
            if not goals:
                raise ValueError("LLM returned empty goal list.")
        except asyncio.TimeoutError:
            logger.warning(
                "[NeuroRuleDecomposer] Phase 1 timed out after %.1fs; "
                "falling back to QDMRRuleDecomposer.",
                _NEURAL_PHASE_TIMEOUT_S,
            )
            return await QDMRRuleDecomposer().decompose(query)
        except Exception as exc:
            logger.warning(
                "[NeuroRuleDecomposer] Phase 1 failed (%s); "
                "falling back to QDMRRuleDecomposer.",
                exc,
            )
            return await QDMRRuleDecomposer().decompose(query)

        # ── Phase 2: Deterministic agent routing ──────────────────────────────
        # FIX (Issue 7): Because the prompt enforces one intent per goal, each
        # goal text is unambiguous for _classify_clause's first-match logic.
        logger.debug("[NeuroRuleDecomposer] Phase 2: deterministic QDMR routing.")
        nodes: List[TaskNode] = []
        for goal in goals:
            goal_text: str = goal.get("text", "")
            if not goal_text.strip():
                continue
            op = _classify_clause(goal_text)
            agent = QDMR_OP_TO_AGENT[op]
            annotated = _annotate_query(goal_text, op, agent)
            nodes.append(
                TaskNode(
                    id=goal["id"],
                    agent=agent,
                    query=annotated,
                    depends_on=goal.get("depends_on", []),
                    qdmr_op=op.value,
                )
            )

        return nodes


# ──────────────────────────────────────────────────────────────────────────────
# REGISTRY + AUTO-SELECTOR
# (single definition — duplicate removed from previous version)
# ──────────────────────────────────────────────────────────────────────────────


class DecomposerRegistry:
    """Central registry for all decomposer strategies."""

    def __init__(self) -> None:
        self._registry: Dict[str, BaseDecomposer] = {
            d.name: d
            for d in [
                PlanAndSolveDecomposer(),
                LLMCompilerDecomposer(),
                QDMRRuleDecomposer(),
                CoverageDecomposer(),
                NeuroRuleDecomposer(),
            ]
        }

    def get(self, name: str) -> BaseDecomposer:
        # FIX (Issue 2): default falls back to plan_and_solve, not the
        # model-dependent neuro_rule strategy.
        return self._registry.get(name, self._registry["plan_and_solve"])

    def list_strategies(self) -> List[str]:
        return list(self._registry.keys())


class AutoDecomposerSelector:
    """
    Heuristic selector that picks the most appropriate decomposer based on
    observable query properties — without consulting an LLM.

    Selection logic
    ───────────────
    deterministic=True (zero latency, no model required):
        • ≥2 distinct features detected  → coverage
        • Single feature                 → qdmr_rule

    deterministic=False (LLM available):
        • Explicit dependency signals    → neuro_rule   (resolves coreference)
        • Otherwise                      → llm_compiler (cheaper, no coref needed)
        • Fallback (no model)            → plan_and_solve

    explicit (always wins if provided):
        • Any name from DecomposerRegistry → that strategy exactly
    """

    # FIX (Issue 3): Both llm_compiler AND neuro_rule are now reachable.
    # llm_compiler handles structurally complex but linguistically clear queries.
    # neuro_rule handles queries with pronoun chains or ambiguous referents.
    _DEPENDENCY_SIGNALS = re.compile(
        r"\b(then|after|because of|based on|use the result|"
        r"using that|given the output|in light of|"
        r"this topic|the above|the same|it then|them to)\b",
        re.IGNORECASE,
    )

    def __init__(self, registry: DecomposerRegistry) -> None:
        self._registry = registry

    def select(
        self,
        query: str,
        *,
        deterministic: bool = True,
        explicit: Optional[str] = None,
    ) -> BaseDecomposer:
        if explicit:
            return self._registry.get(explicit)

        if deterministic:
            features = _extract_features(query)
            return self._registry.get(
                "coverage" if len(features) >= 2 else "qdmr_rule"
            )

        # LLM-available path
        if self._DEPENDENCY_SIGNALS.search(query):
            # Pronoun/coreference chains → neuro_rule for clarification first
            return self._registry.get("neuro_rule")
        # Structurally clear multi-step query → compiler-style dependency graph
        return self._registry.get("llm_compiler")


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ──────────────────────────────────────────────────────────────────────────────

async def demo() -> None:
    logging.basicConfig(level=logging.DEBUG)
    registry = DecomposerRegistry()
    selector = AutoDecomposerSelector(registry)

    queries = [
        # Single-intent → qdmr_rule
        "What is the BLAST algorithm and how does it work?",

        # Multi-intent → coverage
        (
            "Explain the difference between BWA and Bowtie2 for short-read alignment, "
            "write a Python script to run both on a FASTQ file, "
            "and find 2 experts who have published on this topic."
        ),

        # Code-only → qdmr_rule routes to code agent
        "Write a Biopython script to parse a GenBank file and extract CDS features.",

        # Mixed + deployment → coverage fires, rag→code→cloud deps generated
        (
            "Describe the GATK variant-calling pipeline, implement it in Snakemake, "
            "and deploy it on AWS Batch."
        ),
    ]

    print("=" * 72)
    print("Multi-Strategy Query Decomposition Demo")
    print("Strategies:", registry.list_strategies())
    print("=" * 72)

    for query in queries:
        print(f"\nQuery: {query[:80]}{'...' if len(query) > 80 else ''}")
        decomposer = selector.select(query, deterministic=True)
        print(f"Selected decomposer: {decomposer.name}")
        tasks = await decomposer.decompose(query)
        for t in tasks:
            deps = f" (depends on: {t.depends_on})" if t.depends_on else ""
            op   = f" [{t.qdmr_op}]" if t.qdmr_op else ""
            cov  = f" covers={set(t.features_covered)}" if t.features_covered else ""
            print(f"  Task {t.id} [{t.agent}]{op}{cov}{deps}")
            print(f"       -> {t.query[:90]}{'...' if len(t.query) > 90 else ''}")
        print("-" * 72)


if __name__ == "__main__":
    asyncio.run(demo())