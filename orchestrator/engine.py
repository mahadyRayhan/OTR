import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
# import os
from datetime import datetime
import re
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Dict, Any, List
import time

import config
from .decomposers import DecomposerRegistry, AutoDecomposerSelector
from .dag_builders import DAGRegistry
from .evaluation.communication_log import comm_tracker
# from .evaluation.communication_evaluator import CommunicationEvaluator

# Import the GoT Classes
from .adaptive_got_dag import AdaptiveGoTOrchestrator, GoTNode, DepthBudget, GoTOp

# 🚀 CORRECTED IMPORTS: Pointing to the specific agent folders
from rag_agent.knowledge_base import KnowledgeBase
from rag_agent.agent import call_rag_agent
from scholar_agent.agent import call_scholar_agent
from cloud_agent.agent import call_cloud_agent
from code_agent.agent import call_code_agent
from profiler_agent.agent import user_profiler, classify_query, dominant_dim
from orchestrator.decomposers import NeuroRuleDecomposer

# =========================================================
# 🚀 CUGCR ATOMIZER (Deterministic Context Parsing)
# =========================================================
from dataclasses import dataclass
from enum import Enum

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

# ── Entity stopwords for atomizer ─────────────────────────────────────────────
# NLTK English stopwords (title-cased) + domain-generic academic/bio terms that
# match the [A-Z][a-z]{4,} pattern but are not named entities.
def _build_entity_stopwords() -> frozenset:
    base: set[str] = set()
    try:
        import nltk
        from nltk.corpus import stopwords as _sw
        base = {w.title() for w in _sw.words("english")}
    except Exception:
        pass
    # Generic academic/bioinformatics prose words NLTK doesn't cover.
    # Expanded after observing false positives in telemetry (Local, Source, Solution…).
    extra = {
        # discourse / hedge words
        "Crucially", "Importantly", "Notably", "Particularly", "Essentially",
        "Specifically", "Generally", "Typically", "Commonly", "Frequently",
        "Paradoxically", "Conversely", "Similarly", "Consequently",
        # generic academic nouns
        "Refinement", "Improvement", "Detection", "Analysis", "Method",
        "Algorithm", "Approach", "Framework", "Process", "System",
        "Result", "Output", "Input", "Data", "Value", "Answer", "Paper",
        "Study", "Figure", "Table", "Section", "Example", "Step",
        "Performance", "Comparison", "Discussion", "Introduction",
        "Conclusion", "Abstract", "Source", "Solution", "Problem",
        "Challenge", "Phase", "Stage", "Version", "Component", "Module",
        "Structure", "Parameter", "Property", "Feature", "Factor", "Element",
        "Mechanism", "Strategy", "Technique", "Procedure", "Operation",
        # graph / math nouns (not tool names)
        "Communities", "Community", "Network", "Nodes", "Edges",
        "Graph", "Matrix", "Vector", "Score", "Partition", "Cluster",
        "Subgraph", "Connectivity", "Modularity", "Resolution",
        # generic adjectives that open sentences
        "Local", "Global", "Total", "Single", "Entire", "Original",
        "Novel", "Previous", "Current", "Recent", "Basic", "Advanced",
        "Standard", "Simple", "Complex", "Initial", "Final", "Primary",
        "Secondary", "Internal", "External", "Optimal", "Efficient",
        # verb-derived nouns / gerunds that start sentences
        "Moving", "Building", "Ensuring", "Computing", "Running",
        "Loading", "Saving", "Performing", "Generating", "Creating",
        "Based", "Using", "Given", "Through", "Across", "Along",
    }
    return frozenset(base | extra)

_ENTITY_STOPWORDS: frozenset = _build_entity_stopwords()

class AtomType(Enum):
    CODE_BLOCK = "code"
    NAMED_ENTITY = "entity"
    PROSE_CLAIM = "prose"

@dataclass
class ContextAtom:
    type: AtomType
    content: str
    is_protected: bool

# =========================================================
# 🚀 CUGCR DEPENDENCY MANIFESTS (Tightened for Benchmark)
# =========================================================
@dataclass
class DependencyManifest:
    agent_type: str
    required_atom_types: list[AtomType]
    token_budget: int

AGENT_MANIFESTS = {
    "code": DependencyManifest(
        agent_type="code",
        # PROSE_CLAIM added: RAG often returns pure-prose algorithm explanations.
        # Without PROSE_CLAIM, CUGCR discards the entire context and CODE gets
        # only a handful of acronym tokens (e.g. "PPI"), making it write blind.
        required_atom_types=[AtomType.CODE_BLOCK, AtomType.NAMED_ENTITY, AtomType.PROSE_CLAIM],
        # Calibrated from 343 runs: mean raw=839 tok; sweet spot bucket 30–60% of raw
        # yields J=6.80 vs J_raw=5.40. Budget 377 tok → 45% of mean raw.
        # Pearson r(ratio,quality)=-0.277: more compression → higher quality for code.
        token_budget=377
    ),
    "cloud": DependencyManifest(
        agent_type="cloud",
        # Same reasoning: cloud tasks need upstream code context (prose description
        # of what to deploy) not just raw acronyms.
        required_atom_types=[AtomType.CODE_BLOCK, AtomType.NAMED_ENTITY, AtomType.PROSE_CLAIM],
        # Calibrated from 343 runs: mean raw=806 tok; moderate bucket (30–60%)
        # yields J=7.33 with 100% first-pass success and 0 retries. Budget 362 tok → 45%.
        # Heavy compression (0–30%, ~68 tok) collapses quality to J=3.0.
        token_budget=362
    ),
    "rag": DependencyManifest(
        agent_type="rag",
        required_atom_types=[AtomType.NAMED_ENTITY, AtomType.PROSE_CLAIM],
        # Calibrated: best obs at ratio=0.983 (near-full context). RAG as retry
        # recipient benefits from full upstream context; budget set to 95% of mean raw.
        token_budget=914
    ),
    "scholar": DependencyManifest(
        agent_type="scholar",
        required_atom_types=[AtomType.NAMED_ENTITY, AtomType.PROSE_CLAIM],
        # Scholar incoming context is ~275 tok (self-contained API results).
        # Budget kept high so CUGCR never triggers — scholar does not benefit from
        # inter-agent compression. Effective budget via fusion multiplier: 1500*2=3000.
        token_budget=1500
    )
}

# FIX 2: Agent-specific fusion weights for _score_and_select_atoms.
# Scholar needs semantic (dense) understanding, not keyword (sparse) matching.
# Code/Cloud need exact token matching for specs, so sparse stays high.
AGENT_FUSION_WEIGHTS = {
    "code":    {"dense": 0.50, "sparse": 0.50, "budget_multiplier": 1.0},
    "cloud":   {"dense": 0.40, "sparse": 0.60, "budget_multiplier": 1.0},
    "rag":     {"dense": 0.60, "sparse": 0.40, "budget_multiplier": 1.0},
    "scholar": {"dense": 0.85, "sparse": 0.15, "budget_multiplier": 2.0},
    # budget_multiplier=2.0 gives scholar an effective budget of 600
    # so prose passages describing researchers are not discarded.
}

# ──────────────────────────────────────────────────────────────────────────────
# PROFILER INTEGRATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# Per-dimension depth instructions injected into every agent call.
_PROFILE_DEPTH_HINTS: dict[str, dict[str, str]] = {
    "procedural": {
        "rag":     "Focus on practical usage, tool parameters, and step-by-step procedures. Prioritise how-to over theory.",
        "code":    "Provide a complete, executable implementation. Include all imports, parameters, and error handling. Do not truncate any code blocks.",
        "cloud":   "Provide a complete infrastructure definition with all resources, variables, outputs, and IAM roles explicitly specified.",
        "scholar": "Find researchers who actively publish implementations, tools, or pipelines — not just theoretical work.",
    },
    "analytical": {
        "rag":     "Provide comparative analysis: discuss trade-offs, limitations, and alternatives. Cite specific benchmarks or papers where relevant.",
        "code":    "Add inline comments explaining design decisions. Include a short performance or complexity note where appropriate.",
        "cloud":   "Compare at least two architectural approaches with cost, scalability, and fault-tolerance trade-offs before recommending one.",
        "scholar": "Find researchers with opposing or complementary views on the topic to give a balanced expert landscape.",
    },
    "conceptual": {
        "rag":     "Prioritise mechanistic explanation: cover the underlying algorithm, mathematical basis, and intuition before any implementation detail.",
        "code":    "Add a docstring explaining the algorithm being implemented before the code. Keep implementation straightforward and well-commented.",
        "cloud":   "Explain WHY each infrastructure component is chosen before specifying it. Cover the architectural rationale.",
        "scholar": "Find researchers whose primary contribution is theoretical or foundational to the topic.",
    },
}

def _build_profile_hint(dim: str, scores: dict) -> str:
    """Build the one-line profile hint injected into the decomposer prompt."""
    pct = scores.get(dim, 33.3)
    hints = {
        "procedural": f"PROCEDURAL ({pct:.0f}%) — user wants a complete, runnable implementation. Prioritise code/deploy subtasks; keep theory brief.",
        "analytical": f"ANALYTICAL ({pct:.0f}%) — user wants comparison and evaluation. Add trade-off or evaluation subtasks where the query allows.",
        "conceptual": f"CONCEPTUAL ({pct:.0f}%) — user wants deep understanding. Ensure at least one explanatory subtask covers the mechanism thoroughly.",
    }
    return hints.get(dim, "No dominant dimension — balance all subtasks equally.")

# ──────────────────────────────────────────────────────────────────────────────
# CUGCR COMPRESSION-GOAL ROUTING
# ──────────────────────────────────────────────────────────────────────────────

# When upstream context comes from a different agent type, generic cosine
# similarity to the downstream query misses what the downstream agent needs.
# These goals are used ONLY for atom-relevance scoring, not for the actual query.
_COMPRESSION_GOALS: dict[tuple[str, str], str] = {
    ("code",    "cloud"):   "infrastructure requirements: compute resources, instance types, storage capacity, data volumes, and deployment specifications",
    ("rag",     "code"):    "computational methods, algorithms, bioinformatics tools, analysis workflows, parameters, and technical procedures",
    ("rag",     "cloud"):   "deployment requirements, resource specifications, and infrastructure needs",
    ("scholar", "code"):    "technical methods, algorithms, and tools used by the researchers",
    ("scholar", "cloud"):   "infrastructure and compute requirements described by the researchers",
    ("cloud",   "code"):    "resource configuration details, service names, and parameter values to implement programmatically",
}

def _detect_upstream_agent(result_text: str) -> str:
    """Infer upstream agent type from the result string prefix."""
    t = result_text[:120]
    if "[Provenance-Aware Code Agent]" in t or "[Code Agent" in t:
        return "code"
    if "[Scholar Agent" in t:
        return "scholar"
    if "[Provenance-Aware Cloud Agent]" in t or "[Cloud Agent" in t:
        return "cloud"
    return "rag"

def _compression_goal_for(previous_results: dict, downstream_agent: str, fallback_query: str) -> str:
    """Return the best scoring query for CUGCR atom selection."""
    upstream_types = {_detect_upstream_agent(v) for v in previous_results.values() if isinstance(v, str)}
    for up in upstream_types:
        goal = _COMPRESSION_GOALS.get((up, downstream_agent))
        if goal:
            return goal
    return fallback_query

_PLACEHOLDER_RE = re.compile(
    r'\[(?:result|explanation|output|summary)\s+of\s+(?:goal|task)\s+\d+\]'
    r'|'
    r'\[PLACEHOLDER[^\]]*\]'
    r'|'
    r'@@REF@@',
    re.IGNORECASE,
)

def atomize_output(raw_output: str) -> list[ContextAtom]:
    """Extracts protected code, biological entities, and prose deterministically."""
    atoms = []

    for match in re.finditer(r'```[\s\S]*?```', raw_output):
        atoms.append(ContextAtom(AtomType.CODE_BLOCK, match.group(), is_protected=True))

    prose_only = re.sub(r'```[\s\S]*?```', '', raw_output)
    # Strip markdown headers — words in "## Building Well-Connected Communities"
    # are structural section titles, not named entities. Without this, every
    # capitalized header word (Building, Source, Refined) becomes a false-positive entity.
    prose_only = re.sub(r'^#{1,6}\s+.*$', '', prose_only, flags=re.MULTILINE)
    # Strip bold/italic markers but keep the text (bold terms may be real entities)
    prose_only = re.sub(r'\*{1,3}', '', prose_only)

    bio_pattern = (
        r'\b('
        r'[A-Z]{2,}[0-9]*'              # all-caps acronyms: RNA, STAR, CRISPR
        r'|[A-Z][a-z]+[A-Z]\w*'         # CamelCase: Scanpy, AnnData, SAMtools
        r'|[A-Z]{2,}[a-z]+\w*'          # mixed start: DESeq2, HISAT2, BWA
        r'|[A-Z][a-z]+-\d+'             # hyphen-digit: Leiden-1
        r'|[A-Z][a-z]{4,}'              # long proper nouns ≥6 chars: Leiden, Louvain, Kallisto
        r')\b'
    )
    seen_entities: set[str] = set()
    for match in re.finditer(bio_pattern, prose_only):
        ent = match.group()
        if ent not in seen_entities and ent.title() not in _ENTITY_STOPWORDS:
            seen_entities.add(ent)
            atoms.append(ContextAtom(AtomType.NAMED_ENTITY, ent, is_protected=True))

    for sentence in prose_only.split('. '):
        if len(sentence.strip()) > 15:
            atoms.append(ContextAtom(AtomType.PROSE_CLAIM, sentence.strip() + ".", is_protected=False))

    return atoms

def build_rigid_prompt(original_query: str, raw_context: str, token_budget: int = 400, downstream_agent: str = "rag") -> str:
    """Filters atoms and injects them into a rigid, drift-proof template, filling up to token_budget."""
    atoms = atomize_output(raw_context)
    manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
    valid_atoms = [a for a in atoms if a.type in manifest.required_atom_types]

    code_blocks = [a.content for a in valid_atoms if a.type == AtomType.CODE_BLOCK]
    entities    = list(set([a.content for a in valid_atoms if a.type == AtomType.NAMED_ENTITY]))
    prose       = [a.content for a in valid_atoms if a.type == AtomType.PROSE_CLAIM]

    used_tokens   = 0
    context_block = ""

    # Code blocks are protected — always include them first regardless of budget.
    if code_blocks:
        block_text   = "### Upstream Code Artifacts\n" + "\n".join(code_blocks) + "\n\n"
        used_tokens += len(block_text) // 4
        context_block += block_text

    if entities:
        ent_selected = []
        for ent in entities:
            cost = (len(ent) + 2) // 4  # +2 accounts for ", " separator
            if used_tokens + cost <= token_budget:
                ent_selected.append(ent)
                used_tokens += cost
        if ent_selected:
            context_block += "### Key Technical Entities\n" + ", ".join(ent_selected) + "\n\n"

    if prose:
        prose_selected = []
        for claim in prose:
            cost = len(claim) // 4
            if used_tokens + cost <= token_budget:
                prose_selected.append(claim)
                used_tokens += cost
        if prose_selected:
            context_block += "### Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose_selected) + "\n\n"

    print(f"      [build_rigid_prompt] agent={downstream_agent.upper()} budget={token_budget} tok | filled={used_tokens} tok")
    return f"""{context_block}---
## Your Task
{original_query}
"""

class OrchestratorEngine:
    def __init__(self):
        self.registry = DecomposerRegistry()
        self.selector = AutoDecomposerSelector(self.registry)
        self.dag_builder = DAGRegistry().get(config.Config.DAG_BUILDER)
        self.current_run_cugcr_triggered = False
        self._last_preliminary_crs = 1.0
        self._last_cugcr_triggered = False   # per-call flag reset in _resolve_dependencies
        self._node_metadata: dict = {}       # node_id → {paper_count, cugcr_triggered, crs, ...}
        self._current_query_dim: str = "procedural"  # updated at start of each pipeline run
        
        # 🚀 LOAD KNOWLEDGE BASE (FAISS, Embedder, Cross-Encoder)
        print("⏳ Loading KnowledgeBase for RAG Agent...")
        self.kb = KnowledgeBase()
        self.kb.load()

        # 🚀 LOAD CUGCR EMBEDDING ROUTER
        self.embedder = None
        if getattr(config.Config, "CONTEXT_ROUTING_MODE", "light") == "heavy":
            print("⏳ Loading local semantic embedding model for Heavy Context Routing...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "your_google_api_key_here":
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(config.Config.GENERATION_MODEL)
        else:
            self.llm = None
            
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "sk-your-openai-key-here":
            self.openai_client = AsyncOpenAI(api_key=openai_key)
        else:
            self.openai_client = None

    # def _score_and_select_atoms(self, atoms: list, downstream_query: str, downstream_agent: str, budget_multiplier: float = 1.0) -> list:
    #     manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
    #     current_budget = int(manifest.token_budget * budget_multiplier)

    #     selected = []
    #     used_tokens = 0
        
    #     for atom in atoms:
    #         if atom.is_protected and atom.type in manifest.required_atom_types:
    #             selected.append(atom)
    #             used_tokens += len(atom.content.split())
        
    #     unprotected = [a for a in atoms if a not in selected]
    #     if not unprotected or not self.embedder:
    #         return selected
            
    #     query_emb = self.embedder.encode([downstream_query])
    #     atom_docs = [a.content for a in unprotected]
    #     atom_embs = self.embedder.encode(atom_docs)
    #     dense_scores = cosine_similarity(query_emb, atom_embs)[0]
        
    #     vectorizer = TfidfVectorizer().fit([downstream_query] + atom_docs)
    #     tfidf_matrix = vectorizer.transform(atom_docs)
    #     query_tfidf = vectorizer.transform([downstream_query])
    #     sparse_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        
    #     final_scores = (0.6 * dense_scores) + (0.4 * sparse_scores)
        
    #     scored_unprotected = sorted(zip(unprotected, final_scores), key=lambda x: x[1], reverse=True)
    #     for atom, score in scored_unprotected:
    #         cost = len(atom.content.split())
    #         if used_tokens + cost <= current_budget:
    #             selected.append(atom)
    #             used_tokens += cost
                
    #     return selected

    def _score_and_select_atoms(self, atoms: list, downstream_query: str, downstream_agent: str, budget_multiplier: float = 1.0) -> list:
        manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
        
        # FIX 2: Look up agent-specific fusion weights
        fw = AGENT_FUSION_WEIGHTS.get(downstream_agent, {"dense": 0.6, "sparse": 0.4, "budget_multiplier": 1.0})
        effective_multiplier = budget_multiplier * fw["budget_multiplier"]
        current_budget = int(manifest.token_budget * effective_multiplier)
        
        print(f"      [CUGCR] Agent={downstream_agent.upper()} | Budget={current_budget} tok | Dense={fw['dense']} Sparse={fw['sparse']}")

        # FIX: Strict upfront type filtering to enforce atomization boundaries
        valid_atoms = [a for a in atoms if a.type in manifest.required_atom_types]

        selected   = []
        used_tokens = 0
        
        for atom in valid_atoms:
            if atom.is_protected:
                selected.append(atom)
                used_tokens += len(atom.content.split())
        
        unprotected = [a for a in valid_atoms if not a.is_protected]
        if not unprotected or not self.embedder:
            return selected
            
        query_emb = self.embedder.encode([downstream_query])
        atom_docs  = [a.content for a in unprotected]
        atom_embs  = self.embedder.encode(atom_docs)
        dense_scores = cosine_similarity(query_emb, atom_embs)[0]
        
        vectorizer   = TfidfVectorizer().fit([downstream_query] + atom_docs)
        tfidf_matrix = vectorizer.transform(atom_docs)
        query_tfidf  = vectorizer.transform([downstream_query])
        sparse_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        
        # FIX 2: Use agent-specific weights instead of hardcoded 0.6/0.4
        final_scores = (fw["dense"] * dense_scores) + (fw["sparse"] * sparse_scores)
        
        scored_unprotected = sorted(zip(unprotected, final_scores), key=lambda x: x[1], reverse=True)
        for atom, score in scored_unprotected:
            # FIX: Introduce an absolute relevance floor to prevent GIGO amplification
            if score < 0.15:
                continue
            cost = len(atom.content.split())
            if used_tokens + cost <= current_budget:
                selected.append(atom)
                used_tokens += cost
                
        return selected

    def _depth_instruction(self, agent_name: str) -> str:
        """Return the profile-driven depth instruction for this agent type."""
        if not config.Config.USE_PROFILER:
            return ""
        hints = _PROFILE_DEPTH_HINTS.get(self._current_query_dim, {})
        return hints.get(agent_name, "")

    async def _execute_agent(self, agent_name: str, query: str, depth: int = 1) -> tuple[str, dict]:
        agent_registry = config.Config.AGENT_REGISTRY_PROMPT
        strategy_match = re.search(r'--strategy=(\w+)', query.lower())
        strategy = strategy_match.group(1) if strategy_match else "cot"
        clean_query = re.sub(r'--strategy=\w+', '', query).strip()
        meta: dict = {"agent": agent_name, "strategy": strategy, "depth": depth}

        # Append profile-driven depth instruction to every non-scholar agent.
        # Scholar is API-driven; instruction text would be ignored.
        depth_instr = self._depth_instruction(agent_name)
        if depth_instr and agent_name != "scholar":
            clean_query = f"{clean_query}\n\n[DEPTH REQUIREMENT]: {depth_instr}"

        try:
            if agent_name == "code":
                result = await call_code_agent(clean_query, strategy, depth, agent_registry)
                return result, meta
            if agent_name == "cloud":
                # Use per-query dim hint for cloud profile string instead of
                # accumulated Dirichlet profile, which blurs across queries.
                if config.Config.USE_PROFILER:
                    _cloud_hints = _PROFILE_DEPTH_HINTS.get(self._current_query_dim, {})
                    profile_str  = _cloud_hints.get("cloud", "")
                else:
                    profile_str = ""
                result = await call_cloud_agent(clean_query, strategy, depth, agent_registry, profile_str)
                return result, meta
            if agent_name == "scholar":
                limit = 1
                text_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
                query_normalized = clean_query.lower()
                for word, digit in text_to_num.items():
                    query_normalized = re.sub(rf'\b{word}\b', digit, query_normalized)
                match = re.search(r'\b([1-5])\b', query_normalized)
                if match: limit = int(match.group(1))
                result = await call_scholar_agent(clean_query, False, "Unknown", limit, strategy, depth, agent_registry)
                return result, meta
            else:
                rag_data = await call_rag_agent(kb=self.kb, user_input=clean_query, strategy=strategy, depth=depth, agent_registry=agent_registry)
                meta["paper_count"] = rag_data.get("paper_count", 0)
                meta["rag_source"] = rag_data.get("source_used", "")
                result = f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data.get('answer', '')}"
                return result, meta
        except Exception as e:
            return f"Error executing agent: {e}", meta

    async def _resolve_dependencies(self, query: str, depends_on: List[Any], previous_results: Dict[str, Any], ablation_mode: str = "adaptive", downstream_agent: str = "rag") -> str:
        # Reset per-call state so stale values from previous nodes never bleed into
        # the comm_tracker low_confidence flag for this edge.
        self._last_preliminary_crs = 1.0

        if not depends_on or not self.llm:
            return query

        context = "\n".join([f"--- Result ---\n{v}" for k, v in previous_results.items() if k in depends_on])
        
        if ablation_mode == "none": 
            return query 
        
        if ablation_mode == "raw": 
            return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"

        if ablation_mode == "adaptive":
            # Scholar's API search works best with the clean original query.
            # Injecting upstream RAG prose corrupts the search and hurts expert-finding.
            if downstream_agent == "scholar":
                return query

            manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
            budget_limit = manifest.token_budget
            upstream_token_count = len(context) // 4
            self._last_cugcr_triggered = False  # reset for this specific call

            if upstream_token_count <= budget_limit:
                print(f"      [Gateway] Context ({upstream_token_count} tok) fits in {downstream_agent.upper()} budget ({budget_limit} tok). Bypassing CUGCR -> RAW Passing.")
                return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"

            print(f"      [Gateway] Context ({upstream_token_count} tok) EXCEEDS {downstream_agent.upper()} budget ({budget_limit} tok). Triggering CUGCR Compression.")
            self.current_run_cugcr_triggered = True
            self._last_cugcr_triggered = True

            # ── FIX 1: CRS Gate ───────────────────────────────────────────────────────
            # Compute a quick preliminary relevance score using dense embeddings
            # before committing to compression. If upstream quality is too low,
            # compressing it will make the noise denser and harder to ignore.
            preliminary_crs = 1.0  # default: assume ok if embedder not available
            if self.embedder:
                try:
                    # Use compression-goal query for CRS scoring so cross-agent chains
                    # (CODE→CLOUD) are evaluated against what the downstream agent needs,
                    # not the raw task query. This prevents useful contexts from being
                    # blocked by the gate just because the queries are domain-shifted.
                    _crs_scoring_q = _compression_goal_for(previous_results, downstream_agent, query)
                    ctx_emb   = self.embedder.encode([context[:1500]])
                    query_emb = self.embedder.encode([_crs_scoring_q])
                    preliminary_crs = float(cosine_similarity(query_emb, ctx_emb)[0][0])
                except Exception:
                    preliminary_crs = 1.0

            if preliminary_crs < 0.25:
                # Upstream context is genuinely irrelevant — compression amplifies noise.
                # 0.25 threshold restored now that CRS uses the compression-goal query,
                # which naturally scores higher for valid cross-agent chains.
                print(f"      [CRS Gate] Preliminary CRS={preliminary_crs:.2f} is below threshold. Skipping CUGCR — injecting warning instead.")
                warning = (
                    f"⚠️ LOW CONFIDENCE CONTEXT (CRS={preliminary_crs:.2f}): "
                    f"Upstream retrieval quality is low. Validate all references independently.\n\n"
                )
                self._last_preliminary_crs = preliminary_crs  # record so low_confidence flag is accurate
                self.current_run_cugcr_triggered = False  # didn't actually compress
                self._last_cugcr_triggered = False        # CRS gate blocked — not a real CUGCR trigger
                return f"{warning}{query}\n\n[UPSTREAM CONTEXT (truncated)]:\n{context[:400]}"
            # ── End Fix 1 ─────────────────────────────────────────────────────────────

            # For the normal CUGCR path, also save it so got_executor always has a value
            self._last_preliminary_crs = preliminary_crs       # ← AND HERE (normal path)

            # ── Compression-goal routing ──────────────────────────────────────
            # For cross-agent chains (e.g. CODE→CLOUD) the downstream task query
            # is semantically distant from what was in the upstream context, so
            # generic cosine similarity drops useful atoms. Use a targeted scoring
            # query that describes what the downstream agent actually needs.
            scoring_query = _compression_goal_for(previous_results, downstream_agent, query)
            if scoring_query != query:
                print(f"      [CUGCR] Using compression goal for {downstream_agent.upper()}: '{scoring_query[:80]}...'")

            atoms = atomize_output(context)
            if getattr(config.Config, "CONTEXT_ROUTING_MODE", "light") == "heavy" and self.embedder:
                selected_atoms = self._score_and_select_atoms(atoms, scoring_query, downstream_agent)

                code_blocks = [a.content for a in selected_atoms if a.type == AtomType.CODE_BLOCK]
                entities    = list(set([a.content for a in selected_atoms if a.type == AtomType.NAMED_ENTITY]))
                prose       = [a.content for a in selected_atoms if a.type == AtomType.PROSE_CLAIM]

                context_block = ""
                if code_blocks: context_block += "### Upstream Code Artifacts\n"     + "\n".join(code_blocks) + "\n\n"
                if entities:    context_block += "### Key Technical Entities\n"       + ", ".join(entities[:10]) + "\n\n"
                if prose:       context_block += "### Relevant Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose) + "\n\n"

                return f"{context_block}---\n## Your Task\n{query}\n"
            else:
                return build_rigid_prompt(original_query=query, raw_context=context, token_budget=budget_limit, downstream_agent=downstream_agent)
            
        return query

    async def _gpt_as_judge(self, user_input: str, final_answer: str) -> str:
        if not self.openai_client:
            return "Score: 0/10 - OpenAI API Key missing for Judge."
        judge_prompt = f"""
        You are an impartial AI Judge evaluating a multi-agent system.
        User Query: "{user_input}"
        System Answer: "{final_answer}"
        
        CRITERIA (0 to 10):
        - Did it perform ALL tasks? (Deduct 3 points for missing tasks).
        - If asked for code, is there code? If asked for experts, are there real names?
        - CRITICAL DEPENDENCY CHECK: If the query asks the system to use specific information, but the answer looks like generic LLM hallucinations or ignores the dependencies, SCORE IT A MAXIMUM OF 4/10.
        
        Format exactly as: 'Score: X/10 - [Reason]'
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=config.Config.JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Score: 0/10 - Judge Failed: {e}"

    def _sanitize_placeholders(self, query: str, parent_nodes: list) -> str:
        """
        Last-resort pass: replace every unresolved placeholder with a short
        excerpt of the actual parent output.  Called after _resolve_dependencies,
        so it only fires when something slipped through (e.g. the low-CRS
        warning path that skips CUGCR compression).
    
        Parameters
        ----------
        query        : the query string that will be sent to _execute_agent
        parent_nodes : list of GoTNode objects whose .result holds upstream output
    
        Returns
        -------
        query with all placeholder patterns replaced by real content.
        """
        if not _PLACEHOLDER_RE.search(query):
            return query   # fast exit — nothing to fix
    
        # Build a compact summary of all parent outputs (≤300 chars each)
        excerpts = []
        for i, node in enumerate(parent_nodes or []):
            content = (node.result or "").strip()
            if content:
                excerpts.append(f"[Upstream result {i + 1}]: {content[:300]}")
    
        replacement = (
            "\n".join(excerpts)
            if excerpts
            else "No upstream result available — proceed with general knowledge."
        )
    
        sanitized = _PLACEHOLDER_RE.sub(replacement, query)
    
        if sanitized != query:
            print(
                f"      [Placeholder Fix] Resolved {len(_PLACEHOLDER_RE.findall(query))} "
                f"unsubstituted reference(s) before agent call."
            )
    
        return sanitized

    async def decompose_only(self, user_input: str) -> tuple:
        """Decompose the query once and return (tasks, decomposer_name).

        Call this once per benchmark query, then pass the result to run() via
        cached_tasks= so all three ablation modes share the identical DAG plan.
        This eliminates the non-deterministic decomposer as a confound.
        """
        if config.Config.DEFAULT_DECOMPOSER == "auto":
            decomposer_name = self.selector.select(user_input, deterministic=False).name
        elif config.Config.DEFAULT_DECOMPOSER == "ultimate_neuro_symbolic":
            decomposer_name = "neuro_rule"
        else:
            decomposer_name = config.Config.DEFAULT_DECOMPOSER

        decomposer = self.registry.get(decomposer_name)

        # Build profile hint from a clean per-query classification (no Dirichlet state)
        _dim_scores = classify_query(user_input)
        _dom        = max(_dim_scores, key=_dim_scores.get)
        _hint       = _build_profile_hint(_dom, _dim_scores)
        if isinstance(decomposer, NeuroRuleDecomposer):
            tasks = await decomposer.decompose(user_input, self.llm, profile_hint=_hint)
        else:
            tasks = await decomposer.decompose(user_input, self.llm)

        print(f"   [Decompose-Once] {len(tasks)} tasks via {decomposer_name.upper()} | dim={_dom.upper()}:")
        for t in tasks:
            deps_str = f" (depends: {t.depends_on})" if t.depends_on else ""
            print(f"      Task {t.id} → [{t.agent.upper()}]: '{t.query[:60]}'{deps_str}")

        return tasks, decomposer_name

    async def _execute_pipeline(self, user_input: str, decomposer_name: str, ablation_mode: str = "adaptive", start_time: float = None, cached_tasks=None) -> Dict[str, Any]:
        self.current_run_cugcr_triggered = False
        self._node_metadata = {}

        profiler_t0 = time.monotonic()
        profiler_task = None
        if config.Config.USE_PROFILER:
            profiler_task = asyncio.create_task(user_profiler.update_profile(user_input, self.llm))

        # ── Per-query intent classification (stateless, instant) ─────────────────
        # Use _classify_deterministic so we get a clean signal for THIS query
        # regardless of accumulated Dirichlet history. In benchmark mode this is
        # the authoritative profile; in interactive mode the Dirichlet profile is
        # used for long-term style adaptation while this drives depth instructions.
        _query_dim_scores = classify_query(user_input)
        _query_dom_dim    = max(_query_dim_scores, key=_query_dim_scores.get)
        self._current_query_dim = _query_dom_dim   # read by _execute_agent
        print(f"   [Profiler] Per-query dim={_query_dom_dim.upper()} "
              f"(C={_query_dim_scores['conceptual']:.0f}% "
              f"P={_query_dim_scores['procedural']:.0f}% "
              f"A={_query_dim_scores['analytical']:.0f}%)")

        decomp_t0 = time.monotonic()
        if cached_tasks is not None:
            # DAG plan pre-computed by decompose_only() — skip the LLM decomposition call.
            tasks = cached_tasks
            decomposer_latency_ms = 0
        else:
            decomposer = self.registry.get(decomposer_name)
            _hint = _build_profile_hint(_query_dom_dim, _query_dim_scores)
            if isinstance(decomposer, NeuroRuleDecomposer):
                tasks = await decomposer.decompose(user_input, self.llm, profile_hint=_hint)
            else:
                tasks = await decomposer.decompose(user_input, self.llm)
            decomposer_latency_ms = round((time.monotonic() - decomp_t0) * 1000)

        print(f"\n   [Plan for {decomposer_name.upper()}]:")
        for t in tasks:
            deps_str = f" (Depends on: {t.depends_on})" if t.depends_on else ""
            print(f"      Task {t.id} -> [{t.agent.upper()}]: '{t.query}'{deps_str}")

        dag_dict = self.dag_builder.build(tasks)
        results = {}
        profiler_latency_ms = None
        orchestrator_tokens = {"input": 0, "output": 0}

        if dag_dict.get("is_got"):
            if profiler_task:
                await profiler_task
            profiler_latency_ms = round((time.monotonic() - profiler_t0) * 1000)

            # Style guideline: in benchmark mode use the clean per-query dim
            # (no Dirichlet accumulation bleed). In interactive mode use the
            # long-term Dirichlet profile for cross-session style consistency.
            if config.Config.USE_PROFILER:
                if config.Config.RUN_MODE == "benchmark":
                    # Build a one-shot style guideline from the per-query classification
                    _dom_hints = _PROFILE_DEPTH_HINTS.get(_query_dom_dim, {})
                    style_guideline = (
                        f"RESPONSE STYLE PROFILE (per-query, no accumulation):\n"
                        f"  Dominant intent: {_query_dom_dim.upper()} "
                        f"({_query_dim_scores[_query_dom_dim]:.0f}%)\n"
                        f"  Synthesizer instruction: {_dom_hints.get('rag', '')}"
                    )
                else:
                    style_guideline = user_profiler.get_style_guideline(use_session=False)
            else:
                style_guideline = ""

            got_orchestrator = AdaptiveGoTOrchestrator(
                quality_threshold=0.80,
                beam_width=2,
                max_expansions=6,
                depth_budget=DepthBudget(base_depth=1, weight_per_level=20, max_depth=3)
            )

            async def got_executor(node: GoTNode) -> str:
                query = node.task.query
                parent_nodes = [got_orchestrator.dag._nodes[pid] for pid in node.parent_ids] if got_orchestrator.dag else []

                if node.op == GoTOp.AGGREGATE:
                    print(f"      [GoT Operator] Safe Processing for AGGREGATE node...")
                    real_parents = [p for p in parent_nodes if p.result]
                    if len(real_parents) == 1:
                        return real_parents[0].result
                    full_context = "\n\n".join([f"--- Source {i+1} ---\n{p.result}" for i, p in enumerate(real_parents)])
                    agg_prompt = f"""
                    Synthesize a final response to the user's original query using ONLY these outputs:
                    User's Original Query: "{user_input}"
                    Agent Outputs:
                    {full_context}
                    {style_guideline}
                    """
                    res = await asyncio.to_thread(self.llm.generate_content, agg_prompt.strip())
                    if hasattr(res, "usage_metadata") and res.usage_metadata:
                        orchestrator_tokens["input"]  += getattr(res.usage_metadata, "prompt_token_count", 0) or 0
                        orchestrator_tokens["output"] += getattr(res.usage_metadata, "candidates_token_count", 0) or 0
                    return res.text.strip()

                if node.op == GoTOp.REFINE:
                    print(f"      [GoT Operator] Safe Processing for REFINE node with Reflexion...")
                    if ablation_mode == "none":
                        result, _ = await self._execute_agent(node.task.agent, query, depth=node.depth)
                        return result

                    previous_answer = parent_nodes[0].result if parent_nodes else "No previous answer recorded."
                    original_query = parent_nodes[0].task.query if parent_nodes else query

                    refine_prompt = f"""
                    ## ⚠ Retry Attempt Required for Agent: {node.task.agent.upper()}
                    ### Original Task:
                    {original_query}
                    ### Your Previous Failed Attempt:
                    {previous_answer}
                    ### Judge's Critique:
                    {node.judge_reasoning}

                    ### CRITICAL INSTRUCTION:
                    1. Write a <reflection> block analyzing your mistake.
                    2. Write a <revised_query> that will be sent to the {node.task.agent.upper()} agent to try again.
                       IMPORTANT: If the original task asked for a specific number of items (like 2 experts, 3 papers), your revised query MUST explicitly include that exact number (e.g., "Find 2 experts in...").

                    Format:
                    <reflection>
                    ...
                    </reflection>
                    <revised_query>
                    ...
                    </revised_query>
                    """
                    res = await asyncio.to_thread(self.llm.generate_content, refine_prompt.strip())

                    revised_query = original_query
                    match = re.search(r'<revised_query>(.*?)</revised_query>', res.text, re.DOTALL)
                    if match:
                        revised_query = match.group(1).strip()

                    print(f"      [Reflexion] Revised Query: {revised_query[:60]}...")
                    full_query_to_agent = f"{revised_query}\n\n[Previous Attempt Reflection]:\n{res.text.strip()}"
                    result, _ = await self._execute_agent(node.task.agent, full_query_to_agent, depth=node.depth)
                    return result

                node_cugcr = False
                node_crs   = self._last_preliminary_crs
                if node.parent_ids and node.op not in (GoTOp.AGGREGATE, GoTOp.REFINE):
                    prev_results = {p.node_id: p.result for p in parent_nodes if p.result}
                    resolved_query = await self._resolve_dependencies(query, node.parent_ids, prev_results, ablation_mode, downstream_agent=node.task.agent)
                    node_cugcr = self._last_cugcr_triggered
                    node_crs   = self._last_preliminary_crs

                    injected_text = resolved_query.replace(query, "").strip()
                    current_wave = len(got_orchestrator.dag.wave_scores) + 1 if got_orchestrator.dag else 1
                    comm_tracker.log_injection(
                        wave=current_wave,
                        from_ids=node.parent_ids,
                        to_id=node.node_id,
                        to_agent=node.task.agent,
                        context=injected_text,
                        orig_query=query,
                        resolved_query=resolved_query,
                        low_confidence=(ablation_mode == "adaptive" and self._last_preliminary_crs < 0.15)
                    )
                    query = resolved_query

                query = self._sanitize_placeholders(query, parent_nodes)

                _exec_t0 = time.monotonic()
                print(f"      [Executing] Node {node.node_id[:4]} ({node.task.agent.upper()}): {query[:60]}...")
                result, agent_meta = await self._execute_agent(node.task.agent, query, depth=node.depth)
                print(f"      [Done]      Node {node.node_id[:4]} ({node.task.agent.upper()}) ← {time.monotonic()-_exec_t0:.1f}s")

                self._node_metadata[node.node_id] = {
                    **agent_meta,
                    "cugcr_triggered": node_cugcr,
                    "preliminary_crs": round(node_crs, 3),
                }

                if node.parent_ids and node.op not in (GoTOp.AGGREGATE, GoTOp.REFINE):
                    comm_tracker.update_result(to_id=node.node_id, output=result, quality=0.0)

                return result

            print("\n   [GoT DAG] Commencing Adaptive Wave Execution...")
            final_answer = await got_orchestrator.run(
                tasks=tasks,
                user_profile=user_profiler.profile if config.Config.USE_PROFILER else None, 
                model=self.llm,
                agent_executor=got_executor
            )
            
        else:
            final_answer = "Error: System currently requires 'got_dag' builder."

        print("⚖️  [Orchestrator] Final Global Evaluation...")
        judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
        match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
        numeric_score = int(match.group(1)) if match else 0

        # First pass: build dag_nodes_dump with execution metadata only.
        # Context metrics (relevance, utilization, dependency_accuracy) are computed
        # later by the comm_tracker block — a second enrichment pass fills them in.
        dag_nodes_dump = []
        if dag_dict.get("is_got") and 'got_orchestrator' in locals() and got_orchestrator.dag:
            for n_id, node in got_orchestrator.dag._nodes.items():
                n_meta = self._node_metadata.get(n_id, {})
                dag_nodes_dump.append({
                    "node_id":             n_id,
                    "agent":               node.task.agent,
                    "query":               node.task.query,
                    "operator":            node.op.value,
                    "depth":               node.depth,
                    "parent_ids":          node.parent_ids,
                    "failure_mode":        node.failure_mode.value if node.failure_mode else None,
                    "quality_score":       node.quality_score,
                    "judge_critique":      node.judge_reasoning,
                    "elapsed_s":           node.elapsed_s,
                    "output":              node.result,
                    # context metrics — populated in second pass after comm_tracker computes them
                    "context_relevance_pct": None,
                    "context_utilized_pct":  None,
                    "context_tokens":        0,
                    "crs_gate_blocked":      False,
                    "dependency_accuracy":   True,
                    # per-node execution metadata (available now)
                    "cugcr_triggered":     n_meta.get("cugcr_triggered", False),
                    "preliminary_crs":     n_meta.get("preliminary_crs", None),
                    "paper_count":         n_meta.get("paper_count", None),
                    "rag_strategy":        n_meta.get("strategy", None),
                })

        retry_events = []
        retries = 0
        max_retries = 2

        while numeric_score < 7 and retries < max_retries:
            print(f"   [Global Diagnosis] Score {numeric_score}/10 is low. Diagnosing failure...")
            prev_score  = numeric_score
            prev_answer = final_answer
            retries += 1

            node_trace = ""
            for n in dag_nodes_dump:
                if n.get("operator") not in ("aggregate",):
                    output_preview = (n.get("output") or "")[:400]
                    node_trace += (
                        f"\n--- Agent: {n.get('agent','?').upper()} "
                        f"| Operator: {n.get('operator','?')} "
                        f"| NQS: {n.get('quality_score','?')} ---\n"
                        f"Query: {n.get('query','')[:200]}\n"
                        f"Output preview: {output_preview}\n"
                        f"Failure mode: {n.get('failure_mode', 'None')}\n"
                        f"Judge critique: {n.get('judge_critique', 'None')}\n"
                    )

            diag_prompt = f"""
            The final answer failed to satisfy the original query.

            ORIGINAL QUERY: "{user_input}"

            FULL AGENT EXECUTION TRACE (use this to find the root cause):
            {node_trace}

            FINAL SYNTHESIZED ANSWER:
            "{final_answer}"

            JUDGE SCORE: {judge_score_raw}

            Based on the execution trace above, identify EXACTLY which agent produced
            the bad output that caused the final answer to fail. Look at the NQS scores
            and failure modes — the lowest-scoring agent is likely the culprit.

            Available agents: {config.Config.AGENT_REGISTRY_PROMPT}

            IMPORTANT: If the original query asked for a specific number of items (e.g., 2 experts), the revised_query MUST explicitly include that number so the downstream agent knows the quota.

            Output ONLY a JSON object:
            {{"failing_agent": "scholar", "revised_query": "specific corrective instruction"}}
            """
            try:
                res = await asyncio.to_thread(self.llm.generate_content, diag_prompt)
                diag_data = json.loads(res.text.replace('```json', '').replace('```', '').strip())
                agent = diag_data.get('failing_agent', 'rag')
                new_q = diag_data.get('revised_query', user_input)

                print(f"   [Diagnostician] Retrying agent '{agent}' with query: '{new_q}' at HIGHER DEPTH (3).")

                # Inject CUGCR-compressed context from successful peer nodes so the
                # retrying agent sees what others already produced.
                _ctx_ids = [
                    n["node_id"] for n in dag_nodes_dump
                    if n.get("output") and n.get("agent") != agent
                    and n.get("operator") not in ("aggregate",)
                ]
                _ctx_results = {
                    n["node_id"]: n["output"] for n in dag_nodes_dump
                    if n.get("output") and n.get("agent") != agent
                    and n.get("operator") not in ("aggregate",)
                }
                if _ctx_ids:
                    new_q = await self._resolve_dependencies(
                        new_q, _ctx_ids, _ctx_results, "adaptive", downstream_agent=agent
                    )
                    print(f"   [Diagnostician] CUGCR context from {len(_ctx_ids)} peer node(s) injected.")

                fix_result, _ = await self._execute_agent(agent, new_q, depth=3)

                synth_prompt = f"Update the final answer with this new information.\nOld Answer:\n{final_answer}\n\nNew Info:\n{fix_result}\n\nNew Final Answer:"
                final_answer = (await asyncio.to_thread(self.llm.generate_content, synth_prompt)).text.strip()

                judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
                print(f"   [Re-Judge] New Judge Score after retry: {judge_score_raw}")
                match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
                numeric_score = int(match.group(1)) if match else 0

                retry_events.append({
                    "retry_num":      retries,
                    "failing_agent":  agent,
                    "revised_query":  new_q[:200],
                    "score_before":   prev_score,
                    "score_after":    numeric_score,
                    "improved":       numeric_score > prev_score,
                })

                if numeric_score <= prev_score:
                    final_answer = prev_answer
                    numeric_score = prev_score
                    break

            except Exception as e:
                print(f"[Diagnostics Error] Global retry failed: {e}")
                break

        latency = round(time.time() - start_time, 2) if start_time else 0.0

        if comm_tracker.logs:
            try:
                from orchestrator.evaluation.communication_evaluator import compute_cus_deterministic
                import re as _re

                PLACEHOLDER_RE = _re.compile(
                    r"\[result of|\[explanation of|\[output of|"
                    r"\[PLACEHOLDER|@@REF@@|\[goal \d|\[task \d",
                    _re.IGNORECASE
                )

                for log in comm_tracker.logs:
                    has_context = bool(log.raw_context_injected)
                    has_output  = bool(log.agent_output)

                    # ── CUS ──────────────────────────────────────────────────
                    if has_context and has_output:
                        log.utilization_score = compute_cus_deterministic(
                            injected_context  = log.raw_context_injected,
                            downstream_output = log.agent_output,
                        )
                    else:
                        log.utilization_score = 0.0

                    # ── CRS ──────────────────────────────────────────────────
                    # CRS = cosine similarity between context embedding and
                    # the downstream query. compute_cus_deterministic is a
                    # token-overlap scorer designed for long outputs — using it
                    # on a short query always yields ~0% due to the length
                    # penalty. Dense embedding similarity is the right metric.
                    if has_context and log.original_query and self.embedder:
                        try:
                            ctx_emb   = self.embedder.encode([log.raw_context_injected[:800]])
                            qry_emb   = self.embedder.encode([log.original_query])
                            log.context_relevance = float(cosine_similarity(qry_emb, ctx_emb)[0][0])
                        except Exception:
                            log.context_relevance = 0.0
                    else:
                        log.context_relevance = 0.0

                    # ── Dependency Accuracy ───────────────────────────────────
                    if has_output:
                        log.dependency_accuracy = not bool(
                            PLACEHOLDER_RE.search(log.agent_output)
                        )
                    else:
                        log.dependency_accuracy = False

                print("\n   [Comm Summary] Context Injection Performance:")
                for log in comm_tracker.logs:
                    status    = "✅" if log.dependency_accuracy else "❌ (Placeholder Leak)"
                    rel       = (log.context_relevance or 0) * 100
                    util      = (log.utilization_score or 0) * 100
                    ctx_label = f"{log.context_token_count} tok" if log.context_token_count else "no context"
                    print(f"      -> To [{log.to_agent.upper()}]: "
                          f"Relevance {rel:.0f}% | Utilized {util:.0f}% | "
                          f"Context: {ctx_label} {status}")
            except Exception as e:
                print(f"   [Comm Evaluator Error] {e}")

        # Second pass: enrich dag_nodes_dump with now-computed context metrics.
        # EdgeLog objects were mutated in-place above — read the final values here.
        _edge_by_node = {log.to_id: log for log in comm_tracker.logs}
        for node_dict in dag_nodes_dump:
            edge = _edge_by_node.get(node_dict["node_id"])
            if edge:
                node_dict["context_relevance_pct"] = round((edge.context_relevance or 0) * 100, 1)
                node_dict["context_utilized_pct"]  = round((edge.utilization_score  or 0) * 100, 1)
                node_dict["context_tokens"]        = edge.context_token_count or 0
                node_dict["crs_gate_blocked"]      = edge.low_confidence
                node_dict["dependency_accuracy"]   = edge.dependency_accuracy

        comm_data = comm_tracker.export_to_dict()
        comm_tracker.clear()

        run_telemetry = {
            # ── Identity ─────────────────────────────────────────────────────────
            "timestamp":             datetime.now().isoformat(),
            "ablation_mode":         ablation_mode,
            "decomposer_used":       decomposer_name,
            "use_profiler":          config.Config.USE_PROFILER,
            # ── Query ────────────────────────────────────────────────────────────
            "user_query":            user_input,
            # ── Latency ──────────────────────────────────────────────────────────
            "latency_seconds":       latency,
            "profiler_latency_ms":   profiler_latency_ms,
            "decomposer_latency_ms": decomposer_latency_ms,
            # ── Orchestrator token usage (aggregator + global retries) ────────────
            "orchestrator_tokens":   orchestrator_tokens,
            # ── Profiler ─────────────────────────────────────────────────────────
            "profiler_state":        user_profiler.profile.profile if config.Config.USE_PROFILER else {},
            # ── DAG plan ─────────────────────────────────────────────────────────
            "initial_tasks":         [{"id": t.id, "agent": t.agent, "query": t.query} for t in tasks],
            "got_dag_nodes":         dag_nodes_dump,
            # ── Communication edges ───────────────────────────────────────────────
            "communication_edges":   comm_data,
            # ── CUGCR ────────────────────────────────────────────────────────────
            "cugcr_triggered":       self.current_run_cugcr_triggered,
            # ── Self-healing / Reflexion ──────────────────────────────────────────
            "first_pass_success":    retries == 0,
            "global_retries":        retries,
            "retry_events":          retry_events,
            # ── Final judge ───────────────────────────────────────────────────────
            "final_judge_score":     numeric_score,
            "final_judge_raw":       judge_score_raw,
            "final_answer":          final_answer,
        }

        telemetry_file = "experiment_telemetry.json"
        all_data = []
        if os.path.exists(telemetry_file):
            try:
                with open(telemetry_file, "r") as f:
                    all_data = json.load(f)
            except json.JSONDecodeError:
                pass 

        all_data.append(run_telemetry)

        with open(telemetry_file, "w") as f:
            json.dump(all_data, f, indent=4)

        return {
            "final_answer": final_answer, 
            "judge_score": judge_score_raw, 
            "numeric_score": numeric_score,
            "communication_metrics": comm_data
        }

    async def run(self, user_input: str, ablation_mode: str = "adaptive", start_time: float = None,
                  cached_tasks=None, cached_decomposer_name: str = None) -> Dict[str, Any]:
        """Run the full pipeline.

        Parameters
        ----------
        cached_tasks : list, optional
            Pre-decomposed task list from decompose_only().  When supplied together
            with cached_decomposer_name, the LLM decomposition step is skipped so
            all ablation modes operate on the identical DAG plan.
        cached_decomposer_name : str, optional
            Decomposer name returned by decompose_only().
        """
        if cached_tasks is not None and cached_decomposer_name is not None:
            print(f"⚙️  [Orchestrator] Using cached DAG plan ({len(cached_tasks)} tasks, {cached_decomposer_name.upper()}) — mode={ablation_mode.upper()}")
            return await self._execute_pipeline(user_input, cached_decomposer_name, ablation_mode, start_time, cached_tasks=cached_tasks)

        # Fresh decomposition (used when called standalone, not via ablation benchmark)
        if config.Config.DEFAULT_DECOMPOSER == "auto":
            decomposer_name = self.selector.select(user_input, deterministic=False).name
        elif config.Config.DEFAULT_DECOMPOSER == "ultimate_neuro_symbolic":
            decomposer_name = "neuro_rule"
        else:
            decomposer_name = config.Config.DEFAULT_DECOMPOSER

        print(f"⚙️  [Orchestrator] Selected Decomposer: {decomposer_name.upper()}")
        return await self._execute_pipeline(user_input, decomposer_name, ablation_mode, start_time)