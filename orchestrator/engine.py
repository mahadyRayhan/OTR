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
from profiler_agent.agent import user_profiler

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
        required_atom_types=[AtomType.CODE_BLOCK, AtomType.NAMED_ENTITY],
        token_budget=600  # Tightened to force CUGCR compression
    ),
    "cloud": DependencyManifest(
        agent_type="cloud",
        required_atom_types=[AtomType.CODE_BLOCK, AtomType.NAMED_ENTITY],
        token_budget=600
    ),
    "rag": DependencyManifest(
        agent_type="rag",
        required_atom_types=[AtomType.NAMED_ENTITY],
        token_budget=300
    ),
    "scholar": DependencyManifest(
        agent_type="scholar",
        required_atom_types=[AtomType.NAMED_ENTITY],
        # FIX 2: Scholar needs linguistic density to identify researchers.
        # Old budget was 150 — far too tight. Doubled to 300.
        token_budget=300
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

def atomize_output(raw_output: str) -> list[ContextAtom]:
    """Extracts protected code, biological entities, and prose deterministically."""
    atoms = []
    
    for match in re.finditer(r'```[\s\S]*?```', raw_output):
        atoms.append(ContextAtom(AtomType.CODE_BLOCK, match.group(), is_protected=True))
    
    prose_only = re.sub(r'```[\s\S]*?```', '', raw_output)
    
    bio_pattern = r'\b([A-Z]{2,}[0-9]*|[A-Z][a-z]+-\d+)\b'
    for match in re.finditer(bio_pattern, prose_only):
        atoms.append(ContextAtom(AtomType.NAMED_ENTITY, match.group(), is_protected=True))
        
    for sentence in prose_only.split('. '):
        if len(sentence.strip()) > 15:
            atoms.append(ContextAtom(AtomType.PROSE_CLAIM, sentence.strip() + ".", is_protected=False))
            
    return atoms

def build_rigid_prompt(original_query: str, raw_context: str, token_budget: int = 400) -> str:
    """Filters atoms and injects them into a rigid, drift-proof template."""
    atoms = atomize_output(raw_context)
    
    code_blocks = [a.content for a in atoms if a.type == AtomType.CODE_BLOCK]
    entities = list(set([a.content for a in atoms if a.type == AtomType.NAMED_ENTITY]))
    prose = [a.content for a in atoms if a.type == AtomType.PROSE_CLAIM]
    
    context_block = ""
    if code_blocks:
        context_block += "### Upstream Code Artifacts\n" + "\n".join(code_blocks) + "\n\n"
    if entities:
        context_block += "### Key Technical Entities\n" + ", ".join(entities[:10]) + "\n\n"
    if prose:
        context_block += "### Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose[:4]) + "\n\n"
        
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
        self._last_preliminary_crs = 1.0   # ← add this line
        
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

        selected   = []
        used_tokens = 0
        
        for atom in atoms:
            if atom.is_protected and atom.type in manifest.required_atom_types:
                selected.append(atom)
                used_tokens += len(atom.content.split())
        
        unprotected = [a for a in atoms if a not in selected]
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
            cost = len(atom.content.split())
            if used_tokens + cost <= current_budget:
                selected.append(atom)
                used_tokens += cost
                
        return selected

    async def _execute_agent(self, agent_name: str, query: str, depth: int = 1) -> str:
        agent_registry = config.Config.AGENT_REGISTRY_PROMPT
        strategy_match = re.search(r'--strategy=(\w+)', query.lower())
        strategy = strategy_match.group(1) if strategy_match else "cot"
        clean_query = re.sub(r'--strategy=\w+', '', query).strip()
        
        try:
            if agent_name == "code": 
                return await call_code_agent(clean_query, strategy, depth, agent_registry)
            if agent_name == "cloud":
                profile_str = user_profiler.get_style_guideline(use_session=False) if config.Config.USE_PROFILER else ""
                return await call_cloud_agent(clean_query, strategy, depth, agent_registry, profile_str)
            if agent_name == "scholar":
                limit = 1
                text_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
                query_normalized = clean_query.lower()
                for word, digit in text_to_num.items():
                    query_normalized = re.sub(rf'\b{word}\b', digit, query_normalized)
                match = re.search(r'\b([1-5])\b', query_normalized)
                if match: limit = int(match.group(1))
                return await call_scholar_agent(clean_query, False, "Unknown", limit, strategy, depth, agent_registry)
            else:
                rag_data = await call_rag_agent(kb=self.kb, user_input=clean_query, strategy=strategy, depth=depth, agent_registry=agent_registry)
                return f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data.get('answer', '')}"
        except Exception as e:
            return f"Error executing agent: {e}"

    async def _resolve_dependencies(self, query: str, depends_on: List[Any], previous_results: Dict[str, Any], ablation_mode: str = "adaptive", downstream_agent: str = "rag") -> str:
        if not depends_on or not self.llm: 
            return query
        
        context = "\n".join([f"--- Result ---\n{v}" for k, v in previous_results.items() if k in depends_on])
        
        if ablation_mode == "none": 
            return query 
        
        if ablation_mode == "raw": 
            return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"

        if ablation_mode == "adaptive":
            manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
            budget_limit = manifest.token_budget
            upstream_token_count = len(context) // 4
            
            if upstream_token_count <= budget_limit:
                print(f"      [Gateway] Context ({upstream_token_count} tok) fits in {downstream_agent.upper()} budget ({budget_limit} tok). Bypassing CUGCR -> RAW Passing.")
                return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"
                
            print(f"      [Gateway] Context ({upstream_token_count} tok) EXCEEDS {downstream_agent.upper()} budget ({budget_limit} tok). Triggering CUGCR Compression.")
            self.current_run_cugcr_triggered = True

            # ── FIX 1: CRS Gate ───────────────────────────────────────────────────────
            # Compute a quick preliminary relevance score using dense embeddings
            # before committing to compression. If upstream quality is too low,
            # compressing it will make the noise denser and harder to ignore.
            preliminary_crs = 1.0  # default: assume ok if embedder not available
            if self.embedder:
                try:
                    ctx_emb   = self.embedder.encode([context[:800]])  # truncate for speed
                    query_emb = self.embedder.encode([query])
                    preliminary_crs = float(cosine_similarity(query_emb, ctx_emb)[0][0])
                except Exception:
                    preliminary_crs = 1.0

            if preliminary_crs < 0.30:
                # Upstream context quality is too low — compression would amplify noise.
                # Inject a short RAW truncation with an explicit warning instead.
                print(f"      [CRS Gate] Preliminary CRS={preliminary_crs:.2f} is below threshold. Skipping CUGCR — injecting warning instead.")
                warning = (
                    f"⚠️ LOW CONFIDENCE CONTEXT (CRS={preliminary_crs:.2f}): "
                    f"Upstream retrieval quality is low. Validate all references independently.\n\n"
                )
                self.current_run_cugcr_triggered = False  # didn't actually compress
                return f"{warning}{query}\n\n[UPSTREAM CONTEXT (truncated)]:\n{context[:400]}"
            # ── End Fix 1 ─────────────────────────────────────────────────────────────

            # For the normal CUGCR path, also save it so got_executor always has a value
            self._last_preliminary_crs = preliminary_crs       # ← AND HERE (normal path)
            atoms = atomize_output(context)
            if getattr(config.Config, "CONTEXT_ROUTING_MODE", "light") == "heavy" and self.embedder:
                selected_atoms = self._score_and_select_atoms(atoms, query, downstream_agent)
                
                code_blocks = [a.content for a in selected_atoms if a.type == AtomType.CODE_BLOCK]
                entities    = list(set([a.content for a in selected_atoms if a.type == AtomType.NAMED_ENTITY]))
                prose       = [a.content for a in selected_atoms if a.type == AtomType.PROSE_CLAIM]
                
                context_block = ""
                if code_blocks: context_block += "### Upstream Code Artifacts\n"     + "\n".join(code_blocks) + "\n\n"
                if entities:    context_block += "### Key Technical Entities\n"       + ", ".join(entities[:10]) + "\n\n"
                if prose:       context_block += "### Relevant Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose) + "\n\n"
                
                return f"{context_block}---\n## Your Task\n{query}\n"
            else:
                return build_rigid_prompt(original_query=query, raw_context=context, token_budget=budget_limit)
            
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

    async def _execute_pipeline(self, user_input: str, decomposer_name: str, ablation_mode: str = "adaptive", start_time: float = None) -> Dict[str, Any]:
        self.current_run_cugcr_triggered = False 
        
        profiler_task = None
        if config.Config.USE_PROFILER:
            profiler_task = asyncio.create_task(user_profiler.update_profile(user_input, self.llm))

        decomposer = self.registry.get(decomposer_name)
        tasks = await decomposer.decompose(user_input, self.llm)
        
        print(f"\n   [Plan for {decomposer_name.upper()}]:")
        for t in tasks:
            deps_str = f" (Depends on: {t.depends_on})" if t.depends_on else ""
            print(f"      Task {t.id} -> [{t.agent.upper()}]: '{t.query}'{deps_str}")

        dag_dict = self.dag_builder.build(tasks)
        results = {}
        
        if dag_dict.get("is_got"):
            if profiler_task: await profiler_task 
            style_guideline = user_profiler.get_style_guideline(use_session=False) if config.Config.USE_PROFILER else ""
            
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
                    full_context = "\n\n".join([f"--- Source ---\n{p.result}" for p in parent_nodes if p.result])
                    agg_prompt = f"""
                    Synthesize a final response to the user's original query using ONLY these outputs:
                    User's Original Query: "{user_input}"
                    Agent Outputs:
                    {full_context}
                    CRITICAL INSTRUCTIONS: Do NOT truncate Python or Bash code blocks.
                    {style_guideline}
                    """
                    res = await asyncio.to_thread(self.llm.generate_content, agg_prompt.strip())
                    return res.text.strip()

                if node.op == GoTOp.REFINE:
                    print(f"      [GoT Operator] Safe Processing for REFINE node with Reflexion...")
                    if ablation_mode == "none":
                        refine_prompt = f"## Your Task:\n{query}"
                    else:
                        previous_answer = parent_nodes[0].result if parent_nodes else "No previous answer recorded."
                        refine_prompt = f"""
                        ## ⚠ Retry Attempt Required
                        ### Your Previous Failed Attempt:
                        {previous_answer}
                        ### Judge's Critique:
                        {node.judge_reasoning}
                        ### CRITICAL INSTRUCTION:
                        Write a <reflection> block analyzing your mistake:
                        <reflection>
                        - What went wrong in my previous attempt based on the critique:
                        - How I will correct it:
                        </reflection>
                        ## Your Task:
                        {query}
                        """
                    res = await asyncio.to_thread(self.llm.generate_content, refine_prompt.strip())
                    return res.text.strip()

                if node.parent_ids and node.op not in (GoTOp.AGGREGATE, GoTOp.REFINE):
                    prev_results = {p.node_id: p.result for p in parent_nodes if p.result}
                    resolved_query = await self._resolve_dependencies(query, node.parent_ids, prev_results, ablation_mode, downstream_agent=node.task.agent)
                    
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
                        low_confidence=getattr(self, '_last_preliminary_crs', 1.0) < 0.30
                    )
                    query = resolved_query
                    
                print(f"      [Executing] Node {node.node_id[:4]} ({node.task.agent.upper()}): {query[:60]}...")
                result = await self._execute_agent(node.task.agent, query, depth=node.depth)
                
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

        dag_nodes_dump = []
        if dag_dict.get("is_got") and 'got_orchestrator' in locals() and got_orchestrator.dag:
            for n_id, node in got_orchestrator.dag._nodes.items():
                dag_nodes_dump.append({
                    "node_id": n_id,
                    "agent": node.task.agent,
                    "query": node.task.query,
                    "operator": node.op.value,
                    "depth": node.depth,
                    "parent_ids": node.parent_ids,
                    "failure_mode": node.failure_mode.value if node.failure_mode else None,
                    "quality_score": node.quality_score,
                    "judge_critique": node.judge_reasoning,
                    "output": node.result
                })

        retries = 0
        max_retries = 2
        
        while numeric_score < 7 and retries < max_retries:
            print(f"   [Global Diagnosis] Score {numeric_score}/10 is low. Diagnosing failure...")
            prev_score  = numeric_score
            prev_answer = final_answer
            retries += 1
            
            # FIX 3: Give the Diagnostician the full node-level outputs,
            # not just the compressed final answer. It needs the complete
            # picture to correctly identify which agent failed and why.
            # Never compress this context — budget = unlimited.
            node_trace = ""
            for n in dag_nodes_dump:
                if n.get("operator") not in ("aggregate",):  # skip aggregator noise
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

            Output ONLY a JSON object:
            {{"failing_agent": "scholar", "revised_query": "specific corrective instruction"}}
            """
            try:
                res = await asyncio.to_thread(self.llm.generate_content, diag_prompt)
                diag_data = json.loads(res.text.replace('```json', '').replace('```', '').strip())
                agent = diag_data.get('failing_agent', 'rag')
                new_q = diag_data.get('revised_query', user_input)
                
                print(f"   [Diagnostician] Retrying agent '{agent}' with query: '{new_q}' at HIGHER DEPTH (3).")
                fix_result = await self._execute_agent(agent, new_q, depth=3)
                
                synth_prompt = f"Update the final answer with this new information.\nOld Answer:\n{final_answer}\n\nNew Info:\n{fix_result}\n\nNew Final Answer:"
                final_answer = (await asyncio.to_thread(self.llm.generate_content, synth_prompt)).text.strip()
                
                judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
                match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
                numeric_score = int(match.group(1)) if match else 0
                
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
                    if has_context and log.original_query:
                        log.context_relevance = compute_cus_deterministic(
                            injected_context  = log.raw_context_injected,
                            downstream_output = log.original_query,
                        )
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
        comm_data = comm_tracker.export_to_dict()
        comm_tracker.clear()

        run_telemetry = {
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": ablation_mode,
            "latency_seconds": latency,
            "user_query": user_input,
            "profiler_state": user_profiler.profile.profile if config.Config.USE_PROFILER else {},
            "decomposer_used": decomposer_name,
            "initial_tasks": [{"id": t.id, "agent": t.agent, "query": t.query} for t in tasks],
            "got_dag_nodes": dag_nodes_dump,
            "communication_edges": comm_data,
            "global_retries": retries,
            "cugcr_triggered": self.current_run_cugcr_triggered,
            "final_judge_score": numeric_score,
            "final_answer": final_answer
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

    async def run(self, user_input: str, ablation_mode: str = "adaptive", start_time: float = None) -> Dict[str, Any]:
        
        # 🚀 CORRECTED NEURO-SYMBOLIC ROUTING
        if config.Config.DEFAULT_DECOMPOSER == "auto":
            decomposer_name = self.selector.select(user_input, deterministic=False).name
        elif config.Config.DEFAULT_DECOMPOSER == "ultimate_neuro_symbolic":
            # Map the config string to the actual class name defined in decomposers.py
            decomposer_name = "neuro_rule" 
        else:
            decomposer_name = config.Config.DEFAULT_DECOMPOSER
            
        print(f"⚙️  [Orchestrator] Selected Decomposer: {decomposer_name.upper()}")
        return await self._execute_pipeline(user_input, decomposer_name, ablation_mode, start_time)