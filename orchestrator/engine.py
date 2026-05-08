import asyncio
import json
import os
from datetime import datetime
import os
import re
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Dict, Any, List
import time

import config
from .decomposers import DecomposerRegistry, AutoDecomposerSelector
from .dag_builders import DAGRegistry
from .evaluation.communication_log import comm_tracker
from .evaluation.communication_evaluator import CommunicationEvaluator

# Import the GoT Classes
from .adaptive_got_dag import AdaptiveGoTOrchestrator, GoTNode, DepthBudget, GoTOp

from xtra.agent import call_rag_agent
# from rag_agent.agent import call_rag_agent, KnowledgeBase
from scholar_agent.agent import call_scholar_agent
from cloud_agent.agent import call_cloud_agent
from code_agent.agent import call_code_agent
from profiler_agent.agent import user_profiler

# =========================================================
# 🚀 CUGCR ATOMIZER (Deterministic Context Parsing)
# =========================================================
from dataclasses import dataclass
from enum import Enum
import re

# Add these with your other imports at the top
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
# 🚀 CUGCR DEPENDENCY MANIFESTS
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
        token_budget=1000  # Code needs high context to avoid crashing
    ),
    "cloud": DependencyManifest(
        agent_type="cloud",
        required_atom_types=[AtomType.CODE_BLOCK, AtomType.NAMED_ENTITY],
        token_budget=1000
    ),
    "rag": DependencyManifest(
        agent_type="rag",
        required_atom_types=[AtomType.NAMED_ENTITY],
        token_budget=400   # Theory needs tight boundaries
    ),
    "scholar": DependencyManifest(
        agent_type="scholar",
        required_atom_types=[AtomType.NAMED_ENTITY],
        token_budget=200   # Just names and tools
    )
}

def atomize_output(raw_output: str) -> list[ContextAtom]:
    """Extracts protected code, biological entities, and prose deterministically."""
    atoms = []
    
    # 1. Protected: Extract Code Blocks
    for match in re.finditer(r'```[\s\S]*?```', raw_output):
        atoms.append(ContextAtom(AtomType.CODE_BLOCK, match.group(), is_protected=True))
    
    # Remove code for further processing
    prose_only = re.sub(r'```[\s\S]*?```', '', raw_output)
    
    # 2. Protected: Extract basic bio-entities/acronyms (e.g., BRCA1, GATK)
    bio_pattern = r'\b([A-Z]{2,}[0-9]*|[A-Z][a-z]+-\d+)\b'
    for match in re.finditer(bio_pattern, prose_only):
        atoms.append(ContextAtom(AtomType.NAMED_ENTITY, match.group(), is_protected=True))
        
    # 3. Unprotected: Extract prose sentences
    for sentence in prose_only.split('. '):
        if len(sentence.strip()) > 15:
            atoms.append(ContextAtom(AtomType.PROSE_CLAIM, sentence.strip() + ".", is_protected=False))
            
    return atoms

def build_rigid_prompt(original_query: str, raw_context: str, token_budget: int = 400) -> str:
    """Filters atoms and injects them into a rigid, drift-proof template."""
    atoms = atomize_output(raw_context)
    
    # Separate protected vs unprotected
    code_blocks = [a.content for a in atoms if a.type == AtomType.CODE_BLOCK]
    entities = list(set([a.content for a in atoms if a.type == AtomType.NAMED_ENTITY]))
    prose = [a.content for a in atoms if a.type == AtomType.PROSE_CLAIM]
    
    # Budgeting: Always keep code and entities. Limit prose to save tokens.
    context_block = ""
    if code_blocks:
        context_block += "### Upstream Code Artifacts\n" + "\n".join(code_blocks) + "\n\n"
    if entities:
        context_block += "### Key Technical Entities\n" + ", ".join(entities[:10]) + "\n\n"
    if prose:
        # Only take the first few sentences of prose to stay under budget
        context_block += "### Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose[:4]) + "\n\n"
        
    # Rigid Template: Instruction is ALWAYS at the bottom!
    return f"""{context_block}---
## Your Task
{original_query}
"""
# =========================================================

class OrchestratorEngine:
    def __init__(self):
        self.registry = DecomposerRegistry()
        self.selector = AutoDecomposerSelector(self.registry)
        self.dag_builder = DAGRegistry().get(config.Config.DAG_BUILDER)
        
        # 🚀 NEW: Load the local embedding model if HEAVY mode is active
        self.embedder = None
        if getattr(config.Config, "CONTEXT_ROUTING_MODE", "light") == "heavy":
            print("⏳ Loading local semantic embedding model for Heavy Context Routing...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Generator Model (Gemini)
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "your_google_api_key_here":
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(config.Config.GENERATION_MODEL)
        else:
            self.llm = None
            
        # Judge Model (OpenAI)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "sk-your-openai-key-here":
            self.openai_client = AsyncOpenAI(api_key=openai_key)
        else:
            self.openai_client = None

    # 🚀 CUGCR Semantic Math Algorithm (With Manifests)
    def _score_and_select_atoms(self, atoms: list, downstream_query: str, downstream_agent: str, budget_multiplier: float = 1.0) -> list:
        # Load the manifest for the specific downstream agent
        manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
        current_budget = int(manifest.token_budget * budget_multiplier)

        selected = []
        used_tokens = 0
        
        # 1. Hard requirements: Always keep protected atoms required by this agent
        for atom in atoms:
            if atom.is_protected and atom.type in manifest.required_atom_types:
                selected.append(atom)
                used_tokens += len(atom.content.split())
        
        unprotected = [a for a in atoms if a not in selected]
        if not unprotected or not self.embedder:
            return selected
            
        # 2. Dense Embeddings (Conceptual similarity)
        query_emb = self.embedder.encode([downstream_query])
        atom_docs = [a.content for a in unprotected]
        atom_embs = self.embedder.encode(atom_docs)
        dense_scores = cosine_similarity(query_emb, atom_embs)[0]
        
        # 3. Sparse Embeddings (Exact keyword overlap via TF-IDF)
        vectorizer = TfidfVectorizer().fit([downstream_query] + atom_docs)
        tfidf_matrix = vectorizer.transform(atom_docs)
        query_tfidf = vectorizer.transform([downstream_query])
        sparse_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        
        # 4. Fusion Math: Combine scores (60% Dense, 40% Sparse)
        final_scores = (0.6 * dense_scores) + (0.4 * sparse_scores)
        
        # 5. Greedy Budget Selection
        scored_unprotected = sorted(zip(unprotected, final_scores), key=lambda x: x[1], reverse=True)
        for atom, score in scored_unprotected:
            cost = len(atom.content.split())
            if used_tokens + cost <= current_budget:
                selected.append(atom)
                used_tokens += cost
                
        return selected

    async def _execute_agent(self, agent_name: str, query: str, depth: int = 1) -> str:
        """Standard Agent Execution"""
        
        agent_registry = config.Config.AGENT_REGISTRY_PROMPT
        strategy_match = re.search(r'--strategy=(\w+)', query.lower())
        strategy = strategy_match.group(1) if strategy_match else "cot"
        clean_query = re.sub(r'--strategy=\w+', '', query).strip()
        
        try:
            if agent_name == "code": return await call_code_agent(clean_query, strategy, depth, agent_registry)
            if agent_name == "cloud":
                # Get the profile constraint string we generated earlier in the function
                profile_str = user_profiler.get_style_guideline(use_session=False) if config.Config.USE_PROFILER else ""
                
                return await call_cloud_agent(
                    user_input=clean_query, 
                    strategy=strategy, 
                    depth=depth, 
                    agent_registry=agent_registry,
                    user_profile_str=profile_str # <--- ADD THIS!
                )
            if agent_name == "scholar":
                limit = 1
                # Normalize spelled-out numbers 1-5 to digits so the regex catches them
                text_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
                query_normalized = clean_query.lower()
                for word, digit in text_to_num.items():
                    query_normalized = re.sub(rf'\b{word}\b', digit, query_normalized)
                    
                match = re.search(r'\b([1-5])\b', query_normalized)
                if match: limit = int(match.group(1))
                return await call_scholar_agent(clean_query, is_collaboration=False, limit=limit, strategy=strategy, depth=depth, agent_registry=agent_registry)
            else: # RAG
                rag_data = await call_rag_agent(user_input=clean_query, strategy=strategy, depth=depth, agent_registry=agent_registry)
                return f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data['answer']}"
        except Exception as e:
            return f"Error executing agent: {e}"

    # async def _resolve_dependencies(self, query: str, depends_on: List[Any], previous_results: Dict[str, Any], ablation_mode: str = "adaptive", downstream_agent: str = "rag") -> str:
    #     if not depends_on or not self.llm: 
    #         return query
        
    #     context = "\n".join([f"--- Result ---\n{v}" for k, v in previous_results.items() if any(str(dep) in k for dep in depends_on)])
        
    #     if ablation_mode == "none": return query 
    #     if ablation_mode == "raw": return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"

    #     if ablation_mode == "adaptive":
    #         atoms = atomize_output(context)
    #         if getattr(config.Config, "CONTEXT_ROUTING_MODE", "light") == "heavy" and self.embedder:
                
    #             # 🚀 Pass the downstream agent type to apply the correct manifest!
    #             selected_atoms = self._score_and_select_atoms(atoms, query, downstream_agent)
                
    #             code_blocks = [a.content for a in selected_atoms if a.type == AtomType.CODE_BLOCK]
    #             entities = list(set([a.content for a in selected_atoms if a.type == AtomType.NAMED_ENTITY]))
    #             prose = [a.content for a in selected_atoms if a.type == AtomType.PROSE_CLAIM]
                
    #             context_block = ""
    #             if code_blocks: context_block += "### Upstream Code Artifacts\n" + "\n".join(code_blocks) + "\n\n"
    #             if entities: context_block += "### Key Technical Entities\n" + ", ".join(entities[:10]) + "\n\n"
    #             if prose: context_block += "### Relevant Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose) + "\n\n"
                
    #             return f"{context_block}---\n## Your Task\n{query}\n"
    #         else:
    #             return build_rigid_prompt(original_query=query, raw_context=context)
            
    #     return query

    async def _resolve_dependencies(self, query: str, depends_on: List[Any], previous_results: Dict[str, Any], ablation_mode: str = "adaptive", downstream_agent: str = "rag") -> str:
        if not depends_on or not self.llm: 
            return query
        
        context = "\n".join([f"--- Result ---\n{v}" for k, v in previous_results.items() if any(str(dep) in k for dep in depends_on)])
        
        if ablation_mode == "none": return query 
        if ablation_mode == "raw": return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"

        if ablation_mode == "adaptive":
            # 🚀 NEW: The Dynamic Routing Gateway
            # 1. Look up the downstream agent's safe token budget
            # from .adaptive_got_dag import AGENT_MANIFESTS # Ensure this is imported or defined in engine.py
            manifest = AGENT_MANIFESTS.get(downstream_agent, AGENT_MANIFESTS["rag"])
            budget_limit = manifest.token_budget
            
            # 2. Estimate upstream token count (approx 4 chars per token)
            upstream_token_count = len(context) // 4
            
            # 3. THE BYPASS VALVE: If context is small, use RAW to preserve prose!
            if upstream_token_count <= budget_limit:
                print(f"      [Gateway] Context ({upstream_token_count} tok) fits in {downstream_agent.upper()} budget ({budget_limit} tok). Bypassing CUGCR -> RAW Passing.")
                return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"
                
            # 4. CUGCR EMERGENCY VALVE: If context is too big, trigger the Heavy Math!
            print(f"      [Gateway] Context ({upstream_token_count} tok) EXCEEDS {downstream_agent.upper()} budget ({budget_limit} tok). Triggering CUGCR Compression.")
            atoms = atomize_output(context)
            if getattr(config.Config, "CONTEXT_ROUTING_MODE", "light") == "heavy" and self.embedder:
                
                selected_atoms = self._score_and_select_atoms(atoms, query, downstream_agent)
                
                code_blocks = [a.content for a in selected_atoms if a.type == AtomType.CODE_BLOCK]
                entities = list(set([a.content for a in selected_atoms if a.type == AtomType.NAMED_ENTITY]))
                prose = [a.content for a in selected_atoms if a.type == AtomType.PROSE_CLAIM]
                
                context_block = ""
                if code_blocks: context_block += "### Upstream Code Artifacts\n" + "\n".join(code_blocks) + "\n\n"
                if entities: context_block += "### Key Technical Entities\n" + ", ".join(entities[:10]) + "\n\n"
                if prose: context_block += "### Relevant Supporting Evidence\n" + "\n".join(f"- {c}" for c in prose) + "\n\n"
                
                return f"{context_block}---\n## Your Task\n{query}\n"
            else:
                return build_rigid_prompt(original_query=query, raw_context=context)
            
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
        - CRITICAL DEPENDENCY CHECK: If the query asks the system to use specific information (e.g., "the experts you found", "based on those differences"), but the answer looks like generic LLM hallucinations or ignores the specific dependencies, SCORE IT A MAXIMUM OF 4/10.
        - Ignore 'mock agent' warnings. Score based on routing accuracy.
        
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

    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------
    async def _execute_pipeline(self, user_input: str, decomposer_name: str, ablation_mode: str = "adaptive", start_time: float = None) -> Dict[str, Any]:
        
        # 1. Start Profiler
        profiler_task = None
        if config.Config.USE_PROFILER:
            profiler_task = asyncio.create_task(user_profiler.update_profile(user_input, self.llm))

        # 2. Decompose
        decomposer = self.registry.get(decomposer_name)
        tasks = await decomposer.decompose(user_input, self.llm)
        
        print(f"\n   [Plan for {decomposer_name.upper()}]:")
        for t in tasks:
            dep_str = f" | Depends on: {t.depends_on}" if t.depends_on else ""
            print(f"      Task {t.id} -> [{t.agent.upper()}]: '{t.query}'{dep_str}")

        dag_dict = self.dag_builder.build(tasks)
        results = {}
        
        # 🚀 ADVANCED PATH: Graph-of-Thoughts DAG
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
                
                # Fetch ACTUAL parent node objects from the live DAG
                parent_nodes = [got_orchestrator.dag._nodes[pid] for pid in node.parent_ids] if got_orchestrator.dag else[]

                # AGGREGATE NODE FIX
                if node.op == GoTOp.AGGREGATE:
                    print(f"      [GoT Operator] Safe Processing for AGGREGATE node...")
                    full_context = "\n\n".join([f"--- Source ---\n{p.result}" for p in parent_nodes if p.result])
                    agg_prompt = f"""
                    Synthesize a final response to the user's original query using ONLY these outputs:
                    
                    User's Original Query: "{user_input}"
                    
                    Agent Outputs:
                    {full_context}
                    
                    CRITICAL INSTRUCTIONS: 
                    1. Do NOT summarize or truncate any Python or Bash code blocks. Copy them verbatim.
                    2. If any Agent Output contains an "⚠️ AI EXTRAPOLATION WARNING" or "⚠️ AI GENERATED CODE WARNING", you MUST include that warning verbatim in your final synthesized response so the user is aware.
                    
                    {style_guideline}
                    """
                    res = await asyncio.to_thread(self.llm.generate_content, agg_prompt.strip())
                    return res.text.strip()

                # REFINE NODE FIX
                if node.op == GoTOp.REFINE:
                    print(f"      [GoT Operator] Safe Processing for REFINE node with Reflexion...")
                    
                    # Fetch the previous bad answer from the parent node
                    previous_answer = parent_nodes[0].result if parent_nodes else "No previous answer recorded."
                    
                    refine_prompt = f"""
                    ## ⚠ Retry Attempt Required
                    
                    ### Your Previous Failed Attempt:
                    {previous_answer}

                    ### Judge's Critique:
                    {node.judge_reasoning}

                    ### CRITICAL INSTRUCTION:
                    Before writing your final answer, you MUST write a <reflection> block analyzing your mistake:
                    <reflection>
                    - What went wrong in my previous attempt based on the critique:
                    - How I will correct it:
                    </reflection>
                    
                    ## Your Task:
                    {query}
                    """
                    res = await asyncio.to_thread(self.llm.generate_content, refine_prompt.strip())
                    return res.text.strip()

                # ROOT & BRANCH NODES (Standard State Injection)
                if node.parent_ids and node.op not in (GoTOp.AGGREGATE, GoTOp.REFINE):
                    prev_results = {f"Task {p.task.id}": p.result for p in parent_nodes if p.result}
                    combined_context = "\n".join(prev_results.values())
                    
                    resolved_query = await self._resolve_dependencies(query, node.parent_ids, prev_results, ablation_mode, downstream_agent=node.task.agent)
                    
                    # Log the injection event
                    current_wave = len(got_orchestrator.dag.wave_scores) + 1 if got_orchestrator.dag else 1
                    comm_tracker.log_injection(
                        wave=current_wave,
                        from_ids=node.parent_ids,
                        to_id=node.node_id, 
                        to_agent=node.task.agent,
                        context=combined_context,
                        orig_query=query,
                        resolved_query=resolved_query
                    )
                    
                    query = resolved_query
                    
                print(f"      [Executing] Node {node.node_id[:4]} ({node.task.agent.upper()}): {query[:60]}...")
                
                # Execute the agent
                result = await self._execute_agent(node.task.agent, query, depth=node.depth)
                
                # Update the tracker with the raw output generated by the agent
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
            
        # 🚀 STANDARD PATH: Classic Topological DAG
        else:
            layers = dag_dict["execution_layers"]
            for layer in layers:
                resolved_queries =[]
                for t in layer:
                    if t.depends_on:
                        resolved_queries.append(await self._resolve_dependencies(t.query, t.depends_on, results, ablation_mode))
                    else:
                        resolved_queries.append(t.query)
                
                layer_results = await asyncio.gather(*(self._execute_agent(t.agent, rq) for t, rq in zip(layer, resolved_queries)))
                for t, res in zip(layer, layer_results):
                    results[f"Task {t.id} ({t.agent})"] = res
                    
            if profiler_task: await profiler_task
            style_guideline = user_profiler.get_style_guideline(use_session=False) if config.Config.USE_PROFILER else ""
            
            context = "\n\n".join([f"--- {k} ---\n{v}" for k, v in results.items()])
            synth_prompt = f"Synthesize a final response to '{user_input}' using ONLY these agent outputs:\n{context}\n\n{style_guideline}"
            
            print("🧠 [Orchestrator] Synthesizing Final Answer...")
            final_answer = (await asyncio.to_thread(self.llm.generate_content, synth_prompt)).text.strip()

        if config.Config.USE_PROFILER:
            print(f"   [Profiler State] {user_profiler.profile.summary()}")

        print("⚖️  [Orchestrator] Final Global Evaluation...")
        judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
        match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
        numeric_score = int(match.group(1)) if match else 0

        # 🚀 GLOBAL DIAGNOSTICS & RETRY LOOP
        retries = 0
        max_retries = 2
        
        while numeric_score < 8 and retries < max_retries:
            print(f"   [Global Diagnosis] Score {numeric_score}/10 is low. Diagnosing failure...")
            prev_score = numeric_score
            prev_answer = final_answer
            retries += 1
            diag_prompt = f"""
            The final answer failed to satisfy the original query.
            Original Query: "{user_input}"
            Final Answer: "{final_answer}"
            Judge Score: {judge_score_raw}
            
            Based on the Judge's critique, which specific agent failed or what information is missing?
            Available agents: {config.Config.AGENT_REGISTRY_PROMPT}
            Output ONLY a JSON object:
            {{"failing_agent": "scholar", "revised_query": "specific context or question to fix the failure"}}
            
            IMPORTANT: If the failing agent is 'scholar', the retry query must ask for the FULL number of experts originally requested, not just the missing ones.
            """
            try:
                res = await asyncio.to_thread(self.llm.generate_content, diag_prompt)
                diag_data = json.loads(res.text.replace('```json', '').replace('```', '').strip())
                agent = diag_data.get('failing_agent', 'rag')
                new_q = diag_data.get('revised_query', user_input)
                
                print(f"   [Diagnostician] Retrying agent '{agent}' with query: '{new_q}' at HIGHER DEPTH (3).")
                fix_result = await self._execute_agent(agent, new_q, depth=3)
                
                synth_prompt = f"Update the final answer with this new information.\nCRITICAL: Do NOT summarize the new info. If the new info contains code blocks, you MUST copy the exact code blocks into your new answer.\n\nOld Answer:\n{final_answer}\n\nNew Info:\n{fix_result}\n\nNew Final Answer:"
                final_answer = (await asyncio.to_thread(self.llm.generate_content, synth_prompt)).text.strip()
                
                print("⚖️  [Orchestrator] Re-evaluating Global Answer...")
                judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
                match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
                numeric_score = int(match.group(1)) if match else 0
                
                # FIX 1: Stop re-run if score drops
                if numeric_score <= prev_score:
                    print(f"   [Global Diagnosis] Score did not improve ({prev_score} -> {numeric_score}). Halting retries.")
                    final_answer = prev_answer
                    numeric_score = prev_score
                    break
                    
            except Exception as e:
                print(f"[Diagnostics Error] Global retry failed: {e}")
                break

        # =========================================================
        # 🚀 MASTER TELEMETRY & EVALUATION BLOCK
        # =========================================================
        from .evaluation.communication_evaluator import CommunicationEvaluator

        # 1. Calculate Latency
        latency = round(time.time() - start_time, 2) if start_time else 0.0

        # 2. Evaluate Communication Edges (Relevance & Utilization)
        if comm_tracker.logs:
            try:
                evaluator = CommunicationEvaluator(self.llm)
                await evaluator.evaluate_logs(comm_tracker.logs)
                
                print("\n   [Comm Summary] Context Injection Performance:")
                for log in comm_tracker.logs:
                    status = "✅" if log.dependency_accuracy else "❌ (Placeholder Leak)"
                    # Note: We now print Relevance and Utilization (Efficiency is saved for offline math)
                    rel = log.context_relevance * 100 if log.context_relevance else 0
                    util = log.utilization_score * 100 if log.utilization_score else 0
                    print(f"      -> To [{log.to_agent.upper()}]: Relevance {rel:.0f}% | Utilized {util:.0f}% {status}")
            except Exception as e:
                print(f"   [Comm Evaluator Error] {e}")

        # Export and clear logs
        comm_data = comm_tracker.export_to_dict()
        comm_tracker.clear()

        # 3. Extract the full DAG state
        dag_nodes_dump =[]
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
                    "judge_critique": node.judge_reasoning
                })

        # 4. Build Telemetry Payload
        run_telemetry = {
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": ablation_mode,
            "latency_seconds": latency,
            "user_query": user_input,
            "profiler_state": user_profiler.profile.profile if config.Config.USE_PROFILER else {},
            "decomposer_used": decomposer_name,
            "initial_tasks":[{"id": t.id, "agent": t.agent, "query": t.query} for t in tasks],
            "got_dag_nodes": dag_nodes_dump,
            "communication_edges": comm_data,
            "global_retries": retries,
            "final_judge_score": numeric_score,
            "final_answer": final_answer
        }

        # 5. Read, Append, and Save safely
        telemetry_file = "experiment_telemetry.json"
        all_data = []
        if os.path.exists(telemetry_file):
            try:
                with open(telemetry_file, "r") as f:
                    all_data = json.load(f)
            except json.JSONDecodeError:
                pass # File is empty or corrupted, start fresh

        all_data.append(run_telemetry)

        with open(telemetry_file, "w") as f:
            json.dump(all_data, f, indent=4)

        print(f"\n💾 Saved complete run telemetry to '{telemetry_file}'")

        return {
            "final_answer": final_answer, 
            "judge_score": judge_score_raw, 
            "numeric_score": numeric_score,
            "communication_metrics": comm_data  # <--- Exposes the data!
        }

    async def run(self, user_input: str, ablation_mode: str = "adaptive", start_time: float = None) -> Dict[str, Any]:
        if config.Config.DEFAULT_DECOMPOSER == "auto":
            decomposer_name = self.selector.select(user_input, deterministic=False).name
        else:
            decomposer_name = config.Config.DEFAULT_DECOMPOSER
            
        print(f"⚙️  [Orchestrator] Selected Decomposer: {decomposer_name.upper()}")
        return await self._execute_pipeline(user_input, decomposer_name, ablation_mode, start_time)

    async def run_benchmark(self, questions: List[str]):
        decomposers = ["plan_and_solve", "llm_compiler", "qdmr_rule", "coverage", "ultimate_neuro_symbolic"]
        scores = {d: [] for d in decomposers}

        print("\n" + "="*60)
        print(f"🏆 INITIATING BENCHMARK (JUDGE: {config.Config.JUDGE_MODEL})")
        print("="*60)

        for d in decomposers:
            print(f"\n=======================================")
            print(f"🚀 TESTING ALGORITHM: {d.upper()}")
            print(f"=======================================")

            if config.Config.USE_PROFILER:
                user_profiler.reset_all()

            for i, q in enumerate(questions, 1):
                print(f"\n   Turn {i}/{len(questions)}: '{q}'")
                try:
                    output = await self._execute_pipeline(q, d)
                    scores[d].append(output["numeric_score"])
                    print(f"   [Judge Score]: {output['numeric_score']}/10")
                except Exception as e:
                    print(f"   [Error] {e}")
                    scores[d].append(0)

        print("\n" + "="*60)
        print("📊 FINAL LEADERBOARD")
        print("="*60)
        avg_scores = {d: sum(s)/len(s) if s else 0 for d, s in scores.items()}
        ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (name, avg) in enumerate(ranked, 1):
            print(f"{rank}. {name.upper().ljust(25)} : {avg:.1f} / 10")
        print("="*60)