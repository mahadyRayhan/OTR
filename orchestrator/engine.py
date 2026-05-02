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

from rag_agent.agent import call_rag_agent
from scholar_agent.agent import call_scholar_agent
from cloud_agent.agent import call_cloud_agent
from code_agent.agent import call_code_agent
from profiler_agent.agent import user_profiler

class OrchestratorEngine:
    def __init__(self):
        self.registry = DecomposerRegistry()
        self.selector = AutoDecomposerSelector(self.registry)
        self.dag_builder = DAGRegistry().get(config.Config.DAG_BUILDER)
        
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

    async def _execute_agent(self, agent_name: str, query: str, depth: int = 1) -> str:
        """Standard Agent Execution"""
        
        agent_registry = config.Config.AGENT_REGISTRY_PROMPT
        strategy_match = re.search(r'--strategy=(\w+)', query.lower())
        strategy = strategy_match.group(1) if strategy_match else "cot"
        clean_query = re.sub(r'--strategy=\w+', '', query).strip()
        
        try:
            if agent_name == "code": return await call_code_agent(clean_query, strategy, depth, agent_registry)
            if agent_name == "cloud": return await call_cloud_agent(clean_query, strategy, depth, agent_registry)
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
                rag_data = await call_rag_agent(clean_query, strategy=strategy, depth=depth, agent_registry=agent_registry)
                return f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data['answer']}"
        except Exception as e:
            return f"Error executing agent: {e}"

    async def _resolve_dependencies(self, query: str, depends_on: List[Any], previous_results: Dict[str, Any], ablation_mode: str = "adaptive") -> str:
        if not depends_on or not self.llm: 
            return query
        
        # Combine the context from parent nodes
        context = "\n".join([f"--- Result ---\n{v}" for k, v in previous_results.items() if any(str(dep) in k for dep in depends_on)])
        
        # PROTOCOL A: Isolated (No Communication)
        if ablation_mode == "none":
            return query 

        # PROTOCOL B: Raw Appending (Blind Passing)
        if ablation_mode == "raw":
            return f"{query}\n\n[UPSTREAM CONTEXT]:\n{context}"

        # PROTOCOL C: Adaptive Injection (The New, Safe LLM Weaver)
        if ablation_mode == "adaptive":
            prompt = f"""
            You are a Context Injector. You must provide the Upstream Context to the Downstream Agent without changing the original instruction.
            
            Upstream Context:
            {context}
            
            Original Downstream Query: {query}
            
            CRITICAL INSTRUCTION:
            Rewrite the query to explicitly include the most important facts from the Upstream Context. 
            DO NOT DELETE the original instruction (e.g., if it says 'write a script', your rewritten query MUST still say 'write a script').
            
            Rewritten Query:"""
            res = await asyncio.to_thread(self.llm.generate_content, prompt.strip())
            return res.text.strip()
            
        # Fallback if an unknown mode is passed
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
                    
                    {style_guideline}
                    """
                    res = await asyncio.to_thread(self.llm.generate_content, agg_prompt.strip())
                    return res.text.strip()

                # REFINE NODE FIX
                if node.op == GoTOp.REFINE:
                    print(f"      [GoT Operator] Safe Processing for REFINE node...")
                    refine_prompt = f"Critique from Judge: {node.judge_reasoning}\n\nImprove the answer based on the critique.\n{query}"
                    res = await asyncio.to_thread(self.llm.generate_content, refine_prompt.strip())
                    return res.text.strip()

                # ROOT & BRANCH NODES (Standard State Injection)
                if node.parent_ids and node.op not in (GoTOp.AGGREGATE, GoTOp.REFINE):
                    prev_results = {f"Task {p.task.id}": p.result for p in parent_nodes if p.result}
                    combined_context = "\n".join(prev_results.values())
                    
                    resolved_query = await self._resolve_dependencies(query, node.parent_ids, prev_results, ablation_mode)
                    
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