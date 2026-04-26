# import asyncio
# import os
# import re
# import google.generativeai as genai
# from openai import AsyncOpenAI
# from typing import Dict, Any, List

# import config
# from .decomposers import DecomposerRegistry, AutoDecomposerSelector
# from .dag_builders import DAGRegistry

# from rag_agent.agent import call_rag_agent
# from scholar_agent.agent import call_scholar_agent
# from cloud_agent.agent import call_cloud_agent
# from code_agent.agent import call_code_agent
# from profiler_agent.agent import user_profiler

# class OrchestratorEngine:
#     def __init__(self):
#         self.registry = DecomposerRegistry()
#         self.selector = AutoDecomposerSelector(self.registry)
#         self.dag_builder = DAGRegistry().get(config.Config.DAG_BUILDER)
        
#         # Generator Model (Gemini)
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if api_key and api_key != "your_google_api_key_here":
#             genai.configure(api_key=api_key)
#             self.llm = genai.GenerativeModel(config.Config.GENERATION_MODEL)
#         else:
#             self.llm = None
            
#         # Judge Model (OpenAI)
#         openai_key = os.getenv("OPENAI_API_KEY")
#         if openai_key and openai_key != "sk-your-openai-key-here":
#             self.openai_client = AsyncOpenAI(api_key=openai_key)
#         else:
#             self.openai_client = None

#     async def _execute_agent(self, agent_name: str, query: str) -> str:
#         try:
#             if agent_name == "code": return await call_code_agent(query)
#             if agent_name == "cloud": return call_cloud_agent(query)
#             if agent_name == "scholar":
#                 limit = 1
#                 match = re.search(r'\b([1-5])\b', query)
#                 if match: limit = int(match.group(1))
#                 return await call_scholar_agent(query, is_collaboration=False, limit=limit)
#             else: # RAG
#                 strategy_match = re.search(r'--strategy=(\w+)', query.lower())
#                 rag_strategy = strategy_match.group(1) if strategy_match else "tot"
#                 clean_query = re.sub(r'--strategy=\w+', '', query).strip()
#                 rag_data = await call_rag_agent(clean_query, strategy=rag_strategy)
#                 return f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data['answer']}"
#         except Exception as e:
#             return f"Error: {e}"

#     async def _resolve_dependencies(self, query: str, depends_on: List[int], previous_results: Dict[str, Any]) -> str:
#         if not depends_on or not self.llm: return query
#         context = ""
#         for dep_id in depends_on:
#             for k, v in previous_results.items():
#                 if f"Task {dep_id}" in k:
#                     context += f"--- Result from {k} ---\n{v}\n\n"
#         prompt = f"You are a State Injector. Update the Original Query by replacing any placeholders with the actual specific name/information found in the Context. If it doesn't need updating, return it exactly as is. ONLY output the rewritten query.\nContext:\n{context}\nOriginal Query: {query}\nRewritten Query:"
#         res = await asyncio.to_thread(self.llm.generate_content, prompt.strip())
#         return res.text.strip()

#     async def _gpt_as_judge(self, user_input: str, final_answer: str) -> str:
#         if not self.openai_client:
#             return "Score: 0/10 - OpenAI API Key missing for Judge."
#         judge_prompt = f"""
#         You are an impartial AI Judge evaluating a multi-agent system.
#         User Query: "{user_input}"
#         System Answer: "{final_answer}"
        
#         CRITERIA (0 to 10):
#         - Did it perform ALL tasks? (Deduct 3 points for missing tasks).
#         - If asked for code, is there code? If asked for experts, are there real names?
#         - Ignore 'mock agent' warnings. Score based on routing accuracy.
#         Format exactly as: 'Score: X/10 - [Reason]'
#         """
#         try:
#             response = await self.openai_client.chat.completions.create(
#                 model=config.Config.JUDGE_MODEL,
#                 messages=[{"role": "user", "content": judge_prompt}],
#                 temperature=0.0
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             return f"Score: 0/10 - Judge Failed: {e}"

#     async def _execute_pipeline(self, user_input: str, decomposer_name: str) -> Dict[str, Any]:
        
#         # 1. OPTIONAL: Start Profiler
#         profiler_task = None
#         style_guideline = ""
#         if config.Config.USE_PROFILER:
#             profiler_task = asyncio.create_task(user_profiler.update_profile(user_input, self.llm))

#         # 2. Decompose and Build DAG
#         decomposer = self.registry.get(decomposer_name)
#         tasks = await decomposer.decompose(user_input, self.llm)
        
#         print(f"\n   [Plan for {decomposer_name.upper()}]:")
#         for t in tasks:
#             dep_str = f" | Depends on: Task(s) {t.depends_on}" if t.depends_on else ""
#             print(f"      Task {t.id} -> [{t.agent.upper()}]: '{t.query}'{dep_str}")

#         dag = self.dag_builder.build(tasks)
#         layers = dag["execution_layers"]
        
#         # 3. Execute Layers
#         results = {}
#         for layer in layers:
#             resolved_queries = []
#             for t in layer:
#                 if t.depends_on:
#                     resolved_queries.append(await self._resolve_dependencies(t.query, t.depends_on, results))
#                 else:
#                     resolved_queries.append(t.query)
            
#             layer_results = await asyncio.gather(*(self._execute_agent(t.agent, rq) for t, rq in zip(layer, resolved_queries)))
#             for t, res in zip(layer, layer_results):
#                 results[f"Task {t.id} ({t.agent})"] = res
                
#         # 4. OPTIONAL: Wait for Profiler
#         if config.Config.USE_PROFILER and profiler_task:
#             await profiler_task
#             style_guideline = user_profiler.get_style_guideline(use_session=False)
#             print(f"   [Profiler State] {user_profiler.profile.summary()}")
        
#         # 5. Synthesize & Judge
#         context = "\n\n".join([f"--- {k} ---\n{v}" for k, v in results.items()])
#         synth_prompt = f"Synthesize a final response to '{user_input}' using ONLY these agent outputs:\n{context}\n\n{style_guideline}"
#         final_answer = (await asyncio.to_thread(self.llm.generate_content, synth_prompt)).text.strip()
        
#         judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
#         match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
#         numeric_score = int(match.group(1)) if match else 0

#         return {"final_answer": final_answer, "judge_score": judge_score_raw, "numeric_score": numeric_score}

#     async def run(self, user_input: str) -> Dict[str, Any]:
#         if config.Config.DEFAULT_DECOMPOSER == "auto":
#             decomposer_name = self.selector.select(user_input, deterministic=False).name
#         else:
#             decomposer_name = config.Config.DEFAULT_DECOMPOSER
            
#         print(f"⚙️  [Orchestrator] Selected Decomposer: {decomposer_name.upper()}")
#         return await self._execute_pipeline(user_input, decomposer_name)

#     async def run_benchmark(self, questions: List[str]):
#         decomposers = ["plan_and_solve", "llm_compiler", "qdmr_rule", "coverage", "ultimate_neuro_symbolic"]
#         scores = {d: [] for d in decomposers}

#         print("\n" + "="*60)
#         print(f"🏆 INITIATING BENCHMARK (JUDGE: {config.Config.JUDGE_MODEL})")
#         print("="*60)

#         for d in decomposers:
#             print(f"\n=======================================")
#             print(f"🚀 TESTING ALGORITHM: {d.upper()}")
#             print(f"=======================================")

#             for i, q in enumerate(questions, 1):
#                 print(f"\n   Turn {i}/{len(questions)}: '{q}'")
#                 try:
#                     output = await self._execute_pipeline(q, d)
#                     scores[d].append(output["numeric_score"])
#                     print(f"   [Judge Score]: {output['numeric_score']}/10")
#                 except Exception as e:
#                     print(f"   [Error] {e}")
#                     scores[d].append(0)

#         print("\n" + "="*60)
#         print("📊 FINAL LEADERBOARD")
#         print("="*60)
#         avg_scores = {d: sum(s)/len(s) if s else 0 for d, s in scores.items()}
#         ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
#         for rank, (name, avg) in enumerate(ranked, 1):
#             print(f"{rank}. {name.upper().ljust(25)} : {avg:.1f} / 10")
#         print("="*60)

# import asyncio
# import os
# import re
# import google.generativeai as genai
# from openai import AsyncOpenAI
# from typing import Dict, Any, List

# import config
# from .decomposers import DecomposerRegistry, AutoDecomposerSelector

# # 🚀 IMPORT THE NEW ADAPTIVE GoT DAG!
# from .dag_builders import AdaptiveGoTOrchestrator, GoTNode, DepthBudget

# from rag_agent.agent import call_rag_agent
# from scholar_agent.agent import call_scholar_agent
# from cloud_agent.agent import call_cloud_agent
# from code_agent.agent import call_code_agent
# from profiler_agent.agent import user_profiler

# class OrchestratorEngine:
#     def __init__(self):
#         self.registry = DecomposerRegistry()
#         self.selector = AutoDecomposerSelector(self.registry)
        
#         # Generator Model (Gemini)
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if api_key and api_key != "your_google_api_key_here":
#             genai.configure(api_key=api_key)
#             self.llm = genai.GenerativeModel(config.Config.GENERATION_MODEL)
#         else:
#             self.llm = None
            
#         # Judge Model (OpenAI)
#         openai_key = os.getenv("OPENAI_API_KEY")
#         if openai_key and openai_key != "sk-your-openai-key-here":
#             self.openai_client = AsyncOpenAI(api_key=openai_key)
#         else:
#             self.openai_client = None

#     # ---------------------------------------------------------
#     # REAL AGENT EXECUTOR (Passed to the GoT DAG)
#     # ---------------------------------------------------------
#     async def _execute_agent(self, node: GoTNode) -> str:
#         """
#         The GoT DAG calls this function whenever a node is ready to execute.
#         We route the GoTNode's query to the actual physical agent.
#         """
#         agent_name = node.task.agent
#         query = node.task.query
        
#         # 1. Check if the query has placeholders that need resolving based on parent nodes
#         if node.parent_ids:
#             query = await self._resolve_dependencies(query, node.parent_ids, node)

#         print(f"      [Executing] Node {node.node_id[:4]} ({agent_name.upper()}): {query[:60]}...")
        
#         try:
#             if agent_name == "code": return await call_code_agent(query)
#             if agent_name == "cloud": return call_cloud_agent(query)
#             if agent_name == "scholar":
#                 limit = 1
#                 match = re.search(r'\b([1-5])\b', query)
#                 if match: limit = int(match.group(1))
#                 return await call_scholar_agent(query, is_collaboration=False, limit=limit)
#             else: # RAG
#                 strategy_match = re.search(r'--strategy=(\w+)', query.lower())
#                 rag_strategy = strategy_match.group(1) if strategy_match else "tot"
#                 clean_query = re.sub(r'--strategy=\w+', '', query).strip()
#                 rag_data = await call_rag_agent(clean_query, strategy=rag_strategy)
#                 return f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data['answer']}"
#         except Exception as e:
#             return f"Error executing agent: {e}"

#     async def _resolve_dependencies(self, query: str, parent_ids: List[str], current_node: GoTNode) -> str:
#         """Dynamically rewrites the query using the results of parent nodes."""
#         if not self.llm: return query
        
#         # We must fetch the actual result strings from the parent nodes in the DAG
#         # Note: We assume the DAG object has populated the results of parents by now.
#         context = ""
#         # (This is a simplified injection. In a full system, the DAG passes the parent context directly)
        
#         prompt = f"Update the Original Query by replacing any placeholders with info from the Context. ONLY output the rewritten query.\nContext:\n{context}\nOriginal Query: {query}\nRewritten Query:"
#         res = await asyncio.to_thread(self.llm.generate_content, prompt.strip())
#         return res.text.strip()

#     # ---------------------------------------------------------
#     # EVALUATION
#     # ---------------------------------------------------------
#     async def _gpt_as_judge(self, user_input: str, final_answer: str) -> str:
#         if not self.openai_client:
#             return "Score: 0/10 - OpenAI API Key missing for Judge."
#         judge_prompt = f"""
#         You are an impartial AI Judge evaluating a multi-agent system.
#         User Query: "{user_input}"
#         System Answer: "{final_answer}"
        
#         CRITERIA (0 to 10):
#         - Did it perform ALL tasks? (Deduct 3 points for missing tasks).
#         - If asked for code, is there code? If asked for experts, are there real names?
#         - Ignore 'mock agent' warnings. Score based on routing accuracy.
#         Format exactly as: 'Score: X/10 - [Reason]'
#         """
#         try:
#             response = await self.openai_client.chat.completions.create(
#                 model=config.Config.JUDGE_MODEL,
#                 messages=[{"role": "user", "content": judge_prompt}],
#                 temperature=0.0
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             return f"Score: 0/10 - Judge Failed: {e}"

#     # ---------------------------------------------------------
#     # MAIN PIPELINE
#     # ---------------------------------------------------------
#     async def _execute_pipeline(self, user_input: str, decomposer_name: str) -> Dict[str, Any]:
        
#         # 1. Run Profiler (Async)
#         profiler_task = None
#         if config.Config.USE_PROFILER:
#             profiler_task = asyncio.create_task(user_profiler.update_profile(user_input, self.llm))

#         # 2. Decompose
#         decomposer = self.registry.get(decomposer_name)
#         tasks = await decomposer.decompose(user_input, self.llm)
        
#         print(f"\n   [Decomposed Tasks ({decomposer_name.upper()})]:")
#         for t in tasks:
#             dep_str = f" | Depends on: {t.depends_on}" if t.depends_on else ""
#             print(f"      Task {t.id} -> [{t.agent.upper()}]: '{t.query}'{dep_str}")

#         # 3. Wait for Profiler to get the user's mindstate
#         if config.Config.USE_PROFILER and profiler_task:
#             await profiler_task
        
#         # 4. 🚀 INITIALIZE THE ADAPTIVE GoT DAG ORCHESTRATOR
#         # It takes the Profile weights and automatically calculates depth budgets!
#         got_orchestrator = AdaptiveGoTOrchestrator(
#             quality_threshold=0.80, # Agents must score an 8/10 to pass
#             beam_width=2,           # Max 2 failed nodes expand per wave
#             max_expansions=6,       # Hard cap to save money
#             depth_budget=DepthBudget(base_depth=1, weight_per_level=20, max_depth=3)
#         )

#         print("\n   [GoT DAG] Commencing Adaptive Wave Execution...")
        
#         # 5. Run the GoT execution loop!
#         final_answer = await got_orchestrator.run(
#             tasks=tasks,
#             user_profile=user_profiler.profile, # Passes the Bayesian Math!
#             model=self.llm,                     # Gemini acts as the Internal Node Judge
#             agent_executor=self._execute_agent  # Passes our real tools to the DAG
#         )
        
#         # 6. Global Evaluation
#         print("🧠 [Orchestrator] Running Final Global Evaluation...")
#         judge_score_raw = await self._gpt_as_judge(user_input, final_answer)
#         match = re.search(r'Score:\s*(\d+)/10', judge_score_raw, re.IGNORECASE)
#         numeric_score = int(match.group(1)) if match else 0

#         return {"final_answer": final_answer, "judge_score": judge_score_raw, "numeric_score": numeric_score}

#     async def run(self, user_input: str) -> Dict[str, Any]:
#         if config.Config.DEFAULT_DECOMPOSER == "auto":
#             decomposer_name = self.selector.select(user_input, deterministic=False).name
#         else:
#             decomposer_name = config.Config.DEFAULT_DECOMPOSER
            
#         print(f"⚙️  [Orchestrator] Selected Decomposer: {decomposer_name.upper()}")
#         return await self._execute_pipeline(user_input, decomposer_name)

#     async def run_benchmark(self, questions: List[str]):
#         decomposers = ["plan_and_solve", "llm_compiler", "qdmr_rule", "coverage", "ultimate_neuro_symbolic"]
#         scores = {d: [] for d in decomposers}

#         print("\n" + "="*60)
#         print(f"🏆 INITIATING BENCHMARK (JUDGE: {config.Config.JUDGE_MODEL})")
#         print("="*60)

#         for d in decomposers:
#             print(f"\n=======================================")
#             print(f"🚀 TESTING ALGORITHM: {d.upper()}")
#             print(f"=======================================")

#             user_profiler.reset_all()

#             for i, q in enumerate(questions, 1):
#                 print(f"\n   Turn {i}/{len(questions)}: '{q}'")
#                 try:
#                     output = await self._execute_pipeline(q, d)
#                     scores[d].append(output["numeric_score"])
#                     print(f"   [Judge Score]: {output['numeric_score']}/10")
#                 except Exception as e:
#                     print(f"   [Error] {e}")
#                     scores[d].append(0)

#         print("\n" + "="*60)
#         print("📊 FINAL LEADERBOARD")
#         print("="*60)
#         avg_scores = {d: sum(s)/len(s) if s else 0 for d, s in scores.items()}
#         ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
#         for rank, (name, avg) in enumerate(ranked, 1):
#             print(f"{rank}. {name.upper().ljust(25)} : {avg:.1f} / 10")
#         print("="*60)

import asyncio
import os
import re
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Dict, Any, List

import config
from .decomposers import DecomposerRegistry, AutoDecomposerSelector
from .dag_builders import DAGRegistry

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

    async def _execute_agent(self, agent_name: str, query: str) -> str:
        """Standard Agent Execution"""
        try:
            if agent_name == "code": return await call_code_agent(query)
            if agent_name == "cloud": return call_cloud_agent(query)
            if agent_name == "scholar":
                limit = 1
                match = re.search(r'\b([1-5])\b', query)
                if match: limit = int(match.group(1))
                return await call_scholar_agent(query, is_collaboration=False, limit=limit)
            else: # RAG
                strategy_match = re.search(r'--strategy=(\w+)', query.lower())
                rag_strategy = strategy_match.group(1) if strategy_match else "tot"
                clean_query = re.sub(r'--strategy=\w+', '', query).strip()
                rag_data = await call_rag_agent(clean_query, strategy=rag_strategy)
                return f"Source: {rag_data.get('source_used')}\nAnswer: {rag_data['answer']}"
        except Exception as e:
            return f"Error executing agent: {e}"

    async def _resolve_dependencies(self, query: str, depends_on: List[Any], previous_results: Dict[str, Any]) -> str:
        if not depends_on or not self.llm: return query
        context = ""
        for dep_id in depends_on:
            for k, v in previous_results.items():
                if str(dep_id) in k:
                    context += f"--- Result from {k} ---\n{v}\n\n"
        prompt = f"You are a State Injector. Update the Original Query by replacing any placeholders with the specific information found in the Context. If it doesn't need updating, return it exactly as is. ONLY output the rewritten query.\nContext:\n{context}\nOriginal Query: {query}\nRewritten Query:"
        res = await asyncio.to_thread(self.llm.generate_content, prompt.strip())
        return res.text.strip()

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
    async def _execute_pipeline(self, user_input: str, decomposer_name: str) -> Dict[str, Any]:
        
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

            # 🚀 BUG FIXES 1, 2, 3, 4, 6, 7, 8: CLOSURE EXECUTOR
            async def got_executor(node: GoTNode) -> str:
                query = node.task.query
                
                # Fetch ACTUAL parent node objects from the live DAG (Fixes Bug 3 & 8)
                parent_nodes = [got_orchestrator.dag._nodes[pid] for pid in node.parent_ids] if got_orchestrator.dag else []

                # AGGREGATE NODE FIX (Fixes Bug 1, 2, 4, 7)
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

                # REFINE NODE FIX (Fixes Bug 6 & 7)
                if node.op == GoTOp.REFINE:
                    print(f"      [GoT Operator] Safe Processing for REFINE node...")
                    refine_prompt = f"Critique from Judge: {node.judge_reasoning}\n\nImprove the answer based on the critique.\n{query}"
                    res = await asyncio.to_thread(self.llm.generate_content, refine_prompt.strip())
                    return res.text.strip()

                # ROOT & BRANCH NODES (Standard State Injection - Fixes Bug 7)
                if node.parent_ids:
                    prev_results = {f"Task {p.task.id}": p.result for p in parent_nodes if p.result}
                    query = await self._resolve_dependencies(query, node.parent_ids, prev_results)
                    
                print(f"      [Executing] Node {node.node_id[:4]} ({node.task.agent.upper()}): {query[:60]}...")
                return await self._execute_agent(node.task.agent, query)

            print("\n   [GoT DAG] Commencing Adaptive Wave Execution...")
            # Bug 5 Fix: Pass fallback profile if profiler is off
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
                resolved_queries = []
                for t in layer:
                    if t.depends_on:
                        resolved_queries.append(await self._resolve_dependencies(t.query, t.depends_on, results))
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

        return {"final_answer": final_answer, "judge_score": judge_score_raw, "numeric_score": numeric_score}

    async def run(self, user_input: str) -> Dict[str, Any]:
        if config.Config.DEFAULT_DECOMPOSER == "auto":
            decomposer_name = self.selector.select(user_input, deterministic=False).name
        else:
            decomposer_name = config.Config.DEFAULT_DECOMPOSER
            
        print(f"⚙️  [Orchestrator] Selected Decomposer: {decomposer_name.upper()}")
        return await self._execute_pipeline(user_input, decomposer_name)

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