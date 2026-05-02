import re
import json
import asyncio
from typing import List
import google.generativeai as genai
from .communication_log import EdgeLog

class CommunicationEvaluator:
    def __init__(self, model: genai.GenerativeModel):
        self.model = model

    def _check_dependency_accuracy(self, output: str) -> bool:
        if not output: return False
        return not (bool(re.search(r'\[.*?task.*?\]', output, re.IGNORECASE)) or "@@REF@@" in output)

    async def _evaluate_context_relevance(self, query: str, context: str) -> float:
        """Evaluates if the upstream context was actually relevant to the downstream task."""
        if not context: return 0.0
        prompt = f"""
            You are an AI evaluator measuring "Context Relevance".
            Downstream Task: "{query}"
            Upstream Context Provided: "{context[:2000]}"
            
            Does the Upstream Context contain information that is ON TOPIC for the Downstream Task?
            Rate strictly 1 to 5:
            1: Completely off-topic.
            3: Contains some related background information.
            5: Contains specific facts directly requested by the task.
            Return ONLY valid JSON: {{"score": <int>}}
        """
        try:
            res = await asyncio.to_thread(self.model.generate_content, prompt)
            data = json.loads(res.text.replace("```json", "").replace("```", "").strip())
            return max(0.0, min(1.0, (float(data.get("score", 1.0)) - 1) / 4.0))
        except: return 0.5

    async def _evaluate_semantic_utilization(self, context: str, output: str) -> float:
        """Uses the LLM Judge to determine if the context was actively utilized."""
        if not context or not output: return 0.0
        prompt = f"""
        You are an AI evaluator measuring "Context Utilization".
        Upstream Context: "{context[:2000]}"
        Downstream Output: "{output[:2000]}"
        
        Did the Output actively use/synthesize the Context? Rate strictly 1 to 5:
        1: Ignored completely.
        3: Used partially or superficially.
        5: Highly integrated and essential to the output.
        Return ONLY valid JSON: {{"score": <int>}}
        """
        try:
            res = await asyncio.to_thread(self.model.generate_content, prompt)
            data = json.loads(res.text.replace("```json", "").replace("```", "").strip())
            return max(0.0, min(1.0, (float(data.get("score", 1.0)) - 1) / 4.0))
        except: return 0.0

    async def evaluate_logs(self, logs: List[EdgeLog]):
        print(f"\n📊 [Evaluator] Grading {len(logs)} communication edges (Relevance & Utilization)...")
        async def process_single_log(log: EdgeLog):
            if not log.agent_output or not log.raw_context_injected:
                log.utilization_score = 0.0
                log.context_relevance = 0.0
                log.dependency_accuracy = False
                return
            log.dependency_accuracy = self._check_dependency_accuracy(log.agent_output)
            # Run evaluations concurrently
            rel, util = await asyncio.gather(
                self._evaluate_context_relevance(log.resolved_query, log.raw_context_injected),
                self._evaluate_semantic_utilization(log.raw_context_injected, log.agent_output)
            )
            log.context_relevance = round(rel, 3)
            log.utilization_score = round(util, 3)

        await asyncio.gather(*(process_single_log(log) for log in logs))