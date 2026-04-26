import asyncio
from typing import Optional
import google.generativeai as genai

class BaseReasoningStrategy:
    """Base interface for cognitive reasoning strategies."""
    name: str = "base"

    async def execute(self, query: str, context: str, model: Optional[genai.GenerativeModel]) -> str:
        raise NotImplementedError("Strategies must implement the execute method.")
        
    def _mock_fallback(self, query: str, context: str, template: str) -> str:
        """A safe fallback if the API key is missing."""
        clean_context = context.replace('\n', ' ')[:250] + "..."
        return template.format(query=query, context=clean_context)

class ChainOfThoughtStrategy(BaseReasoningStrategy):
    name = "cot"
    
    async def execute(self, query: str, context: str, model: Optional[genai.GenerativeModel]) -> str:
        if not model:
            return self._mock_fallback(query, context, 
                "Step 1: Analyze query.\nStep 2: Extract facts -> {context}\nFinal Answer: Based on the facts, {context}")
            
        prompt = f"""
        You are an AI answering a query based on a specific context. Use Chain of Thought reasoning.
        Query: {query}
        Context: {context}
        
        Format your response exactly as:
        Step 1: [Analyze the query]
        Step 2: [Extract relevant information from context]
        Step 3: [Synthesize]
        Final Answer:[Your clear, concise answer]
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()

class ReactStrategy(BaseReasoningStrategy):
    name = "react"
    
    async def execute(self, query: str, context: str, model: Optional[genai.GenerativeModel]) -> str:
        if not model:
            return self._mock_fallback(query, context, 
                "Thought 1: Search for {query}.\nAction: Read DB.\nObservation: {context}\nFinal Answer: {context}")
            
        prompt = f"""
        Simulate a Reason+Act (ReAct) loop to answer the query using the context.
        Query: {query}
        Context: {context}
        
        Format your response exactly as:
        Thought 1:[What do I need to find?]
        Action 1: Search context for keywords.
        Observation 1:[What did I find in the context?]
        Thought 2: [Does this answer the query?]
        Final Answer:[Your clear, concise answer]
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()

class TreeOfThoughtsStrategy(BaseReasoningStrategy):
    name = "tot"
    
    async def execute(self, query: str, context: str, model: Optional[genai.GenerativeModel]) -> str:
        if not model:
            return self._mock_fallback(query, context, 
                "Branch 1: Focus on tools.\nBranch 2: Focus on methods -> {context}\nResolution: Branch 2 is best.\nFinal Answer: {context}")
            
        prompt = f"""
        Use Tree of Thoughts reasoning to answer the query based on the context.
        Query: {query}
        Context: {context}
        
        Format your response exactly as:
        Branch 1:[Hypothesis 1 based on context] (Evaluation: [High/Med/Low] confidence)
        Branch 2: [Hypothesis 2 based on context] (Evaluation: [High/Med/Low] confidence)
        Resolution: [Select the best branch and explain why]
        Final Answer: [Your clear, concise answer]
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()

class GraphOfThoughtsStrategy(BaseReasoningStrategy):
    name = "got"
    
    async def execute(self, query: str, context: str, model: Optional[genai.GenerativeModel]) -> str:
        if not model:
            return self._mock_fallback(query, context, 
                "Node A: Query intent.\nNode B: Context -> {context}\nEdge: Merge A+B.\nFinal Answer: {context}")
            
        prompt = f"""
        Use Graph of Thoughts reasoning to combine different pieces of information to answer the query.
        Query: {query}
        Context: {context}
        
        Format your response exactly as:
        Node A (Query Intent): [Extract core intent]
        Node B (Context Fact 1): [Extract first key fact]
        Node C (Context Fact 2):[Extract second key fact if applicable]
        Edge (Synthesis): [Merge Nodes A, B, and C]
        Final Answer:[Your clear, concise answer]
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()

# Plugin Registry
class ReasoningEngine:
    def __init__(self):
        self._strategies = {}
        # Register strategies here to make them plug-and-play
        self.register(ChainOfThoughtStrategy())
        self.register(ReactStrategy())
        self.register(TreeOfThoughtsStrategy())
        self.register(GraphOfThoughtsStrategy())
        
    def register(self, strategy: BaseReasoningStrategy):
        self._strategies[strategy.name] = strategy
        
    def get_strategy(self, name: str) -> BaseReasoningStrategy:
        # Default to CoT if strategy not found
        return self._strategies.get(name.lower(), self._strategies["cot"])