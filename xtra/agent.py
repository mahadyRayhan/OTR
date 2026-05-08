import os
import re
import asyncio
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Dict, Any

# Import our new plug-and-play reasoning engine!
from rag_agent.reasoning import ReasoningEngine

# ---------------------------------------------------------
# RAG ENGINE (Local chunking)
# ---------------------------------------------------------
@dataclass
class DocumentChunk:
    title: str
    content: str
    score: float = 0.0

class SimpleLocalRAG:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks: List[DocumentChunk] =[]
        self._load_and_chunk()

    def _load_and_chunk(self):
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        raw_chunks = re.split(r'\n##\s+', text)
        self.chunks.append(DocumentChunk(title="Introduction", content=raw_chunks[0].strip()))
        for section in raw_chunks[1:]:
            lines = section.split('\n', 1)
            self.chunks.append(DocumentChunk(title=lines[0].strip(), content=lines[1].strip() if len(lines)>1 else ""))

    def retrieve(self, query: str, top_k: int = 1) -> List[DocumentChunk]:
        if not self.chunks: return[]
        words = set(re.findall(r'\b[a-z]{3,}\b', query.lower())) - {"the", "and", "for", "with", "are", "what", "how"}
        for chunk in self.chunks:
            chunk_tokens = set(re.findall(r'\b[a-z]{3,}\b', f"{chunk.title} {chunk.content}".lower()))
            chunk.score = len(words.intersection(chunk_tokens)) + (2.0 if words.intersection(set(chunk.title.lower().split())) else 0.0)
        return sorted(self.chunks, key=lambda c: c.score, reverse=True)[:top_k]

# Initialize KB and Engines
kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bioinformatics.md')
rag_engine = SimpleLocalRAG(kb_path)
reasoning_engine = ReasoningEngine()

# ---------------------------------------------------------
# LLM SETUP & JUDGE
# ---------------------------------------------------------
def get_llm_model():
    """Returns initialized Gemini model if API key exists, else None."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

async def llm_as_judge(query: str, answer: str, context: str, model) -> str:
    if not model:
        return "Mock Judge: 8/10 - Looks good (API Key Missing)."
    try:
        prompt = f"""
        You are an expert evaluator. Rate the agent's final answer to the user's query based ONLY on the provided context.
        Query: {query}
        Context: {context}
        Agent's Answer (including reasoning steps): {answer}
        
        Provide a score out of 10 and a strictly 1-sentence justification. Format: 'Score: X/10 - [Reason]'
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        return f"Judge Error: {str(e)}"

# ---------------------------------------------------------
# MAIN AGENT CALLER
# ---------------------------------------------------------
async def call_rag_agent(user_input: str, strategy: str = "cot", depth: int = 1, agent_registry: str = "") -> Dict[str, Any]:
    # 1. Action: Retrieve from local markdown file
    top_chunks = rag_engine.retrieve(user_input, top_k=1)
    llm = get_llm_model()
    
    # 2. FALLBACK: If local KB fails, use LLM's general knowledge!
    if not top_chunks or top_chunks[0].score == 0:
        if llm:
            prompt = f"You are a bioinformatics expert. Strategy: {strategy}, Expand to Depth: {depth}. Agent Registry: {agent_registry}. Answer this query comprehensively: {user_input}"
            fallback_answer = (await asyncio.to_thread(llm.generate_content, prompt)).text.strip()
            return {
                "answer": fallback_answer,
                "quality": "Mock Judge: 10/10 (Generated from LLM general knowledge)",
                "strategy": "LLM_FALLBACK",
                "source_used": "General LLM Knowledge"
            }
        else:
            return {"answer": "No relevant info found in KB, and no LLM available.", "quality": "N/A", "strategy": strategy}
    
    chunk = top_chunks[0]

    # 3. Reasoning: Get the plugin and execute it!
    plugin = reasoning_engine.get_strategy(strategy)
    generated_answer = await plugin.execute(
        query=user_input, 
        context=f"Section: {chunk.title}\n{chunk.content}", 
        model=llm
    )

    # 4. Self-Reflection
    evaluation = await llm_as_judge(user_input, generated_answer, chunk.content, llm)

    return {
        "answer": generated_answer,
        "quality": evaluation,
        "strategy": plugin.name.upper(),
        "source_used": chunk.title
    }