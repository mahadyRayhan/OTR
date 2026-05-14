"""
code_agent/agent.py

Provenance-Aware Code Agent for the OTR multi-agent system.

This agent uses a local ChromaDB instance to retrieve verified Python code and 
markdown explanations from official Jupyter Notebooks (.ipynb files). 
It then uses Gemini to assemble those verified blocks into a cohesive Python 
script, strictly enforcing provenance and minimizing hallucinations.

Engine-facing signature (called by orchestrator/engine.py -> _execute_agent):

    async def call_code_agent(
        user_input: str,
        strategy:   str = "cot",
        depth:      int = 1,
        agent_registry: str = "",
    ) -> str
"""

import os
import asyncio
import chromadb
import google.generativeai as genai
from typing import Optional

# Ensure chromadb doesn't spam warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Constants & Setup
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def _get_model() -> Optional[genai.GenerativeModel]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

# ---------------------------------------------------------------------------
# Vector DB Retrieval
# ---------------------------------------------------------------------------
def _query_vector_db(query: str, n_results: int = 5) -> str:
    """Queries local ChromaDB and returns the raw markdown/code chunks."""
    if not os.path.exists(DB_DIR):
        print(f"[Code Agent] Warning: ChromaDB directory not found at {DB_DIR}")
        return ""
        
    try:
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path=DB_DIR)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        collection = client.get_collection(name="scanpy_code_vault", embedding_function=sentence_transformer_ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        retrieved_text = ""
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i].get('source', 'Unknown Notebook')
            retrieved_text += f"\n--- Source File: {source} ---\n{doc}\n"
            
        return retrieved_text
    except Exception as e:
        print(f"[Code Agent] Vector DB Query Error: {e}")
        return ""

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
# PROVENANCE_PROMPT = """
# You are a Provenance-Aware Bioinformatics Code Engineer.
# Your task is to write a complete, logical Python script for the following request:

# USER REQUEST: "{subtask}"

# You have been provided with the following verified code blocks and pipeline explanations extracted directly from official Jupyter Notebooks:

# <verified_database_context>
# {context}
# </verified_database_context>

# CRITICAL RULES FOR GENERATION:
# 1. PIPELINE LOGIC: Read the "Explanation" sections in the context to understand the correct order of operations (e.g., Normalization must happen before PCA).
# 2. PRIMARY SOURCE: You must construct the script primarily using the exact code structures, functions, and parameters found in the <verified_database_context>.
# 3. GAP FILLING: If a necessary basic step (e.g., importing `scanpy`, loading a dummy dataset to make the code runnable) is missing from the context, you MAY write it yourself.
# 4. THE WARNING LABEL: If you write ANY functional algorithmic code (e.g., a specific clustering method, statistical test, or plotting function) that is not explicitly present in the <verified_database_context>, you MUST wrap that specific block of code with this exact comment header:

# # =====================================================================
# # ⚠️ WARNING: THIS SECTION IS GENERATED USING LLM
# # This algorithmic step was not found in the official knowledge base.
# # =====================================================================

# 5. REFUSAL: If the core biological algorithm requested by the user is missing entirely from the context, do NOT invent it. Output a python comment saying: "# ERROR: The requested core algorithm is not present in the local knowledge base."

# Return ONLY the final assembled Python script wrapped in markdown ```python ``` blocks. Do not include conversational text outside the blocks. Include inline comments explaining the pipeline steps.
# """

PROVENANCE_PROMPT = """
You are a Strict Provenance-Aware Bioinformatics Code Engineer.
Your task is to write a complete Python script for the following request:

USER REQUEST: "{subtask}"

You have been provided with the following verified code blocks from official documentation:
<verified_database_context>
{context}
</verified_database_context>

CRITICAL RULES FOR GENERATION (YOU MUST FOLLOW THESE EXACTLY):

1. PRIMARY SOURCE: You must construct the script primarily using the exact code structures found in the <verified_database_context>.

2. MANDATORY AI WARNING DECLARATION: You MUST determine if the core biological algorithm requested by the user is missing from the <verified_database_context>. 
   - If it IS missing, you are forced to extrapolate. The VERY FIRST LINE of your python code block MUST BE this exact warning:

# =====================================================================
# ⚠️ AI GENERATED CODE WARNING
# This specific algorithmic step was generated by the LLM. 
# It was not found in the official verified knowledge base.
# =====================================================================

3. GAP FILLING: If you use the warning above, you may then proceed to write the algorithm using your internal LLM knowledge to fulfill the user request.

4. FORMAT: Return ONLY the final assembled Python script wrapped in markdown ```python ``` blocks. Do not include conversational text outside the blocks. Include inline comments.
"""

# ---------------------------------------------------------------------------
# Public Entry Point (Called by Orchestrator)
# ---------------------------------------------------------------------------
async def call_code_agent(
    user_input: str, 
    strategy: str = "cot", 
    depth: int = 1, 
    agent_registry: str = ""
) -> str:
    """
    Called by orchestrator/engine.py -> _execute_agent() when agent_name == "code".
    """
    model = _get_model()
    if not model:
        return f"[Code Agent - Mock] No API Key. Would generate code for: {user_input}"

    try:
        # 1. Retrieve trusted code chunks and pipeline logic from ChromaDB
        retrieved_context = _query_vector_db(user_input, n_results=5)
        
        if not retrieved_context.strip():
            return "[Code Agent] ERROR: Local Vector DB is empty or missing. Please run build_vector_db.py to ingest the .ipynb notebooks."

        # 2. Build the Provenance Prompt
        prompt = PROVENANCE_PROMPT.format(subtask=user_input, context=retrieved_context)
        
        # 3. Generate the script via Gemini
        response = await asyncio.to_thread(model.generate_content, prompt)
        output_text = response.text.strip()
        
        # 4. Extract just the Python code
        clean_python_script = []
        if "```" in output_text:
            parts = output_text.split("```")
            for p in parts[1::2]:
                cleaned = p.lstrip("python").lstrip("python3").strip()
                if cleaned: 
                    clean_python_script.append(cleaned)
        else:
            clean_python_script.append(output_text)
            
        full_code = "\n\n".join(clean_python_script)

        # 🚀 NEW: Save the generated Python script to disk
        output_dir = os.path.join(os.path.dirname(__file__), "generated_scripts")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe, unique filename based on the query
        safe_name = "".join([c if c.isalnum() else "_" for c in user_input[:30]]).strip("_").lower()
        filename = f"{safe_name}.py"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_code)

        # 5. Return exactly what the Orchestrator expects (using safe string formatting)
        final_output = (
            f"[Provenance-Aware Code Agent] strategy={strategy} | depth={depth}\n"
            f"Script generated and saved locally to: code_agent/generated_scripts/{filename}\n\n"
            f"### Generated Python Script:\n"
            f"```python\n"
            f"{full_code}\n"
            f"```\n"
        )
        return final_output

    except Exception as e:
        return f"[Code Agent Error] {e}"