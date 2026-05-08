# """
# notebook_agent/agent.py

# Notebook Agent for the OTR multi-agent system.

# Given a researcher subtask from the orchestrator, this agent:
#   1. Runs embedding-based similarity against pre-configured catalog templates.
#   2. Score >= 0.65  →  adapts the best-matching template.
#   3. Score <  0.65 (or empty catalog)  →  generates a brand-new .ipynb from scratch.

# The generated notebook is saved to disk and the return value contains both a summary 
# AND the raw Python code, allowing the downstream Orchestrator to synthesize it properly.
# """

# import os
# import re
# import json
# import asyncio
# import textwrap
# import numpy as np
# from datetime import datetime
# from typing import Optional

# import google.generativeai as genai

# # ---------------------------------------------------------------------------
# # Catalog import with safe fallback for missing resources
# # ---------------------------------------------------------------------------
# try:
#     from code_agent.notebook_agent.templates.metadata.catalog import get_catalog, get_catalog_text
# except ImportError:
#     try:
#         from code_agent.templates.metadata.catalog import get_catalog, get_catalog_text
#     except ImportError:
#         # Safe fallback if the user has not built the catalog resources yet
#         def get_catalog(): return[]
#         def get_catalog_text(): return[]

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------
# ADAPT_THRESHOLD = 0.65
# OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "generated_notebooks")
# GEMINI_MODEL    = "gemini-2.5-flash"

# # ---------------------------------------------------------------------------
# # Gemini setup
# # ---------------------------------------------------------------------------
# def _get_model() -> Optional[genai.GenerativeModel]:
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         return None
#     genai.configure(api_key=api_key)
#     return genai.GenerativeModel(GEMINI_MODEL)

# # ---------------------------------------------------------------------------
# # Embedding retriever
# # ---------------------------------------------------------------------------
# try:
#     from sentence_transformers import SentenceTransformer as _ST
#     _encoder   = _ST("all-MiniLM-L6-v2")
#     _cat_texts = get_catalog_text()
#     _cat_embs  = _encoder.encode(_cat_texts, convert_to_numpy=True) if _cat_texts else[]
#     _USE_EMB   = True if _cat_texts else False
# except Exception:
#     _USE_EMB   = False

# def _cosine(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# _STOP = {"the", "and", "for", "with", "are", "what", "how", "that", "this",
#          "using", "from", "into", "data", "analysis", "pipeline"}

# def _keyword_score(query: str, nb: dict) -> float:
#     q  = set(re.findall(r"[a-z]{3,}", query.lower())) - _STOP
#     nb_text = " ".join(nb.get("tags", []) + nb.get("tools", []) +[nb.get("title", ""), nb.get("description", "")])
#     nw = set(re.findall(r"[a-z]{3,}", nb_text.lower())) - _STOP
#     return len(q & nw) / (len(q) + 1e-8)

# def _retrieve_top_k(query: str, k: int = 3) -> list[dict]:
#     catalog = get_catalog()
#     if not catalog:
#         return[] # Safe return if resources are missing

#     if _USE_EMB:
#         q_emb  = _encoder.encode([query], convert_to_numpy=True)[0]
#         scores =[_cosine(q_emb, e) for e in _cat_embs]
#     else:
#         scores =[_keyword_score(query, nb) for nb in catalog]
        
#     ranked = sorted(zip(scores, catalog), key=lambda x: x[0], reverse=True)
#     results = []
#     for score, nb in ranked[:k]:
#         entry = dict(nb)
#         entry["similarity_score"] = round(score, 4)
#         results.append(entry)
#     return results

# # ---------------------------------------------------------------------------
# # Prompt templates
# # ---------------------------------------------------------------------------
# _ADAPT_PLAN_PROMPT = """\
# You are an expert Jupyter notebook engineer for neuroscience and bioinformatics.

# A researcher needs:
# \"\"\"{subtask}\"\"\"

# The closest existing notebook template is:
#   Title:         {nb_title}
#   Description:   {nb_description}
#   Pipeline type: {pipeline_type}
#   Tools:         {tools}
#   Compute:       {compute}
#   Similarity:    {score:.2f}

# Reasoning strategy: {strategy} (depth={depth})

# Describe exactly how to adapt this template for the researcher's need.
# Return ONLY valid JSON — no markdown fences, no prose outside the object:
# {{
#   "pipeline_type":   "<canonical pipeline type>",
#   "notebook_title":  "<title for the adapted notebook>",
#   "compute_estimate": {{
#     "memory":                  "<XGB>",
#     "cpu":                     "<N cores>",
#     "gpu":                     "<not required | V100 recommended | V100 required | A100 required>",
#     "estimated_runtime_hours": <number>
#   }},
#   "sections":[
#     {{
#       "title":   "<Section heading>",
#       "purpose": "<one sentence on what this section does>",
#       "key_code": "<the most important function call or import for this section>"
#     }}
#   ],
#   "tools_to_add":    ["<tool>"],
#   "tools_to_remove": ["<tool>"],
#   "adaptation_notes": "<brief explanation of what specifically changes vs the template>"
# }}
# """

# _CREATE_PLAN_PROMPT = """\
# You are an expert Jupyter notebook engineer for neuroscience and bioinformatics.

# A researcher needs:
# \"\"\"{subtask}\"\"\"

# No existing template matches closely enough (best score: {score:.2f}).
# The structurally closest reference is:
#   Title:       {nb_title}
#   Description: {nb_description}

# Reasoning strategy: {strategy} (depth={depth})

# Design a brand-new notebook for this request.
# Return ONLY valid JSON — no markdown fences, no prose outside the object:
# {{
#   "pipeline_type":  "<canonical pipeline type>",
#   "notebook_title": "<descriptive title>",
#   "compute_estimate": {{
#     "memory":                  "<XGB>",
#     "cpu":                     "<N cores>",
#     "gpu":                     "<not required | V100 recommended | V100 required | A100 required>",
#     "estimated_runtime_hours": <number>
#   }},
#   "sections":[
#     {{
#       "title":    "<Section heading>",
#       "purpose":  "<one sentence>",
#       "key_code": "<most important function call or import for this section>"
#     }}
#   ],
#   "tools":   ["<tool1>", "<tool2>"],
#   "rationale": "<why no existing template fits>"
# }}
# """

# _CELL_CODE_PROMPT = """\
# You are writing a single Jupyter notebook code cell for a bioinformatics/neuroscience pipeline.

# Notebook title: {notebook_title}
# Section: "{section_title}"
# Purpose: {section_purpose}
# Key code hint: {key_code}
# Tools available: {tools}

# Write ONLY the Python code for this cell.
# - Include a brief comment block at the top explaining what the cell does.
# - Use realistic, runnable code (proper imports, function calls, variable names).
# - Do NOT include markdown, explanations outside comments, or cell delimiters.
# - Assume previous cells have already imported standard libraries (os, numpy, pandas, matplotlib).
# """

# # ---------------------------------------------------------------------------
# # JSON parser
# # ---------------------------------------------------------------------------
# def _parse_json(text: str) -> dict:
#     text = text.strip()
#     if "```" in text:
#         for part in text.split("```")[1::2]:
#             cleaned = part.strip().lstrip("json").strip()
#             try:
#                 return json.loads(cleaned)
#             except json.JSONDecodeError:
#                 continue
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         pass
#     s, e = text.find("{"), text.rfind("}")
#     if s != -1 and e > s:
#         try:
#             return json.loads(text[s:e + 1])
#         except json.JSONDecodeError:
#             pass
#     raise ValueError(f"Could not extract JSON:\n{text[:300]}")

# # ---------------------------------------------------------------------------
# # Notebook builder
# # ---------------------------------------------------------------------------
# def _make_cell(cell_type: str, source: str) -> dict:
#     if cell_type == "markdown":
#         return {"cell_type": "markdown", "metadata": {}, "source": source}
#     return {
#         "cell_type":       "code",
#         "execution_count": None,
#         "metadata":        {},
#         "outputs":[],
#         "source":          source,
#     }

# async def _build_notebook(plan: dict, subtask: str, model, strategy: str) -> str:
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     title   = plan.get("notebook_title", "Generated Notebook")
    
#     # Safely handle tools
#     tools_list = plan.get("tools", plan.get("tools_to_add",[]))
#     tools_str = ", ".join(tools_list) if isinstance(tools_list, list) else str(tools_list)
    
#     sections = plan.get("sections", [])
#     cells =[]

#     # Title cell
#     cells.append(_make_cell("markdown", "\n".join([
#         f"# {title}",
#         "",
#         f"**Generated by OTR Notebook Agent** | strategy={strategy} | {datetime.now().strftime('%Y-%m-%d')}",
#         "",
#         f"**Research need:** {subtask}",
#         "",
#         f"**Pipeline type:** `{plan.get('pipeline_type', 'unknown')}`",
#         "",
#         f"**Tools:** {tools_str or 'see sections below'}",
#     ])))

#     # Compute estimate cell
#     ce = plan.get("compute_estimate", {})
#     cells.append(_make_cell("markdown", "\n".join([
#         "## Compute Requirements",
#         "",
#         f"| Resource | Estimate |",
#         f"|----------|----------|",
#         f"| Memory   | {ce.get('memory', 'N/A')} |",
#         f"| CPU      | {ce.get('cpu', 'N/A')} |",
#         f"| GPU      | {ce.get('gpu', 'N/A')} |",
#         f"| Runtime  | ~{ce.get('estimated_runtime_hours', '?')} hours |",
#     ])))

#     # Standard imports cell
#     cells.append(_make_cell("code", textwrap.dedent("""\
#         # Standard imports — present in every OTR-generated notebook
#         import os
#         import numpy as np
#         import pandas as pd
#         import matplotlib.pyplot as plt
#         import warnings
#         warnings.filterwarnings("ignore")
#     """)))

#     for sec in sections:
#         sec_title   = sec.get("title",    "Section")
#         sec_purpose = sec.get("purpose",  "")
#         key_code    = sec.get("key_code", "")

#         cells.append(_make_cell("markdown", f"## {sec_title}\n\n{sec_purpose}"))

#         if model:
#             code_prompt = _CELL_CODE_PROMPT.format(
#                 notebook_title  = title,
#                 section_title   = sec_title,
#                 section_purpose = sec_purpose,
#                 key_code        = key_code,
#                 tools           = tools_str,
#             )
#             try:
#                 resp = await asyncio.to_thread(model.generate_content, code_prompt)
#                 code_src = resp.text.strip()
#                 if "```" in code_src:
#                     parts = code_src.split("```")
#                     for p in parts[1::2]:
#                         cleaned = p.strip().lstrip("python").strip()
#                         if cleaned:
#                             code_src = cleaned
#                             break
#             except Exception as e:
#                 code_src = f"# [Notebook Agent] Could not generate cell: {e}\n# Key hint: {key_code}"
#         else:
#             code_src = f"# TODO: {sec_title}\n# {sec_purpose}\n# Key: {key_code}"

#         cells.append(_make_cell("code", code_src))

#     notebook = {
#         "nbformat": 4,
#         "nbformat_minor": 5,
#         "metadata": {
#             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
#             "language_info": {"name": "python", "version": "3.11.0"},
#             "otr_metadata": {
#                 "pipeline_type":    plan.get("pipeline_type"),
#                 "compute_estimate": ce,
#                 "generated_by":     "OTR Notebook Agent",
#                 "strategy":         strategy,
#             },
#         },
#         "cells": cells,
#     }

#     safe_title = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]
#     filename   = f"{safe_title}.ipynb"
#     filepath   = os.path.join(OUTPUT_DIR, filename)
#     with open(filepath, "w", encoding="utf-8") as f:
#         json.dump(notebook, f, indent=1)

#     return filepath


# # ---------------------------------------------------------------------------
# # Public entry point
# # ---------------------------------------------------------------------------
# async def call_code_agent(
#     user_input:     str,
#     strategy:       str = "cot",
#     depth:          int = 1,
#     agent_registry: str = "",
# ) -> str:
    
#     model = _get_model()

#     if not model:
#         return "[Notebook Agent - Mock] No GOOGLE_API_KEY. Would generate notebook for: " + user_input

#     try:
#         top_matches = _retrieve_top_k(user_input, k=3)
        
#         # Safe fallback if catalog is empty or missing
#         if top_matches:
#             best        = top_matches[0]
#             score       = best["similarity_score"]
#         else:
#             best        = {"id": "none", "title": "N/A", "description": "No templates available in local catalog.", "pipeline_type": "custom", "tools":[], "compute_estimate": {}}
#             score       = 0.0

#         if score >= ADAPT_THRESHOLD:
#             plan_prompt = _ADAPT_PLAN_PROMPT.format(
#                 subtask       = user_input,
#                 nb_title      = best["title"],
#                 nb_description= best["description"],
#                 pipeline_type = best.get("pipeline_type", "unknown"),
#                 tools         = ", ".join(best.get("tools",[])),
#                 compute       = json.dumps(best.get("compute_estimate", {})),
#                 score         = score,
#                 strategy      = strategy,
#                 depth         = depth,
#             )
#             mode = "adapted"
#         else:
#             plan_prompt = _CREATE_PLAN_PROMPT.format(
#                 subtask       = user_input,
#                 score         = score,
#                 nb_title      = best["title"],
#                 nb_description= best["description"],
#                 strategy      = strategy,
#                 depth         = depth,
#             )
#             mode = "created"

#         plan_response = await asyncio.to_thread(model.generate_content, plan_prompt)
#         plan          = _parse_json(plan_response.text)

#         filepath = await _build_notebook(plan, user_input, model, strategy)
#         filename = os.path.basename(filepath)

#         ce       = plan.get("compute_estimate", {})
#         sections =[s.get("title", "") for s in plan.get("sections", [])]

#         # Base summary array
#         lines =[
#             f"[Notebook Agent] {mode.upper()} | strategy={strategy} | depth={depth}",
#             f"Notebook:      {filename}",
#             f"Pipeline type: {plan.get('pipeline_type', 'unknown')}",
#             f"Compute:       {ce.get('memory','?')} RAM | {ce.get('cpu','?')} | "
#             f"GPU: {ce.get('gpu','?')} | ~{ce.get('estimated_runtime_hours','?')}h",
#             f"Sections:      {' → '.join(sections)}",
#             f"Based on:      {best['id']} ({best['title']}, score={score:.2f})"
#         ]
        
#         if mode == "adapted":
#             lines.append(f"Adaptation:    {plan.get('adaptation_notes', '')}")
#         else:
#             lines.append(f"Rationale:     {plan.get('rationale', '')}")

#         # 🚀 THE FIX: Extract ONLY the clean Python code from the saved notebook.
#         # Do NOT pass the raw JSON .ipynb structure back to the Orchestrator, 
#         # as it will blow up the context window and confuse the LLM Judge.
        
#         with open(filepath, "r", encoding="utf-8") as f:
#             nb_data = json.load(f)
            
#         clean_python_script = []
        
#         # 1. Add a header so the Judge knows this came from a notebook
#         clean_python_script.append(f"# Automatically generated from notebook: {filename}")
#         clean_python_script.append(f"# Pipeline Type: {plan.get('pipeline_type', 'unknown')}\n")
        
#         # 2. Iterate through the cells and extract ONLY the code
#         for cell in nb_data.get("cells", []):
#             if cell.get("cell_type") == "code":
#                 source = cell.get("source", "")
#                 if isinstance(source, list):
#                     source = "".join(source)
                
#                 # Strip out any markdown fences that Gemini might have accidentally included
#                 source = source.replace("```python", "").replace("```", "").strip()
                
#                 if source:
#                     clean_python_script.append(source)
#                     clean_python_script.append("\n# " + "-"*40 + "\n") # Visual separator between cells
                
#         # 3. Join it all into a single, clean, readable Python string
#         full_code_string = "\n".join(clean_python_script)
        
#         # 4. Truncate the code if it is insanely massive to protect the context window
#         # (Allows roughly 3,000 tokens of pure Python code)
#         if len(full_code_string) > 12000: 
#             full_code_string = full_code_string[:12000] + "\n\n# ... [CODE TRUNCATED DUE TO LENGTH] ..."
        
#         # 5. Append the clean code block to the output
#         lines.append("\n### Generated Python Script (Extracted from Notebook):")
#         lines.append("```python\n" + full_code_string + "\n```")

#         return "\n".join(lines)

#     except Exception as e:
#         return f"[Notebook Agent Error] {e}"

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