import os
import asyncio
import chromadb
import google.generativeai as genai

# GEMINI_MODEL = "gemini-flash-latest"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def _get_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

def _query_vector_db(query: str, n_results: int = 4) -> str:
    if not os.path.exists(DB_DIR): return ""
    try:
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path=DB_DIR)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_collection(name="cloud_architecture_vault", embedding_function=sentence_transformer_ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        
        retrieved_text = ""
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            retrieved_text += f"\n--- Source: {source} ---\n{doc}\n"
        return retrieved_text
    except: return ""

CLOUD_PROMPT = """
You are an Expert Cloud Infrastructure Architect for Bioinformatics.
Your task is to design a deployment architecture or write Infrastructure-as-Code for the following request:

USER REQUEST: "{subtask}"

USER PROFILE (Cost & Complexity Preference):
{profile_guideline}

You have been provided with the following verified architectural guidelines:
<database_context>
{context}
</database_context>

CRITICAL RULES FOR GENERATION (YOU MUST FOLLOW THESE EXACTLY):

1. PRIMARY SOURCE: Construct your architecture based primarily on the <database_context>.

2. MANDATORY AI EXTRAPOLATION WARNING: You MUST determine if the specific bioinformatics tool requested by the user (e.g., 'MaxQuant', 'CellProfiler') is explicitly named in the <database_context>.
   - If the tool is NOT named in the context, you are forced to extrapolate its hardware requirements. Your response MUST begin with this exact block BEFORE you write anything else:

=========================================================================
⚠️ AI EXTRAPOLATION WARNING
The specific tool requested was not found in the verified knowledge base. 
The following architecture is an AI-generated recommendation based on 
the expected compute profile of the tool.
=========================================================================

3. EXTRAPOLATION RULE: After pasting the warning, use your internal LLM expertise to categorize the tool (e.g., CPU-bound vs. GPU-bound) and apply the closest matching rules from the context.

4. PROFILE ALIGNMENT: Adapt your response style (code-heavy vs. explanation-heavy) based on the USER PROFILE.
"""

async def call_cloud_agent(user_input: str, strategy: str = "cot", depth: int = 1, agent_registry: str = "", user_profile_str: str = "") -> str:
    model = _get_model()
    if not model: return "[Cloud Agent - Mock] No API Key."

    try:
        retrieved_context = _query_vector_db(user_input)
        
        prompt = CLOUD_PROMPT.format(
            subtask=user_input, 
            context=retrieved_context if retrieved_context else "No local cloud data found. Use best practices.",
            profile_guideline=user_profile_str if user_profile_str else "Balance cost and performance."
        )
        
        # Generate the infrastructure code
        response = await asyncio.to_thread(model.generate_content, prompt)
        output_text = response.text.strip()
        
        # 🚀 NEW: Save the generated Infrastructure code to disk
        output_dir = os.path.join(os.path.dirname(__file__), "generated_infrastructure")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe, unique filename based on the query
        safe_name = "".join([c if c.isalnum() else "_" for c in user_input[:30]]).strip("_").lower()
        
        # Determine the file extension
        if "terraform" in user_input.lower() or "aws_" in output_text:
            filename = f"{safe_name}.tf"
        elif "docker" in user_input.lower():
            filename = f"{safe_name}_Dockerfile"
        else:
            filename = f"{safe_name}_architecture.md"
            
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output_text)

        # Return exactly what the Orchestrator expects (using safe string formatting)
        final_output = (
            f"[Provenance-Aware Cloud Agent] strategy={strategy} | depth={depth}\n"
            f"Infrastructure files saved locally to: cloud_agent/generated_infrastructure/{filename}\n\n"
            f"{output_text}\n"
        )
        _infra_kw = ("resource", "aws_", "provider", "module", "terraform", "aws", "ec2", "s3")
        if "```" not in output_text and not any(kw in output_text.lower() for kw in _infra_kw):
            final_output = f"[STRUCTURAL_FAIL: no infrastructure definition detected]\n{final_output}"
        return final_output

    except Exception as e:
        return f"[Cloud Agent Error] {e}"