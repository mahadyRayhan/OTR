import os
import asyncio
import csv
import re
import chromadb
import gemini_client as genai
from orchestrator.context_format import entities, task_text

# GEMINI_MODEL = "gemini-flash-latest"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
# SCIWEAVER_INDEX_DIR selects an alternative index set (e.g. a benchmark-aligned corpus).
DB_DIR = (os.path.join(os.environ["SCIWEAVER_INDEX_DIR"], "cloud") if os.environ.get("SCIWEAVER_INDEX_DIR")
          else os.path.join(os.path.dirname(__file__), "chroma_db"))

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

# AWS on-demand prices (us-east-1), from journal_fixes/scripts/collect_aws_catalog.py.
CATALOG_CSV = os.path.join(os.path.dirname(__file__), "aws_ec2_us-east-1.csv")
INSTANCE_PLACEHOLDER = "<INSTANCE>"
# Accelerator families. The catalog's gpu column is empty for some (FPGA f, Gaudi dl) and set for
# others (Inferentia inf), so families decide: GPU ones (p, g, gr) and non-GPU ones (inf, trn, dl, f, vt).
GPU_FAMILY = re.compile(r"^(?:p\d|g\d|gr\d)")
NON_GPU_ACCELERATOR_FAMILY = re.compile(r"^(?:inf\d|trn\d|dl\d|f\d|vt\d)")
_CATALOG = None


def _catalog() -> list[dict]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = []
        for r in csv.DictReader(open(CATALOG_CSV)):
            try:
                _CATALOG.append({"instance": r["instance"], "family": r["family"], "vcpu": int(r["vcpu"]),
                                 "mem": float(r["memory"].split()[0].replace(",", "")), "gpu": bool(r["gpu"]),
                                 "arm": "Graviton" in r["processor"], "current": r["current_gen"] == "Yes",
                                 "usd": float(r["usd_per_hour"])})
            except ValueError:
                continue
    return _CATALOG


def _fitting_instances(cpus: float, mem: float, request: str, k: int = 3) -> list[dict]:
    """Cheapest current-generation instances with at least cpus vCPUs and mem GiB. Burstable t-family
    instances cannot sustain batch jobs and are skipped; GPU and Arm (Graviton) instances only when
    the request asks for them, since most bioinformatics binaries are x86 builds."""
    want_gpu = bool(re.search(r"\b(?:gpus?|cuda)\b", request, re.I)) and not re.search(
        r"\b(?:no|non|without|not)[\s-]+(?:a\s+)?gpus?\b", request, re.I)
    want_arm = bool(re.search(r"\b(arm|graviton)\b", request, re.I))
    ok = [r for r in _catalog() if r["current"] and not r["family"].startswith("t")
          and not NON_GPU_ACCELERATOR_FAMILY.match(r["family"])
          and bool(GPU_FAMILY.match(r["family"])) == want_gpu
          and r["arm"] == want_arm and r["vcpu"] >= cpus and r["mem"] >= mem]
    return sorted(ok, key=lambda r: r["usd"])[:k]


def _attach_instance(output_text: str, request: str) -> str:
    """Pick the instance from the price catalog for the SIZING line, so the instance always fits the estimate."""
    m = re.search(r"SIZING:\s*cpus\s*=\s*(\d+(?:\.\d+)?)\s*,?\s*memory_gb\s*=\s*(\d+(?:\.\d+)?)", output_text, re.I)
    if not m:
        return output_text.replace(INSTANCE_PLACEHOLDER, "(instance not selected: no SIZING line)")
    cpus, mem = float(m.group(1)), float(m.group(2))
    fits = _fitting_instances(cpus, mem, request)
    if not fits:
        return output_text.replace(INSTANCE_PLACEHOLDER, "(no catalog instance fits)") + (
            f"\n\nCATALOG CHECK: no current-generation instance has {cpus:g} vCPUs and {mem:g} GiB.")
    best = fits[0]
    others = ", ".join(f"{r['instance']} (${r['usd']:.4f}/h)" for r in fits[1:])
    return output_text.replace(INSTANCE_PLACEHOLDER, best["instance"]) + (
        f"\n\nCATALOG-VERIFIED INSTANCE (AWS us-east-1 on-demand, current generation): {best['instance']} - "
        f"{best['vcpu']} vCPU, {best['mem']:g} GiB, ${best['usd']:.4f}/h; cheapest fit for {cpus:g} vCPU / {mem:g} GiB."
        + (f" Next cheapest: {others}." if others else ""))


def _depth_reasoning_prefix(depth: int) -> str:
    """Returns a reasoning instruction appended to the cloud prompt at depth > 1."""
    if depth <= 1:
        return ""
    if depth == 2:
        return (
            "\n\nREASONING REQUIREMENT (depth=2): Before specifying the architecture, "
            "write a brief reasoning block: (1) estimate compute/memory/storage requirements "
            "for the workload, (2) identify the key cost and scalability trade-offs. "
            "Then provide the complete infrastructure definition."
        )
    return (
        "\n\nREASONING REQUIREMENT (depth=3): This is a high-priority node requiring rigorous output. "
        "Structure your response as follows:\n"
        "1. WORKLOAD ANALYSIS: Characterise the bioinformatics tool's resource profile "
        "(CPU/GPU/memory-bound, I/O patterns, parallelism).\n"
        "2. ARCHITECTURAL OPTIONS: Compare two viable approaches with cost, scalability, "
        "and fault-tolerance trade-offs.\n"
        "3. RECOMMENDATION: Justify your final choice.\n"
        "4. IMPLEMENTATION: Complete, production-ready infrastructure definition with "
        "security hardening (IAM least-privilege, encryption at rest, VPC isolation)."
    )


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

5. SIZING: Size the specific step or workload in the request, not a whole pipeline. When the context lists
   default resources for comparable tools or pipeline steps (for example nf-core process labels), base your
   estimate on the closest ones and name them. Over-provisioning wastes money: request the smallest CPU and
   memory that run the step reliably.

6. INSTANCE: Do not choose the EC2 instance type yourself. Write <INSTANCE> wherever the instance type goes;
   it is filled in from the AWS price catalog as the cheapest instance that fits your SIZING line.

7. End your response with one line of exactly this form: SIZING: cpus=<integer> memory_gb=<number>
"""

async def call_cloud_agent(user_input: str, strategy: str = "cot", depth: int = 1, agent_registry: str = "", user_profile_str: str = "") -> tuple[str, dict]:
    _zero = {"input": 0, "output": 0}
    model = _get_model()
    if not model: return "[Cloud Agent - Mock] No API Key.", _zero

    try:
        # Search with the task itself; the upstream context that engine.py prepends would dominate the embedding.
        retrieved_context = _query_vector_db(f"{task_text(user_input)} {entities(user_input)}".strip(), n_results=8)
        
        prompt = CLOUD_PROMPT.format(
            subtask=user_input,
            context=retrieved_context if retrieved_context else "No local cloud data found. Use best practices.",
            profile_guideline=user_profile_str if user_profile_str else "Balance cost and performance."
        )
        prompt += _depth_reasoning_prefix(depth)

        # Generate the infrastructure code
        response = await asyncio.to_thread(model.generate_content, prompt)
        output_text = _attach_instance(response.text.strip(), user_input)
        _usage = getattr(response, "usage_metadata", None)
        _agent_tokens = {
            "input":  getattr(_usage, "prompt_token_count", 0) or 0,
            "output": getattr(_usage, "candidates_token_count", 0) or 0,
        }
        
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
        return final_output, _agent_tokens

    except Exception as e:
        return f"[Cloud Agent Error] {e}", {"input": 0, "output": 0}