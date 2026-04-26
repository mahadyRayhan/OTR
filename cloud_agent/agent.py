import os
import asyncio
import google.generativeai as genai

async def call_cloud_agent(user_input: str, strategy: str = "cot", depth: int = 1, agent_registry: str = "") -> str:
    """Uses Gemini to generate deployment scripts (Terraform/YAML/etc)"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        return f"[Cloud Agent] called with Strategy: {strategy}, Depth: {depth}. Analyzing infrastructure/deployment request: '{user_input}'"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        You are an expert DevOps and Cloud Architect.
        Use {strategy.upper()} strategy, Depth: {depth}.
        Registry: {agent_registry}
        
        Write the IaC (Terraform, CloudFormation, Kubernetes YAML, etc) or deployment script required for: "{user_input}"
        Provide the code in format blocks.
        """
        res = await asyncio.to_thread(model.generate_content, prompt)
        return f"[Cloud Agent]\n{res.text.strip()}"
    except Exception as e:
        return f"[Cloud Agent Error] Failed: {e}"