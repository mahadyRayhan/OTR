import os
import asyncio
import google.generativeai as genai

async def call_code_agent(user_input: str) -> str:
    """Uses Gemini to actually write Python scripts for bioinformatics."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Fallback if no API key
    if not api_key or api_key == "your_google_api_key_here":
        return "[Code Agent - Mock] Preparing to write code for: " + user_input

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert Bioinformatics Software Engineer. 
        Write a clean, documented Python script to fulfill this request: "{user_input}"
        Return ONLY the Python code inside markdown blocks. Include inline comments explaining how it works.
        """
        
        # Run asynchronously so it doesn't block other agents!
        res = await asyncio.to_thread(model.generate_content, prompt)
        return f"[Code Agent]\n{res.text.strip()}"
        
    except Exception as e:
        return f"[Code Agent Error] Failed to generate code: {e}"