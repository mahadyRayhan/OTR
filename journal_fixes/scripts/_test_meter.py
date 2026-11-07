"""Key check: one tiny call per configured model, counted by the usage meter."""
import os, sys, time
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv; load_dotenv(".env")
from orchestrator import usage_meter; usage_meter.install()
import gemini_client as genai
import config
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
models = [config.Config.GENERATION_MODEL, *config.Config.EVAL_JUDGES, config.Config.STRONG_BASELINE_MODEL]
for m in models:
    t0 = time.time()
    txt = genai.GenerativeModel(m).generate_content("Reply with the single word OK").text.strip()
    print(f"{m:32s} -> {txt[:20]!r}  {time.time() - t0:.1f}s")
print(usage_meter.snapshot())
