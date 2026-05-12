# MUST BE AT THE ABSOLUTE TOP
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🚀 ADD THIS TO FORCE LOAD THE API KEYS
from dotenv import load_dotenv
load_dotenv()

import asyncio
from rag_agent.knowledge_base import KnowledgeBase
from rag_agent.agent import call_rag_agent

async def test_rag():
    print("⏳ Loading Knowledge Base...")
    kb = KnowledgeBase()
    kb.load()
    
    query = "Explain the Leiden community detection algorithm for single-cell RNA-seq."
    
    print(f"\n🔍 Searching Vector DB for: '{query}'")
    papers = await kb.search_publications(query)
    
    if not papers:
        print("❌ FAISS returned 0 papers above the confidence threshold.")
        print("💡 FIX: Try lowering 'confidence_threshold' in your config.py or .env file.")
    else:
        print(f"✅ Found {len(papers)} papers!")
        for i, p in enumerate(papers):
            print(f"  [{i+1}] Score {p.score:.2f} | {p.title[:60]}...")
            
    print("\n🤖 Testing Full RAG Agent output...")
    # NOTE: Passing the initialized kb here!
    result = await call_rag_agent(kb=kb, user_input=query, strategy="cot", depth=1)
    
    print("\n================ RAG OUTPUT ================")
    print(f"SOURCES USED: {result.get('source_used')}")
    print("--------------------------------------------")
    print(result.get("answer")[:500] + "...\n[TRUNCATED FOR LENGTH]")
    print("============================================")

if __name__ == "__main__":
    asyncio.run(test_rag())