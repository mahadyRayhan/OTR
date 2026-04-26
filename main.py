# import sys
# import asyncio
# from dotenv import load_dotenv

# load_dotenv()
# from orchestrator.engine import OrchestratorEngine

# async def main():
#     print("==================================================")
#     print("🚀 SUPREME MULTI-AGENT ORCHESTRATOR INITIALIZED")
#     print("Type 'benchmark' to run the comparative evaluation.")
#     print("Type 'exit' to stop.")
#     print("==================================================\n")

#     engine = OrchestratorEngine(dag_name="topological_dag")

#     # 5 Hardcore questions designed to break basic decomposers
#     # test_questions = [
#     #     # Q1: Multi-Domain Parallelism (Tests if it can trigger 3 agents simultaneously without getting confused)
#     #     "Write a Python script using Biopython to align RNA-seq data, find 2 experts on transcriptomics, and explain what AWS EC2 is.",
        
#     #     # Q2: Implicit Coreference (Tests if the Neural phase can resolve "both of them")
#     #     "Compare the accuracy of AlphaFold and Rosetta, and find 3 experts who have published papers on both of them.",
        
#     #     # Q3: Strict Sequential Dependency (Tests if the DAG builder waits for Layer 1 before Layer 2)
#     #     "First, find out which cloud provider developed the 'SageMaker' service. Then, write a Python deployment script for that specific provider.",
        
#     #     # Q4: The Multi-Hop Constraint (Tests deep dependencies: RAG -> Code -> Scholar)
#     #     "Identify the most common text format for unaligned DNA sequences. Write a Python script to parse it, and find 2 experts who develop tools for that specific format.",
        
#     #     # Q5: The "Kitchen Sink" Stress Test (Temporal flow + Coreference + Multi-domain)
#     #     "Explain the central dogma of molecular biology. Based on the translation step, write a Python script that translates DNA to Amino Acids, and then deploy it to an AWS Lambda function."
#     # ]

#     test_questions = [
#         # Q1: High Conceptual (Definition/Theory)
#         "Explain the central dogma of molecular biology and the general mechanisms of translation.",
        
#         # Q2: Analytical Shift (Trade-offs, Comparisons, Experts)
#         "How does standard alignment compare to splice-aware alignment? What are the trade-offs, and can you find 2 experts who published on splice-aware alignment?",
        
#         # Q3: Sudden Procedural/Code Override (Implementation)
#         "Okay, enough theory. Write a Python script using Biopython to translate a DNA sequence into Amino Acids.",
        
#         # Q4: Procedural Dependency (Code + Cloud integration)
#         "Now write a Python script to align RNA-seq data, and then give me instructions on how to deploy that specific script to an AWS Lambda function.",
        
#         # Q5: The Kitchen Sink (Everything combined)
#         "Compare BWA and Bowtie2, write a script to run both, and find 2 experts on transcriptomics."
#     ]

#     while True:
#         try:
#             user_input = await asyncio.to_thread(input, "\n🧑‍💻 You: ")
            
#             if user_input.lower() in ['exit', 'quit']:
#                 print("Shutting down system. Goodbye!")
#                 sys.exit(0)
            
#             if not user_input.strip(): continue

#             # 🚀 TRIGGER BENCHMARK
#             if user_input.lower() == "benchmark":
#                 await engine.run_benchmark(test_questions)
#                 continue
            
#             # Standard Interactive Run
#             output = await engine.run(user_input)
#             print("\n==================================================")
#             print("🤖 SYSTEM SYNTHESIS:")
#             print("==================================================")
#             print(output["final_answer"])
#             print("\n--------------------------------------------------")
#             print(f"🎖️  SYSTEM JUDGE: {output['judge_score']}")
#             print("==================================================")
            
#         except KeyboardInterrupt:
#             print("\nShutting down system. Goodbye!")
#             sys.exit(0)

# if __name__ == "__main__":
#     asyncio.run(main())

import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import config
from orchestrator.engine import OrchestratorEngine

async def main():
    print("==================================================")
    print("🚀 SUPREME MULTI-AGENT ORCHESTRATOR INITIALIZED")
    print(f"   Mode: {config.Config.RUN_MODE.upper()}")
    print(f"   Profiler Active: {config.Config.USE_PROFILER}")
    print("==================================================\n")

    engine = OrchestratorEngine()

    test_questions = [
        "Explain the central dogma of molecular biology and the general mechanisms of translation.",
        "How does standard alignment compare to splice-aware alignment? What are the trade-offs, and can you find 2 experts who published on splice-aware alignment?",
        "Okay, enough theory. Write a Python script using Biopython to translate a DNA sequence into Amino Acids.",
        "Now write a Python script to align RNA-seq data, and then give me instructions on how to deploy that specific script to an AWS Lambda function.",
        "Compare BWA and Bowtie2, write a script to run both, and find 2 experts on transcriptomics."
    ]

    # 🚀 If config is set to benchmark, just run it and exit!
    if config.Config.RUN_MODE == "benchmark":
        await engine.run_benchmark(test_questions)
        sys.exit(0)

    # 🚀 Otherwise, start Interactive Chat Loop
    print("Type 'exit' to stop.")
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\n🧑‍💻 You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down system. Goodbye!")
                sys.exit(0)
            
            if not user_input.strip(): continue

            output = await engine.run(user_input)
            print("\n==================================================")
            print("🤖 SYSTEM SYNTHESIS:")
            print("==================================================")
            print(output["final_answer"])
            print("\n--------------------------------------------------")
            print(f"🎖️  SYSTEM JUDGE: {output['judge_score']}")
            print("==================================================")
            
        except KeyboardInterrupt:
            print("\nShutting down system. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())