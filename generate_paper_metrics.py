import json
import os
from collections import defaultdict

def generate_metrics(json_path="experiment_telemetry.json"):
    if not os.path.exists(json_path):
        print(f"Data file not found.")
        return

    with open(json_path, "r") as f:
        logs = json.load(f)

    # Metric 1: The Sweet Spot Data (Utilization vs Node Quality)
    # -----------------------------------------------------------
    print("\n" + "="*50)
    print("📈 GRAPH 1 DATA: UTILIZATION vs NODE QUALITY")
    print("="*50)
    print("Use this to draw a Scatter Plot or Bar Chart showing the Inverted U-Curve")
    
    bins = {"0-20%": [], "20-40%": [], "40-60%":[], "60-80%": [], "80-100%":[]}
    
    for run in logs:
        node_qualities = {n["node_id"]: n["quality_score"] for n in run.get("got_dag_nodes",[]) if n.get("quality_score") is not None}
        for edge in run.get("communication_edges",[]):
            util = edge.get("utilization_score")
            qual = node_qualities.get(edge.get("to_id", edge.get("to_node_id")), 0.0)
            if util is not None and qual > 0:
                if util <= 0.2: bins["0-20%"].append(qual)
                elif util <= 0.4: bins["20-40%"].append(qual)
                elif util <= 0.6: bins["40-60%"].append(qual)
                elif util <= 0.8: bins["60-80%"].append(qual)
                else: bins["80-100%"].append(qual)

    for b, scores in bins.items():
        avg = sum(scores)/len(scores) if scores else 0
        print(f"Utilization {b} -> Avg Node Quality: {avg:.2f}/1.0 (Count: {len(scores)})")

    # Metric 2: Edge Fidelity (Relevance & Token Overhead)
    # -----------------------------------------------------------
    print("\n" + "="*50)
    print("📊 GRAPH 2 DATA: AGENT-PAIR COMMUNICATION MATRIX")
    print("="*50)
    
    edge_stats = defaultdict(lambda: {"rel": [], "util": [], "tok":[]})
    
    for run in logs:
        if run.get("ablation_mode") != "adaptive": continue # Only measure adaptive edges
        for edge in run.get("communication_edges",[]):
            if not edge.get("from_node_ids"): continue
            
            # Find sender
            sender = "UNKNOWN"
            for n in run.get("got_dag_nodes", []):
                if n["node_id"] in edge["from_node_ids"]:
                    sender = n["agent"]
                    break
            
            pair = f"{sender.upper()} -> {edge.get('to_agent', 'UNKNOWN').upper()}"
            edge_stats[pair]["rel"].append(edge.get("context_relevance", 0))
            edge_stats[pair]["util"].append(edge.get("utilization_score", 0))
            edge_stats[pair]["tok"].append(edge.get("context_token_count", 0))

    print(f"{'Edge Type':<15} | {'Avg Relevance':<15} | {'Avg Utilization':<15} | {'Tokens Passed':<15}")
    print("-"*65)
    for pair, stats in edge_stats.items():
        rel = sum(stats["rel"])/len(stats["rel"]) if stats["rel"] else 0
        util = sum(stats["util"])/len(stats["util"]) if stats["util"] else 0
        tok = sum(stats["tok"])/len(stats["tok"]) if stats["tok"] else 0
        print(f"{pair:<15} | {rel*100:13.1f}% | {util*100:13.1f}% | {tok:13.0f}")

    print("\n" + "="*50)
    print("💡 TO GET ABLATION DATA (No Comm vs Raw vs Adaptive):")
    print("Use the 'run_ablation_study.py' script. It calculates Information Gain (ΔQ).")
    print("="*50)

if __name__ == "__main__":
    generate_metrics()