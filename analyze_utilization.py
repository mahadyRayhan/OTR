import json
import os

def analyze_sweet_spot(json_path="experiment_telemetry.json"): # 🚀 Fixed filename
    if not os.path.exists(json_path):
        print(f"File '{json_path}' not found. Run your orchestrator first to generate data.")
        return

    with open(json_path, "r") as f:
        logs = json.load(f)

    # 1. Gather the Data
    data_points = []
    
    for run in logs:
        # 🚀 Step A: Create a lookup dictionary for TRUE node quality scores
        # We get this from the DAG dump, not the edge logger!
        node_qualities = {}
        for node in run.get("got_dag_nodes", []):
            if node.get("quality_score") is not None:
                node_qualities[node["node_id"]] = node["quality_score"]

        # 🚀 Step B: Match the utilization to the true quality score
        for edge in run.get("communication_edges", []):
            node_id = edge.get("to_id")
            util = edge.get("utilization_score")
            
            # Look up the true quality score using the node_id
            qual = node_qualities.get(node_id, 0.0)

            if util is not None and qual > 0.0:
                data_points.append((util, qual))

    if len(data_points) < 3:
        print(f"Found {len(data_points)} valid data points. We need at least 3!")
        print("Run a few more questions in the orchestrator first.")
        return

    # 2. Bin the data to see the curve manually
    bins = {"0-20%": [], "20-40%": [], "40-60%": [], "60-80%": [], "80-100%": []}
    
    for util, qual in data_points:
        if util <= 0.2: bins["0-20%"].append(qual)
        elif util <= 0.4: bins["20-40%"].append(qual)
        elif util <= 0.6: bins["40-60%"].append(qual)
        elif util <= 0.8: bins["60-80%"].append(qual)
        else: bins["80-100%"].append(qual)

    print("\n📊 CONTEXT UTILIZATION vs. QUALITY SCORE")
    print("========================================")
    best_bin = ""
    highest_avg = 0.0

    for b, scores in bins.items():
        if not scores:
            print(f" {b} Utilization -> No data yet.")
            continue
        avg_qual = sum(scores) / len(scores)
        print(f" {b} Utilization -> Avg Quality: {avg_qual:.2f}/1.0 (based on {len(scores)} runs)")
        
        if avg_qual > highest_avg:
            highest_avg = avg_qual
            best_bin = b

    print("========================================")
    print(f"🎯 MATHEMATICAL SWEET SPOT: The data suggests aiming for {best_bin} utilization.")

if __name__ == "__main__":
    analyze_sweet_spot()