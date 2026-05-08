import json

with open("experiment_telemetry.json", "r") as f:
    data = json.load(f)

# the file might be a list of runs, we'll check the last one
run_data = data[-1] if isinstance(data, list) else data

print("Analyzing experiment_telemetry.json...")
print(f"Timestamp: {run_data.get('timestamp')}")
print(f"User Query: {run_data.get('user_query')}")

nodes = run_data.get("got_dag_nodes", [])
rag_nodes = [n for n in nodes if n.get("agent") == "rag"]
code_nodes = [n for n in nodes if n.get("agent") == "code"]

print(f"\nRAG Nodes: {len(rag_nodes)}")
for node in rag_nodes:
    ans = node.get("result", "")
    score = node.get("quality_score", "N/A")
    failure = node.get("failure_mode", "None")
    ans_trunc = ans[:150].replace('\n', ' ') + "..." if ans else "None"
    print(f"  - [{node.get('node_id')}] (Score: {score}, Failure: {failure}) Query: {node.get('query')} \n    -> Result: {ans_trunc}")

print(f"\nCODE Nodes: {len(code_nodes)}")
for node in code_nodes:
    ans = node.get("result", "")
    score = node.get("quality_score", "N/A")
    failure = node.get("failure_mode", "None")
    ans_trunc = ans[:150].replace('\n', ' ') + "..." if ans else "None"
    print(f"  - [{node.get('node_id')}] (Score: {score}, Failure: {failure}) Query: {node.get('query')} \n    -> Result: {ans_trunc}")

print("\nCommunication Metrics:")
comm_metrics = run_data.get("communication_metrics", [])
for metric in comm_metrics:
    print(f"  - From {metric.get('source_node')} to {metric.get('target_node')} ({metric.get('target_agent')}):")
    print(f"    Relevance: {metric.get('relevance_score')}, Utilized: {metric.get('utilization_score')}")
    print(f"    Feedback: {metric.get('evaluator_feedback', '')[:100]}...")

