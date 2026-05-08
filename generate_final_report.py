import json
import os
import statistics

def generate_report(json_path="experiment_telemetry.json"):
    if not os.path.exists(json_path):
        print("Telemetry file not found.")
        return

    with open(json_path, "r") as f:
        logs = json.load(f)

    if not logs: return

    # --- Calculations ---
    scores = [r.get("final_judge_score", 0) for r in logs]
    successes = sum(1 for s in scores if s >= 8.0)
    success_rate = (successes / len(scores)) * 100
    mean_gts = statistics.mean(scores)
    std_gts = statistics.stdev(scores) if len(scores) > 1 else 0.0

    total_tokens, latencies, all_crs, dsr_passes, dsr_total = 0, [], [], 0, 0
    recoveries, failure_events = 0, 0

    for r in logs:
        latencies.append(r.get("latency_seconds", 0))
        for edge in r.get("communication_edges", []):
            total_tokens += edge.get("context_token_count", 0)
            all_crs.append(edge.get("context_relevance", 0))
            if edge.get("dependency_accuracy") is not None:
                dsr_total += 1
                if edge.get("dependency_accuracy"): dsr_passes += 1
        
        # Recovery
        had_failure = any((n.get("quality_score") or 1.0) < 0.75 for n in r.get("got_dag_nodes", []))
        if had_failure:
            failure_events += 1
            if r.get("final_judge_score", 0) >= 8.0:
                recoveries += 1

    avg_tokens = total_tokens / len(logs)
    avg_latency = statistics.mean(latencies)
    avg_crs = (sum(all_crs) / len(all_crs)) * 100 if all_crs else 0
    dsr_rate = (dsr_passes / dsr_total * 100) if dsr_total > 0 else 0.0
    rec_rate = (recoveries / failure_events * 100) if failure_events > 0 else 0.0

    print("\n" + "="*80)
    print("🏆 FINAL SYSTEM PERFORMANCE REPORT (DYNAMIC GATEWAY ARCHITECTURE)")
    print("="*80)
    
    print("\n[ 1. END-TO-END OUTCOMES ]")
    print(f"  - Overall Task Success Rate (Score >= 8):  {success_rate:.1f}%")
    print(f"  - Mean Global Task Score (GTS):            {mean_gts:.2f} / 10.0 (±{std_gts:.2f})")

    print("\n[ 2. AGENT COMMUNICATION HEALTH ]")
    print(f"  - Average Context Relevance:               {avg_crs:.1f}%")
    print(f"  - Dependency Satisfaction Rate (DSR):      {dsr_rate:.1f}% (No Placeholder Leaks)")

    print("\n[ 3. SYSTEM EFFICIENCY & ROBUSTNESS ]")
    print(f"  - Average Token Overhead per Query:        {avg_tokens:.0f} tokens")
    print(f"  - Average Latency per Query:               {avg_latency:.1f} seconds")
    print(f"  - Self-Healing Recovery Rate:              {rec_rate:.1f}% ({recoveries} out of {failure_events} fatal errors fixed)")

    print("\n" + "="*80)
    print("🎯 THE 'SWEET SPOT' HYPOTHESIS TEST")
    print("="*80)
    
    bins = {"0-20%": [], "20-40%": [], "40-60%": [], "60-80%": [], "80-100%": []}
    
    for run in logs:
        node_qualities = {n["node_id"]: n["quality_score"] for n in run.get("got_dag_nodes",[]) if n.get("quality_score") is not None}
        for edge in run.get("communication_edges",[]):
            util = edge.get("utilization_score")
            qual = node_qualities.get(edge.get("to_id"), 0.0)
            if util is not None and qual > 0:
                if util <= 0.2: bins["0-20%"].append(qual)
                elif util <= 0.4: bins["20-40%"].append(qual)
                elif util <= 0.6: bins["40-60%"].append(qual)
                elif util <= 0.8: bins["60-80%"].append(qual)
                else: bins["80-100%"].append(qual)

    for b, scores in bins.items():
        avg = sum(scores)/len(scores) if scores else 0
        print(f"Utilization {b:<10} -> Mean NQS: {avg:.2f}/1.0 (Based on {len(scores)} edges)")
    print("================================================================================\n")

if __name__ == "__main__":
    generate_report()