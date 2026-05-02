import json
import os
import statistics
from collections import defaultdict

def generate_paper_metrics(json_path="experiment_telemetry.json"):
    if not os.path.exists(json_path):
        print("Telemetry file not found. Run the benchmark first.")
        return

    with open(json_path, "r") as f:
        logs = json.load(f)

    queries = defaultdict(dict)
    modes_data = {"none": [], "raw": [], "adaptive":[]}
    
    for log in logs:
        m = log.get("ablation_mode", "adaptive")
        q = log.get("user_query", "")
        queries[q][m] = log
        modes_data[m].append(log)

    baseline_runs = modes_data["none"]
    baseline_gts = sum([r.get("final_judge_score", 0) for r in baseline_runs]) / max(1, len(baseline_runs))

    print("\n" + "="*80)
    print("🏆 LAYER 1: TASK OUTCOME (End-to-End Effectiveness)")
    print("="*80)
    print(f"{'Protocol':<15} | {'Task Success Rate':<20} | {'Mean GTS ± StdDev':<20} | {'Pairwise Win Rate'}")
    print("-" * 80)
    
    for mode in["none", "raw", "adaptive"]:
        runs = modes_data[mode]
        if not runs: continue
        
        # 1. Mean and Std of Judge Score (Robustness)
        scores =[r.get("final_judge_score", 0) for r in runs]
        mean_gts = statistics.mean(scores) if scores else 0
        std_gts = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        # 2. Task Success Rate (Score >= 8.0)
        successes = sum(1 for s in scores if s >= 8.0)
        success_rate = (successes / len(scores) * 100) if scores else 0
        
        # 3. Pairwise Win Rate (vs None)
        wins, total_pairs = 0, 0
        for q, m_dict in queries.items():
            if mode in m_dict and "none" in m_dict:
                total_pairs += 1
                if m_dict[mode].get("final_judge_score", 0) > m_dict["none"].get("final_judge_score", 0):
                    wins += 1
        win_rate = (wins / total_pairs * 100) if total_pairs > 0 else 0.0

        print(f"{mode.upper():<15} | {success_rate:>17.1f}% | {mean_gts:>10.2f} ± {std_gts:<6.2f} | {win_rate:>15.1f}%")

    print("\n" + "="*80)
    print("📡 LAYER 2: COMMUNICATION QUALITY (Agent-to-Agent)")
    print("="*80)
    print(f"{'Protocol':<15} | {'Info Gain (ΔQ)':<15} | {'Context Relevance':<20} | {'Dependency Satisfaction (DSR)'}")
    print("-" * 80)

    for mode in ["none", "raw", "adaptive"]:
        runs = modes_data[mode]
        if not runs: continue
        
        mean_gts = statistics.mean([r.get("final_judge_score", 0) for r in runs]) if runs else 0
        delta_q = mean_gts - baseline_gts if mode != "none" else 0.0

        all_crs =[]
        dsr_passes, dsr_total = 0, 0

        for r in runs:
            for edge in r.get("communication_edges",[]):
                all_crs.append(edge.get("context_relevance", 0))
                # 4. Dependency Satisfaction Rate (Did it resolve placeholders?)
                if edge.get("dependency_accuracy") is not None:
                    dsr_total += 1
                    if edge.get("dependency_accuracy"):
                        dsr_passes += 1

        avg_crs = (sum(all_crs) / len(all_crs)) * 100 if all_crs else 0
        dsr_rate = (dsr_passes / dsr_total * 100) if dsr_total > 0 else 0.0

        print(f"{mode.upper():<15} | {delta_q:>+15.2f} | {avg_crs:>18.1f}% | {dsr_rate:>26.1f}%")

    print("\n" + "="*80)
    print("⚙️ LAYER 3: SYSTEM COST & ROBUSTNESS")
    print("="*80)
    print(f"{'Protocol':<15} | {'Token Cost':<12} | {'Comm Efficiency':<18} | {'Latency':<10} | {'Recovery Rate'}")
    print("-" * 80)

    pareto_data = []

    for mode in ["none", "raw", "adaptive"]:
        runs = modes_data[mode]
        if not runs: continue
        
        mean_gts = statistics.mean([r.get("final_judge_score", 0) for r in runs]) if runs else 0
        delta_q = mean_gts - baseline_gts

        total_tokens, latencies = 0,[]
        recoveries, failure_events = 0, 0

        for r in runs:
            latencies.append(r.get("latency_seconds", 0))
            for edge in r.get("communication_edges",[]):
                total_tokens += edge.get("context_token_count", 0)
            
            # 5. Recovery Rate (Failed mid-run, but graph recovered to >= 8.0)
            had_failure = any((n.get("quality_score") or 1.0) < 0.75 for n in r.get("got_dag_nodes",[]))
            if had_failure:
                failure_events += 1
                if r.get("final_judge_score", 0) >= 8.0:
                    recoveries += 1

        avg_tokens = total_tokens / len(runs)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        ce = (delta_q / (avg_tokens / 1000)) if avg_tokens > 0 else 0.0
        rec_rate = (recoveries / failure_events * 100) if failure_events > 0 else 100.0

        # Save for Pareto plotting
        pareto_data.append((mode.upper(), avg_tokens, mean_gts))

        print(f"{mode.upper():<15} | {avg_tokens:<12.0f} | {ce:<18.3f} | {avg_latency:<7.1f}s | {rec_rate:.1f}%")

    # 6. Pareto Frontier Data
    print("\n" + "="*80)
    print("📈 PARETO FRONTIER DATA: QUALITY vs COST")
    print("="*80)
    print("Plot these coordinates (X = Cost/Tokens, Y = Quality/GTS) to show efficiency domination:")
    for data in pareto_data:
        print(f"  {data[0]:<10} -> X (Tokens): {data[1]:<6.0f} | Y (Score): {data[2]:.2f}")

    # 7. Hypothesis Test: Context Sensitivity Curve
    print("\n" + "="*80)
    print("🎯 HYPOTHESIS TEST: CONTEXT SENSITIVITY (SWEET SPOT)")
    print("="*80)
    
    bins = {"0-20%": [], "20-40%":[], "40-60%": [], "60-80%": [], "80-100%":[]}
    
    for run in modes_data["adaptive"]:
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

if __name__ == "__main__":
    generate_paper_metrics()