import json
import os
import matplotlib.pyplot as plt
import seaborn as plt_sns
import pandas as pd
import numpy as np

# Set global academic plotting style
plt_sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def load_data(json_path="experiment_telemetry.json"):
    if not os.path.exists(json_path):
        print("Telemetry file not found.")
        return []
    with open(json_path, "r") as f:
        return json.load(f)

# =====================================================================
# Graph 1: The Self-Healing Trajectory (Before & After Scores)
# =====================================================================
def plot_self_healing_trajectory(logs):
    """Plots a paired bar chart showing the Initial Score vs Final Score after Retries."""
    initial_scores = []
    final_scores = []
    labels = []

    for i, log in enumerate(logs):
        if log.get("ablation_mode") != "adaptive": continue
        
        history = log.get("global_score_history", [])
        if not history: continue
        
        initial_scores.append(history[0])
        final_scores.append(history[-1])
        labels.append(f"Q{i+1}")

    if not initial_scores: return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, initial_scores, width, label='Initial Draft (Pre-Healing)', color='salmon')
    ax.bar(x + width/2, final_scores, width, label='Final Output (Post-Healing)', color='mediumseagreen')

    ax.set_ylabel('Global Task Score (Out of 10)')
    ax.set_title('System Resilience: Autonomous Self-Healing via GoT Retries')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=8, color='gray', linestyle='--', label='Passing Threshold (8.0)')
    ax.legend()
    ax.set_ylim(0, 10.5)

    plt.tight_layout()
    plt.savefig("eval_1_self_healing_trajectory.png", dpi=300)
    print("✅ Generated: eval_1_self_healing_trajectory.png")

# =====================================================================
# Graph 2: The "Sweet Spot" (Context Utilization vs Node Quality)
# =====================================================================
def plot_context_sensitivity(logs):
    """Plots the inverted U-curve proving the 'Over-Constraint' phenomenon."""
    utilizations = []
    qualities = []

    for log in logs:
        if log.get("ablation_mode") != "adaptive": continue
        
        node_qualities = {n["node_id"]: n["quality_score"] for n in log.get("got_dag_nodes", []) if n.get("quality_score")}
        
        for edge in log.get("communication_edges", []):
            util = edge.get("utilization_score")
            qual = node_qualities.get(edge.get("to_id"), 0.0)
            if util is not None and qual > 0:
                utilizations.append(util * 100) # Convert to percentage
                qualities.append(qual)

    if not utilizations: return

    plt.figure(figsize=(8, 6))
    plt_sns.regplot(x=utilizations, y=qualities, order=2, scatter_kws={"s": 100, "alpha": 0.6}, line_kws={"color": "red"})
    
    plt.title('The Context Sensitivity Curve (Sweet Spot)')
    plt.xlabel('Agent-to-Agent Context Utilization (%)')
    plt.ylabel('Node Quality Score (0.0 to 1.0)')
    plt.xlim(-5, 105)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("eval_2_context_sensitivity.png", dpi=300)
    print("✅ Generated: eval_2_context_sensitivity.png")

# =====================================================================
# Graph 3: Causal Ablation Study (Task Success Rate)
# =====================================================================
def plot_ablation_success_rate(logs):
    """Plots the Task Success Rate (Score >= 8) across the 3 communication modes."""
    modes_data = {"none": [], "raw": [], "adaptive": []}
    for log in logs:
        m = log.get("ablation_mode", "adaptive")
        if m in modes_data:
            modes_data[m].append(log.get("final_judge_score", 0))

    rates = []
    labels = ['Isolated (NONE)', 'Blind Passing (RAW)', 'Filtered Injection (ADAPTIVE)']
    colors = ['gray', 'steelblue', 'darkorange']

    for mode in ["none", "raw", "adaptive"]:
        scores = modes_data[mode]
        if not scores: 
            rates.append(0)
            continue
        successes = sum(1 for s in scores if s >= 8.0)
        rates.append((successes / len(scores)) * 100)

    if sum(rates) == 0: return

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, rates, color=colors)
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Task Success Rate (%)')
    plt.title('Causal Ablation: Impact of Communication on Task Success')
    plt.ylim(0, max(rates) + 15 if rates else 100)

    plt.tight_layout()
    plt.savefig("eval_3_ablation_success_rate.png", dpi=300)
    print("✅ Generated: eval_3_ablation_success_rate.png")

# =====================================================================
# Graph 4: Pareto Frontier (Cost vs. Quality)
# =====================================================================
def plot_pareto_frontier(logs):
    """Plots the efficiency trade-off between Token Cost and Final Quality."""
    modes_data = {"none": {"scores": [], "tokens": []}, 
                  "raw": {"scores": [], "tokens": []}, 
                  "adaptive": {"scores": [], "tokens": []}}
    
    for log in logs:
        m = log.get("ablation_mode", "adaptive")
        if m not in modes_data: continue
        
        modes_data[m]["scores"].append(log.get("final_judge_score", 0))
        
        edge_tokens = sum(e.get("context_token_count", 0) for e in log.get("communication_edges", []))
        if m == "none": edge_tokens = 0 # Fix for NONE mode
        modes_data[m]["tokens"].append(edge_tokens)

    plot_data = []
    for mode, data in modes_data.items():
        if not data["scores"]: continue
        avg_score = sum(data["scores"]) / len(data["scores"])
        avg_tokens = sum(data["tokens"]) / len(data["tokens"])
        plot_data.append({"Mode": mode.upper(), "Cost": avg_tokens, "Quality": avg_score})

    if not plot_data: return
    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(8, 6))
    plt_sns.scatterplot(data=df, x='Cost', y='Quality', hue='Mode', s=200, palette='Set1', style='Mode', markers=["o", "s", "D"])
    
    # Draw a line connecting them to show the asymptotic curve
    df_sorted = df.sort_values(by="Cost")
    plt.plot(df_sorted['Cost'], df_sorted['Quality'], color='gray', linestyle='--')

    plt.title('Pareto Frontier: Communication Cost vs. Reasoning Quality')
    plt.xlabel('Token Overhead per Query')
    plt.ylabel('Mean Global Task Score (0 to 10)')
    
    plt.tight_layout()
    plt.savefig("eval_4_pareto_frontier.png", dpi=300)
    print("✅ Generated: eval_4_pareto_frontier.png")


if __name__ == "__main__":
    print("Generating Academic Evaluation Graphs...")
    telemetry_data = load_data()
    
    if telemetry_data:
        plot_self_healing_trajectory(telemetry_data)
        plot_context_sensitivity(telemetry_data)
        plot_ablation_success_rate(telemetry_data)
        plot_pareto_frontier(telemetry_data)
        print("\n🎉 All 4 PNG graphs have been generated and saved to your directory!")