# import json
# import os
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import pandas as pd
# import numpy as np
# from collections import defaultdict

# # Set global academic plotting style
# plt_sns.set_theme(style="whitegrid", palette="muted")
# plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 300})

# def load_data(json_path="experiment_telemetry.json"):
#     if not os.path.exists(json_path):
#         print("Telemetry file not found.")
#         return []
#     with open(json_path, "r") as f:
#         return json.load(f)

# # =====================================================================
# # MACRO LEVEL: SYSTEM PERFORMANCE & ABLATION
# # =====================================================================

# def plot_ablation_success_rate(logs):
#     """Graph 1: Overall Task Success Rate (Score >= 8) across ablation modes."""
#     modes_data = {"none": [], "raw": [], "adaptive": []}
#     for log in logs:
#         m = log.get("ablation_mode", "adaptive")
#         if m in modes_data:
#             modes_data[m].append(log.get("final_judge_score", 0))

#     rates = []
#     labels = ['Isolated (NONE)', 'Raw Passing (RAW)', 'CUGCR (ADAPTIVE)']
#     colors = ['gray', 'steelblue', 'darkorange']

#     for mode in ["none", "raw", "adaptive"]:
#         scores = modes_data[mode]
#         if not scores: 
#             rates.append(0)
#             continue
#         successes = sum(1 for s in scores if s >= 8.0)
#         rates.append((successes / len(scores)) * 100)

#     if sum(rates) == 0: return

#     plt.figure(figsize=(8, 6))
#     bars = plt.bar(labels, rates, color=colors)
    
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

#     plt.ylabel('Task Success Rate (%)')
#     plt.title('Overall System Performance: Task Success by Protocol')
#     plt.ylim(0, max(rates) + 15 if rates else 100)

#     plt.tight_layout()
#     plt.savefig("fig1_ablation_success_rate.png")
#     print("✅ Generated: fig1_ablation_success_rate.png")

# def plot_mean_task_score(logs):
#     """Graph 2: Mean Global Task Score with Standard Deviation."""
#     modes_data = {"none": [], "raw": [], "adaptive": []}
#     for log in logs:
#         m = log.get("ablation_mode", "adaptive")
#         if m in modes_data:
#             modes_data[m].append(log.get("final_judge_score", 0))

#     means = []
#     stds = []
#     labels = ['NONE', 'RAW', 'ADAPTIVE']
#     colors = ['#B0BEC5', '#4C72B0', '#E1812C']

#     for mode in ["none", "raw", "adaptive"]:
#         scores = modes_data[mode]
#         if not scores:
#             means.append(0); stds.append(0)
#             continue
#         means.append(np.mean(scores))
#         stds.append(np.std(scores) if len(scores) > 1 else 0)

#     if sum(means) == 0: return

#     plt.figure(figsize=(7, 6))
#     x_pos = np.arange(len(labels))
    
#     plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, ecolor='black', capsize=10, color=colors)
#     plt.xticks(x_pos, labels)
#     plt.ylabel('Mean Global Task Score (0-10)')
#     plt.title('End-to-End Task Quality & Variance')
#     plt.ylim(0, 10.5)
    
#     plt.tight_layout()
#     plt.savefig("fig2_mean_global_score.png")
#     print("✅ Generated: fig2_mean_global_score.png")

# def plot_pareto_frontier(logs):
#     """Graph 3: Pareto Trade-off between Token Cost and Final Quality."""
#     modes_data = {"none": {"scores": [], "tokens": []}, 
#                   "raw": {"scores": [], "tokens": []}, 
#                   "adaptive": {"scores": [], "tokens": []}}
    
#     for log in logs:
#         m = log.get("ablation_mode", "adaptive")
#         if m not in modes_data: continue
        
#         modes_data[m]["scores"].append(log.get("final_judge_score", 0))
#         edge_tokens = sum(e.get("context_token_count", 0) for e in log.get("communication_edges", []))
#         if m == "none": edge_tokens = 0 
#         modes_data[m]["tokens"].append(edge_tokens)

#     plot_data = []
#     for mode, data in modes_data.items():
#         if not data["scores"]: continue
#         avg_score = np.mean(data["scores"])
#         avg_tokens = np.mean(data["tokens"])
#         plot_data.append({"Mode": mode.upper(), "Token Cost": avg_tokens, "Quality": avg_score})

#     if not plot_data: return
#     df = pd.DataFrame(plot_data)

#     plt.figure(figsize=(8, 6))
#     plt_sns.scatterplot(data=df, x='Token Cost', y='Quality', hue='Mode', s=200, palette=['gray', 'blue', 'orange'], style='Mode', markers=["o", "s", "D"])
    
#     df_sorted = df.sort_values(by="Token Cost")
#     plt.plot(df_sorted['Token Cost'], df_sorted['Quality'], color='gray', linestyle='--', alpha=0.6)

#     plt.title('The Pareto Frontier: Engineering Cost vs. Reasoning Quality')
#     plt.xlabel('Average Inter-Agent Token Overhead')
#     plt.ylabel('Mean Global Task Score')
    
#     plt.tight_layout()
#     plt.savefig("fig3_pareto_frontier.png")
#     print("✅ Generated: fig3_pareto_frontier.png")


# # =====================================================================
# # MICRO LEVEL: AGENT COMMUNICATION & RESILIENCE
# # =====================================================================

# def plot_sweet_spot(logs):
#     """Graph 4: Context Sensitivity U-Curve (The Sweet Spot)."""
#     utilizations, qualities = [], []
#     for log in logs:
#         node_qualities = {n["node_id"]: n["quality_score"] for n in log.get("got_dag_nodes", []) if n.get("quality_score")}
#         for edge in log.get("communication_edges", []):
#             util = edge.get("utilization_score")
#             qual = node_qualities.get(edge.get("to_id"), 0.0)
#             if util is not None and qual > 0:
#                 utilizations.append(util * 100)
#                 qualities.append(qual)

#     if not utilizations: return
#     plt.figure(figsize=(8, 6))
#     plt_sns.regplot(x=utilizations, y=qualities, order=2, scatter_kws={"s": 100, "alpha": 0.5, "color": "royalblue"}, line_kws={"color": "darkred", "linewidth": 2})
#     plt.title('Agent Cognitive Load: Context Utilization vs. Reasoning Quality')
#     plt.xlabel('Agent-to-Agent Context Utilization (%)')
#     plt.ylabel('Node Quality Score (0 to 1.0)')
#     plt.xlim(-5, 105)
#     plt.ylim(0, 1.1)
#     plt.tight_layout()
#     plt.savefig("fig4_context_sweet_spot.png")
#     print("✅ Generated: fig4_context_sweet_spot.png")

# def plot_agent_pair_heatmap(logs):
#     """Graph 5: Average Context Relevance by Agent Edge."""
#     pair_data = defaultdict(list)
#     for log in logs:
#         for edge in log.get("communication_edges", []):
#             if not edge.get("from_node_ids"): continue
#             sender = next((n["agent"].upper() for n in log.get("got_dag_nodes", []) if n["node_id"] in edge["from_node_ids"]), "UNKNOWN")
#             receiver = edge.get("to_agent", "UNKNOWN").upper()
#             pair_data[f"{sender} → {receiver}"].append(edge.get("context_relevance", 0) * 100)

#     if not pair_data: return
#     df = pd.DataFrame({'Agent Edge': list(pair_data.keys()), 'Relevance (%)': [np.mean(v) for v in pair_data.values()], 'Count': [len(v) for v in pair_data.values()]}).sort_values('Relevance (%)', ascending=False)

#     plt.figure(figsize=(10, 5))
#     bar_plot = plt_sns.barplot(data=df, x='Relevance (%)', y='Agent Edge', palette="viridis")
#     for i, p in enumerate(bar_plot.patches):
#         plt.text(p.get_width() + 2, p.get_y() + p.get_height()/2, f'n={df.iloc[i]["Count"]}', va='center')
#     plt.title('Average Context Relevance by Agent Edge')
#     plt.xlim(0, 110)
#     plt.tight_layout()
#     plt.savefig("fig5_agent_edge_relevance.png")
#     print("✅ Generated: fig5_agent_edge_relevance.png")

# def plot_recovery_trajectory(logs):
#     """Graph 6: System Resilience - Score Improvement via Diagnostician Retry."""
#     initial_scores, final_scores, query_labels = [], [], []
#     count = 1
#     for log in logs:
#         history = log.get("global_score_history", [])
#         if len(history) > 1 and history[0] < 8.0:
#             initial_scores.append(history[0])
#             final_scores.append(history[-1])
#             query_labels.append(f"Q{count}")
#             count += 1

#     if not initial_scores: return
#     x = np.arange(len(query_labels))
#     width = 0.35
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.bar(x - width/2, initial_scores, width, label='Initial Draft', color='lightcoral')
#     ax.bar(x + width/2, final_scores, width, label='Post-Reflexion Retry', color='mediumseagreen')
#     ax.set_ylabel('Global Task Score (Out of 10)')
#     ax.set_title('System Resilience: Score Improvement via Diagnostician Retry')
#     ax.set_xticks(x)
#     ax.set_xticklabels(query_labels)
#     ax.axhline(y=8, color='gray', linestyle='--', label='Passing Threshold')
#     ax.legend()
#     ax.set_ylim(0, 10.5)
#     plt.tight_layout()
#     plt.savefig("fig6_recovery_trajectory.png")
#     print("✅ Generated: fig6_recovery_trajectory.png")

# def plot_token_efficiency(logs):
#     """Graph 7: Distribution of tokens passed to different downstream agents."""
#     df_data = [{"Receiving Agent": e.get("to_agent", "UNKNOWN").upper(), "Tokens Injected": e.get("context_token_count", 0)} for l in logs for e in l.get("communication_edges", []) if e.get("context_token_count", 0) > 0]
#     if not df_data: return
    
#     plt.figure(figsize=(8, 6))
#     plt_sns.boxplot(data=pd.DataFrame(df_data), x='Receiving Agent', y='Tokens Injected', palette="Set2")
#     plt_sns.stripplot(data=pd.DataFrame(df_data), x='Receiving Agent', y='Tokens Injected', color=".25", alpha=0.6, jitter=True)
#     plt.title('Context Overhead: Tokens Injected per Receiving Agent')
#     plt.ylabel('Token Count')
#     plt.tight_layout()
#     plt.savefig("fig7_token_overhead_boxplot.png")
#     print("✅ Generated: fig7_token_overhead_boxplot.png")

# def plot_node_failures(logs):
#     """Graph 8: Distribution of Node Failures across GoT Operators."""
#     failure_counts = {"ROOT": 0, "BRANCH": 0, "REFINE": 0, "AGGREGATE": 0}
#     for log in logs:
#         for node in log.get("got_dag_nodes", []):
#             score = node.get("quality_score")
#             if score is None or score < 0.75:
#                 op = node.get("operator", "ROOT").upper()
#                 if op in failure_counts: failure_counts[op] += 1

#     labels = [k for k, v in failure_counts.items() if v > 0]
#     sizes = [v for v in failure_counts.values() if v > 0]
#     if not sizes: return

#     plt.figure(figsize=(7, 7))
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt_sns.color_palette("pastel"))
#     plt.title('Distribution of Node Failures across GoT Operators')
#     plt.tight_layout()
#     plt.savefig("fig8_operator_failures.png")
#     print("✅ Generated: fig8_operator_failures.png")

# if __name__ == "__main__":
#     print("Initializing Comprehensive Academic Visualization Suite...")
#     data = load_data()
#     if data:
#         plot_ablation_success_rate(data)  # Graph 1
#         plot_mean_task_score(data)        # Graph 2
#         plot_pareto_frontier(data)        # Graph 3
#         plot_sweet_spot(data)             # Graph 4
#         plot_agent_pair_heatmap(data)     # Graph 5
#         plot_recovery_trajectory(data)    # Graph 6
#         plot_token_efficiency(data)       # Graph 7
#         plot_node_failures(data)          # Graph 8
#         print("\n🎯 Complete! All 8 high-res PNG graphs have been saved to your directory.")

# import json
# import os
# import matplotlib.pyplot as plt
# import seaborn as plt_sns
# import pandas as pd
# import numpy as np

# # Set global academic plotting style
# plt_sns.set_theme(style="whitegrid", palette="muted")
# plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 300})

# def load_data(json_path="experiment_telemetry.json"):
#     if not os.path.exists(json_path):
#         print("Telemetry file not found.")
#         return []
#     with open(json_path, "r") as f:
#         return json.load(f)

# # =====================================================================
# # Graph 1: The Context Sensitivity U-Curve (The "Sweet Spot")
# # =====================================================================
# def plot_sweet_spot(logs):
#     utilizations = []
#     qualities = []

#     for log in logs:
#         node_qualities = {n["node_id"]: n["quality_score"] for n in log.get("got_dag_nodes", []) if n.get("quality_score")}
#         for edge in log.get("communication_edges", []):
#             util = edge.get("utilization_score")
#             qual = node_qualities.get(edge.get("to_id"), 0.0)
#             if util is not None and qual > 0:
#                 utilizations.append(util * 100)
#                 qualities.append(qual)

#     if not utilizations: return

#     plt.figure(figsize=(8, 6))
#     plt_sns.regplot(x=utilizations, y=qualities, order=2, scatter_kws={"s": 100, "alpha": 0.5, "color": "royalblue"}, line_kws={"color": "darkred", "linewidth": 2})
#     plt.title('Context Sensitivity: Utilization vs. Reasoning Quality')
#     plt.xlabel('Agent-to-Agent Context Utilization (%)')
#     plt.ylabel('Node Quality Score (NQS)')
#     plt.xlim(-5, 105)
#     plt.ylim(-0.1, 1.1)
#     plt.tight_layout()
#     plt.savefig("fig1_context_sweet_spot.png")
#     print("✅ Generated: fig1_context_sweet_spot.png")

# # =====================================================================
# # Graph 2: Gateway Behavior (RAW vs CUGCR Edge Analysis)
# # =====================================================================
# def plot_gateway_efficiency(logs):
#     """Analyzes edges to see if the Gateway bypassed or triggered compression."""
#     edge_data = []
    
#     # We estimate the threshold: if tokens > 800, CUGCR likely fired. If < 800, RAW fired.
#     # (Using 800 as a safe boundary based on your manifest budgets of 400 and 1000)
#     for log in logs:
#         node_qualities = {n["node_id"]: n["quality_score"] for n in log.get("got_dag_nodes", []) if n.get("quality_score")}
        
#         for edge in log.get("communication_edges", []):
#             tokens = edge.get("context_token_count", 0)
#             qual = node_qualities.get(edge.get("to_id"), 0.0)
#             rel = edge.get("context_relevance", 0.0) * 100
            
#             if tokens == 0: continue
            
#             # Infer which path the gateway took based on token size
#             routing_path = "CUGCR (Compressed)" if tokens > 800 else "RAW (Bypass)"
            
#             edge_data.append({
#                 "Routing Path": routing_path,
#                 "Tokens Passed": tokens,
#                 "Node Quality": qual,
#                 "Relevance (%)": rel
#             })

#     if not edge_data: return
#     df = pd.DataFrame(edge_data)

#     # Subplot 1: Token Cost Comparison
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     plt_sns.boxplot(data=df, x='Routing Path', y='Tokens Passed', ax=axes[0], palette="Set2")
#     axes[0].set_title('Token Cost by Gateway Decision')
#     axes[0].set_ylabel('Tokens Passed per Edge')

#     # Subplot 2: Quality Comparison
#     plt_sns.barplot(data=df, x='Routing Path', y='Node Quality', ax=axes[1], palette="Set2", errorbar='sd')
#     axes[1].set_title('Downstream Quality by Gateway Decision')
#     axes[1].set_ylabel('Mean Node Quality Score')
#     axes[1].set_ylim(0, 1.0)

#     plt.tight_layout()
#     plt.savefig("fig2_gateway_efficiency.png")
#     print("✅ Generated: fig2_gateway_efficiency.png")

# # =====================================================================
# # Graph 3: System Resilience (Recovery Trajectory)
# # =====================================================================
# def plot_recovery_trajectory(logs):
#     initial_scores = []
#     final_scores = []
#     query_labels = []
    
#     count = 1
#     for log in logs:
#         history = log.get("global_score_history", [])
#         if len(history) > 1 and history[0] < 8.0:
#             initial_scores.append(history[0])
#             final_scores.append(history[-1])
#             query_labels.append(f"Q{count}")
#             count += 1

#     if not initial_scores: return

#     x = np.arange(len(query_labels))
#     width = 0.35
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.bar(x - width/2, initial_scores, width, label='Initial Draft', color='lightcoral')
#     ax.bar(x + width/2, final_scores, width, label='Post-Reflexion Retry', color='mediumseagreen')
#     ax.set_ylabel('Global Task Score (Out of 10)')
#     ax.set_title('System Resilience: Score Improvement via Diagnostician Retry')
#     ax.set_xticks(x)
#     ax.set_xticklabels(query_labels)
#     ax.axhline(y=8, color='gray', linestyle='--', label='Passing Threshold')
#     ax.legend()
#     ax.set_ylim(0, 10.5)
    
#     plt.tight_layout()
#     plt.savefig("fig3_recovery_trajectory.png")
#     print("✅ Generated: fig3_recovery_trajectory.png")


# if __name__ == "__main__":
#     print("Initializing Dynamic Gateway Evaluation Suite...")
#     data = load_data()
#     if data:
#         plot_sweet_spot(data)
#         plot_gateway_efficiency(data)
#         plot_recovery_trajectory(data)
#         print("\n🎯 Complete! Graphs saved based on your Dynamic Gateway architecture.")

import json
import os
import matplotlib.pyplot as plt
import seaborn as plt_sns
import pandas as pd
import numpy as np
from collections import defaultdict

# Set global academic plotting style
plt_sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 300})

def load_data(json_path="experiment_telemetry.json"):
    if not os.path.exists(json_path):
        print("Telemetry file not found.")
        return []
    with open(json_path, "r") as f:
        return json.load(f)

# =====================================================================
# 1. The Context Sensitivity U-Curve (The "Sweet Spot")
# =====================================================================
def plot_sweet_spot(logs):
    utilizations = []
    qualities = []
    for log in logs:
        node_qualities = {n["node_id"]: n["quality_score"] for n in log.get("got_dag_nodes", []) if n.get("quality_score")}
        for edge in log.get("communication_edges", []):
            util = edge.get("utilization_score")
            qual = node_qualities.get(edge.get("to_id"), 0.0)
            if util is not None and qual > 0:
                utilizations.append(util * 100)
                qualities.append(qual)

    if len(utilizations) < 3: return
    plt.figure(figsize=(8, 6))
    plt_sns.regplot(x=utilizations, y=qualities, order=2, scatter_kws={"s": 100, "alpha": 0.5, "color": "royalblue"}, line_kws={"color": "darkred", "linewidth": 2})
    plt.title('Agent Cognitive Load: Context Utilization vs. Reasoning Quality')
    plt.xlabel('Agent-to-Agent Context Utilization (%)')
    plt.ylabel('Node Quality Score (0.0 to 1.0)')
    plt.xlim(-5, 105); plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig("fig1_context_sweet_spot.png")

# =====================================================================
# 2. Agent-Pair Relevance Heatmap
# =====================================================================
def plot_agent_pair_relevance(logs):
    pair_data = defaultdict(list)
    for log in logs:
        for edge in log.get("communication_edges", []):
            if not edge.get("from_node_ids"): continue
            sender = next((n["agent"].upper() for n in log.get("got_dag_nodes", []) if n["node_id"] in edge["from_node_ids"]), "UNKNOWN")
            receiver = edge.get("to_agent", "UNKNOWN").upper()
            pair_data[f"{sender} → {receiver}"].append(edge.get("context_relevance", 0) * 100)

    if not pair_data: return
    df = pd.DataFrame({'Agent Edge': list(pair_data.keys()), 'Relevance (%)': [np.mean(v) for v in pair_data.values()], 'Count': [len(v) for v in pair_data.values()]}).sort_values('Relevance (%)', ascending=False)
    
    plt.figure(figsize=(10, 5))
    bar_plot = plt_sns.barplot(data=df, x='Relevance (%)', y='Agent Edge', palette="viridis")
    for i, p in enumerate(bar_plot.patches):
        plt.text(p.get_width() + 2, p.get_y() + p.get_height()/2, f'n={df.iloc[i]["Count"]}', va='center')
    plt.title('Average Context Relevance by Agent Edge')
    plt.xlim(0, 110)
    plt.tight_layout()
    plt.savefig("fig2_agent_edge_relevance.png")

# =====================================================================
# 3. Dynamic Gateway Behavior (Token Cost vs Quality)
# =====================================================================
def plot_gateway_efficiency(logs):
    """Analyzes edges to see if the Gateway bypassed or triggered compression."""
    edge_data = []
    for log in logs:
        node_qualities = {n["node_id"]: n["quality_score"] for n in log.get("got_dag_nodes", []) if n.get("quality_score")}
        for edge in log.get("communication_edges", []):
            tokens = edge.get("context_token_count", 0)
            qual = node_qualities.get(edge.get("to_id"), 0.0)
            if tokens == 0: continue
            
            routing_path = "CUGCR (Compressed)" if tokens > 800 else "RAW (Bypass)"
            edge_data.append({"Routing Path": routing_path, "Tokens": tokens, "Quality": qual})

    if not edge_data: return
    df = pd.DataFrame(edge_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt_sns.boxplot(data=df, x='Routing Path', y='Tokens', ax=axes[0], palette="Set2")
    axes[0].set_title('Token Cost by Gateway Decision')
    axes[0].set_ylabel('Tokens Passed per Edge')

    plt_sns.barplot(data=df, x='Routing Path', y='Quality', ax=axes[1], palette="Set2", errorbar='sd')
    axes[1].set_title('Downstream Quality by Gateway Decision')
    axes[1].set_ylabel('Mean Node Quality Score')
    axes[1].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig("fig3_gateway_efficiency.png")

# =====================================================================
# 4. Global Task Score Distribution
# =====================================================================
def plot_global_scores(logs):
    scores = [log.get("final_judge_score", 0) for log in logs]
    if not scores: return
    
    plt.figure(figsize=(7, 5))
    plt_sns.histplot(scores, bins=10, kde=True, color="teal")
    plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(scores):.2f}')
    plt.title('Distribution of Final Global Task Scores')
    plt.xlabel('Global Task Score (0-10)')
    plt.ylabel('Frequency (Number of Queries)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig4_global_score_distribution.png")

# =====================================================================
# 5. System Resilience (Score Improvement via Diagnostician Retry)
# =====================================================================
def plot_recovery_trajectory(logs):
    initial_scores, final_scores, query_labels = [], [], []
    count = 1
    for log in logs:
        history = log.get("global_score_history", [])
        if len(history) > 1 and history[0] < 8.0:
            initial_scores.append(history[0])
            final_scores.append(history[-1])
            query_labels.append(f"Q{count}")
            count += 1

    if not initial_scores: return
    x = np.arange(len(query_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, initial_scores, width, label='Initial Draft', color='lightcoral')
    ax.bar(x + width/2, final_scores, width, label='Post-Reflexion Retry', color='mediumseagreen')
    ax.set_ylabel('Global Task Score (Out of 10)')
    ax.set_title('System Resilience: Score Improvement via Diagnostician Retry')
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.axhline(y=8, color='gray', linestyle='--', label='Passing Threshold')
    ax.legend()
    ax.set_ylim(0, 10.5)
    plt.tight_layout()
    plt.savefig("fig5_recovery_trajectory.png")

# =====================================================================
# 6. GoT Operator Failures (Pie Chart)
# =====================================================================
def plot_node_failures(logs):
    failure_counts = {"ROOT": 0, "BRANCH": 0, "REFINE": 0, "AGGREGATE": 0}
    for log in logs:
        for node in log.get("got_dag_nodes", []):
            score = node.get("quality_score")
            if score is None or score < 0.75:
                op = node.get("operator", "ROOT").upper()
                if op in failure_counts: failure_counts[op] += 1

    labels = [k for k, v in failure_counts.items() if v > 0]
    sizes = [v for v in failure_counts.values() if v > 0]
    if not sizes: return

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt_sns.color_palette("pastel"))
    plt.title('Distribution of Node Failures across GoT Operators')
    plt.tight_layout()
    plt.savefig("fig6_operator_failures.png")

# =====================================================================
# 7. Token Injection Boxplot by Agent
# =====================================================================
def plot_token_efficiency(logs):
    df_data = [{"Receiving Agent": e.get("to_agent", "UNKNOWN").upper(), "Tokens Injected": e.get("context_token_count", 0)} for l in logs for e in l.get("communication_edges", []) if e.get("context_token_count", 0) > 0]
    if not df_data: return
    
    plt.figure(figsize=(8, 6))
    plt_sns.boxplot(data=pd.DataFrame(df_data), x='Receiving Agent', y='Tokens Injected', palette="Set2")
    plt_sns.stripplot(data=pd.DataFrame(df_data), x='Receiving Agent', y='Tokens Injected', color=".25", alpha=0.6, jitter=True)
    plt.title('Context Overhead: Tokens Injected per Receiving Agent')
    plt.ylabel('Token Count')
    plt.tight_layout()
    plt.savefig("fig7_token_overhead_boxplot.png")


if __name__ == "__main__":
    print("Initializing Master Visualization Suite...")
    data = load_data()
    if data:
        plot_sweet_spot(data)
        plot_agent_pair_relevance(data)
        plot_gateway_efficiency(data)
        plot_global_scores(data)
        plot_recovery_trajectory(data)
        plot_node_failures(data)
        plot_token_efficiency(data)
        print("\n🎯 Complete! 7 high-res PNG graphs have been generated.")