import os
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR = Path(os.path.dirname(__file__)).parent
TELEMETRY_DIR = ROOT_DIR / "raw_telemetry"
VIZ_DIR = ROOT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

COLORS = {"RAW": "#4C72B0", "ADAPTIVE": "#E1812C"}

# ── 1. Data Loader ───────────────────────────────────────────────────────────
def load_telemetry():
    runs = []
    if not TELEMETRY_DIR.exists():
        print(f"❌ Error: {TELEMETRY_DIR} not found.")
        return []
    for path in TELEMETRY_DIR.glob("run_*.json"):
        with open(path, "r") as f:
            try: runs.append(json.load(f))
            except: pass
    return runs

# ── 2. Atom Extractor (Mirroring engine.py) ──────────────────────────────────
def extract_critical_atoms(text: str):
    """Extracts code blocks and biological entities to check for knowledge transfer."""
    atoms = set()
    
    # Extract Code Blocks
    for match in re.finditer(r'```[\s\S]*?```', text):
        # Just grab the function names or key parameters to check for presence
        code_body = match.group()
        words = re.findall(r'\b[a-zA-Z_]\w*\b', code_body)
        atoms.update([w for w in words if len(w) > 4]) # Ignore short generic variables like 'i', 'x'
    
    # Extract Bio-Entities / Tool Names
    prose_only = re.sub(r'```[\s\S]*?```', '', text)
    bio_pattern = r'\b([A-Z]{2,}[0-9]*|[A-Z][a-z]+-\d+)\b'
    for match in re.finditer(bio_pattern, prose_only):
        atoms.add(match.group())
        
    return atoms

# ── 3. Metrics Extraction ────────────────────────────────────────────────────
def evaluate_knowledge_transfer(runs):
    transfer_data = []

    for run in runs:
        mode = run.get("ablation_mode", "UNKNOWN").upper()
        if mode not in ["RAW", "ADAPTIVE"]: continue # NONE mode passes 0 context, so transfer is 0 by definition
        
        t = run.get("engine_telemetry", {})
        
        # Build a lookup for what every node actually outputted
        node_outputs = {}
        for node in t.get("got_dag_nodes", []):
            node_outputs[node["node_id"]] = node.get("output", "") # NOTE: Requires 'output' to be saved in telemetry!

        # If the telemetry doesn't have raw node outputs, we have to fallback to checking the final_answer
        final_answer = t.get("final_answer", "")

        for edge in t.get("communication_edges", []):
            # 1. Get the raw text that the UPSTREAM agent generated
            # We use the 'context' passed on the edge as a proxy for the upstream output
            upstream_text = edge.get("context", "")
            
            # 2. Extract the critical atoms from the upstream text
            upstream_atoms = extract_critical_atoms(upstream_text)
            
            if not upstream_atoms: continue # Nothing critical to transfer
            
            # 3. Get the text that the DOWNSTREAM agent generated
            to_id = edge.get("to_id")
            downstream_text = node_outputs.get(to_id, final_answer) # Fallback to final answer if node output isn't logged
            
            # 4. Check how many atoms survived the transfer
            surviving_atoms = [atom for atom in upstream_atoms if atom in downstream_text]
            transfer_rate = (len(surviving_atoms) / len(upstream_atoms)) * 100
            
            transfer_data.append({
                "Mode": mode,
                "Edge": f"{edge.get('from_agent', 'UP').upper()} -> {edge.get('to_agent', 'DOWN').upper()}",
                "Upstream Atoms": len(upstream_atoms),
                "Surviving Atoms": len(surviving_atoms),
                "Transfer Rate (%)": transfer_rate
            })

    return pd.DataFrame(transfer_data)

# ── 4. Main Execution & Graphing ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        CROSS-AGENT KNOWLEDGE TRANSFER EFFICIENCY         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    runs = load_telemetry()
    df = evaluate_knowledge_transfer(runs)
    
    if df.empty:
        print("⚠️ Not enough data to calculate Knowledge Transfer. (Ensure context is being passed).")
        return

    # --- TERMINAL PRINTOUT ---
    print("\n🔬 MEAN KNOWLEDGE TRANSFER RATE (%)")
    print(" (Percentage of critical upstream entities that survived into the downstream output)")
    transfer_stats = df.groupby("Mode")["Transfer Rate (%)"].agg(["mean", "count"])
    print(transfer_stats.round(1).to_string())

    print("\n📊 Generating Knowledge Transfer Graphs...")

    # ── Graph Q: Knowledge Transfer Rate (Bar Chart) ──
    plt.figure(figsize=(7, 5))
    plt_sns.set_theme(style="whitegrid")
    
    ax = plt_sns.barplot(data=df, x="Mode", y="Transfer Rate (%)", 
                         order=["RAW", "ADAPTIVE"], palette=COLORS, edgecolor='black', alpha=0.85, capsize=0.1)
                         
    plt.title('Cross-Agent Knowledge Transfer Efficiency\n(Preservation of Critical Entities/Code)', fontweight='bold')
    plt.ylabel('Mean Transfer Rate (%)')
    plt.ylim(0, 105)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "fig21_knowledge_transfer_rate.png")
    print("✅ Generated: fig21_knowledge_transfer_rate.png")

    # ── Graph R: Transfer Rate by Specific Edge (Grouped Bar) ──
    plt.figure(figsize=(10, 6))
    
    # Filter for edges that have enough data points to plot
    edge_counts = df['Edge'].value_counts()
    valid_edges = edge_counts[edge_counts >= 2].index
    df_filtered = df[df['Edge'].isin(valid_edges)]
    
    if not df_filtered.empty:
        ax2 = plt_sns.barplot(data=df_filtered, x="Edge", y="Transfer Rate (%)", hue="Mode", 
                              palette=COLORS, edgecolor='black', alpha=0.85)
                              
        plt.title('Knowledge Transfer Efficiency by Edge Pathway', fontweight='bold')
        plt.ylabel('Mean Transfer Rate (%)')
        plt.xlabel('Communication Pathway')
        plt.ylim(0, 105)
        plt.legend(title='Ablation Mode')
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "fig22_transfer_rate_by_edge.png")
        print("✅ Generated: fig22_transfer_rate_by_edge.png")
    else:
        print("⚠️ Not enough overlapping edge data to generate Figure 22.")

if __name__ == "__main__":
    main()