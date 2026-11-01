"""
run_cugcr_eval.py
=================
Evaluate CUGCR (Context-Understanding Guided Context Routing) along two axes:

  1. Compression analysis  (0 LLM calls — static, from existing telemetry)
     - Compression ratio distribution across all adaptive communication_edges
     - CRS gate trigger rate and score distribution
     - Atom-type survival breakdown (CODE_BLOCK, NAMED_ENTITY, PROSE_CLAIM)
     - Over-compression detection (edges where <20% of input tokens survive)
     - Per-node internal quality split by cugcr_triggered flag

  2. Retention quality test  (~60 LLM judge calls)
     - Load adaptive final answers for 30 best queries from raw_telemetry/
     - Load raw-mode final answers for the same 30 queries (no CUGCR)
     - Judge both with GPT-as-judge (same GS_Score formula as benchmarks)
     - Report whether downstream quality holds when CUGCR fires vs raw passing

Output
------
  results/cugcr_eval/
    cugcr_compression_stats.json   — per-edge compression metrics
    cugcr_quality_results.json     — per-query GS_Score: adaptive vs raw
    cugcr_eval_report.txt          — human-readable summary
    cugcr_compression_chart.png    — compression ratio histogram + CRS dist
    cugcr_quality_chart.png        — GS_Score comparison + per-dimension

Usage
-----
    python scripts/run_cugcr_eval.py                   # full run
    python scripts/run_cugcr_eval.py --no-judge        # compression only (0 LLM calls)
    python scripts/run_cugcr_eval.py --queries 5       # test on first 5 best queries
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI
from config import Config

# ── Paths ──────────────────────────────────────────────────────────────────────
TEL_DIR     = ROOT / "raw_telemetry"
RESULTS_DIR = ROOT / "results" / "cugcr_eval"
BEST_Q_PATH = ROOT / "best_questions.json"

JUDGE_MODEL = Config.JUDGE_MODEL

DIMS = ["Correctness", "Completeness", "Grounding", "Specificity", "Code_Quality", "Coherence"]
GS_WEIGHTS = {
    "Grounding": 0.35, "Specificity": 0.35, "Correctness": 0.15,
    "Completeness": 0.10, "Code_Quality": 0.03, "Coherence": 0.02,
}

# ── Load best queries ──────────────────────────────────────────────────────────
def _load_best_queries() -> list[dict]:
    return json.loads(BEST_Q_PATH.read_text())

# ── Part 1: Compression Analysis ──────────────────────────────────────────────

def _count_atom_types(compressed_text: str) -> dict[str, int]:
    """Count atom-type sections surviving in the CUGCR output."""
    code_blocks   = len(re.findall(r'```[\s\S]*?```', compressed_text))
    entities_line = re.search(r'### Key Technical Entities\n(.+)', compressed_text)
    entities      = len(entities_line.group(1).split(",")) if entities_line else 0
    prose_bullets = len(re.findall(r'^- .+', compressed_text, re.MULTILINE))
    return {"code_blocks": code_blocks, "named_entities": entities, "prose_claims": prose_bullets}


def _context_block_tokens(raw_context_injected: str, original_query: str) -> int:
    """
    Extract just the compressed context portion from raw_context_injected.

    build_rigid_prompt produces:
        {context_block}---\n## Your Task\n{original_query}\n

    So context_block_chars = len(raw_context_injected) - len(original_query) - ~20 (separator)
    """
    sep = "---\n## Your Task\n"
    idx = raw_context_injected.rfind(sep)
    if idx >= 0:
        context_block = raw_context_injected[:idx]
    else:
        # Fallback: subtract the query length
        context_block = raw_context_injected[: max(0, len(raw_context_injected) - len(original_query) - 25)]
    return max(1, len(context_block) // 4)


def analyze_compression(tel_dir: Path) -> dict:
    """
    Parse all adaptive telemetry runs and extract per-edge compression metrics.

    Compression ratio = compressed_context_tokens / raw_context_token_count
      - ratio < 1.0 → CUGCR reduced context size  (true compression)
      - ratio ≈ 1.0 → structural reformat only (light mode, budget not binding)
      - ratio > 1.0 → CUGCR headers/code blocks exceed budget (budget not enforced)

    budget_enforced = True when the context_block is smaller than the raw input,
    meaning the prose/entity budget gating actually cut some content.
    """
    edges = []
    node_records = []

    for fn in sorted(tel_dir.glob("run_*.json")):
        try:
            data = json.loads(fn.read_text())
        except Exception:
            continue
        if data.get("ablation_mode") != "adaptive":
            continue
        et = data.get("engine_telemetry", {})
        query = et.get("user_query", "")

        # ── communication_edges: CUGCR compression events ─────────────────────
        for edge in et.get("communication_edges", []):
            ctx_tok  = edge.get("context_token_count", 0)
            orig_q   = edge.get("original_query", "")
            resol_q  = edge.get("resolved_query", "")
            raw_ctx  = edge.get("raw_context_injected", "")
            crs      = edge.get("context_relevance", None)

            if not (ctx_tok and orig_q and resol_q):
                continue

            in_tok      = ctx_tok
            # Correct output: compressed context block only (strip task query overhead)
            out_tok_ctx = _context_block_tokens(raw_ctx or resol_q, orig_q)

            ratio             = out_tok_ctx / in_tok if in_tok else 1.0
            budget_enforced   = ratio < 0.95          # context actually shrunk ≥5%
            over_compressed   = ratio < 0.15          # <15% retention → risky info loss
            atoms = _count_atom_types(raw_ctx) if raw_ctx else {}

            edges.append({
                "query":             query[:80],
                "downstream_agent":  edge.get("to_agent", "?"),
                "in_tokens":         in_tok,
                "out_tokens":        out_tok_ctx,
                "compression_ratio": round(ratio, 3),
                "budget_enforced":   budget_enforced,
                "over_compressed":   over_compressed,
                "crs_score":         round(crs, 3) if crs is not None else None,
                "code_blocks":       atoms.get("code_blocks", 0),
                "named_entities":    atoms.get("named_entities", 0),
                "prose_claims":      atoms.get("prose_claims", 0),
            })

        # ── got_dag_nodes: per-node quality split ──────────────────────────────
        for node in et.get("got_dag_nodes", []):
            node_records.append({
                "query":          query[:80],
                "agent":          node.get("agent", "?"),
                "cugcr_triggered": node.get("cugcr_triggered", False),
                "quality_score":  node.get("quality_score"),
                "preliminary_crs": node.get("preliminary_crs"),
                "crs_gate_blocked": node.get("crs_gate_blocked", False),
                "depth":          node.get("depth", 0),
            })

    # ── Compute summary stats ──────────────────────────────────────────────────
    ratios = [e["compression_ratio"] for e in edges]
    crs_scores = [e["crs_score"] for e in edges if e["crs_score"] is not None]

    # Node quality split
    q_with    = [n["quality_score"] for n in node_records if n["cugcr_triggered"] and n["quality_score"] is not None]
    q_without = [n["quality_score"] for n in node_records if not n["cugcr_triggered"] and n["quality_score"] is not None]
    crs_blocked = [n for n in node_records if n["crs_gate_blocked"]]

    # Per-agent compression
    by_agent: dict[str, list] = defaultdict(list)
    for e in edges:
        by_agent[e["downstream_agent"]].append(e["compression_ratio"])

    summary = {
        "total_edges_analyzed":  len(edges),
        "total_nodes_analyzed":  len(node_records),
        "note": (
            "compression_ratio = compressed_context_tokens / raw_context_tokens. "
            "Ratio < 1.0 = true compression; ratio > 1.0 = code-block overhead exceeds budget (code blocks are always included). "
            "budget_enforced = True when ratio < 0.95 (prose/entity gating actually cut content)."
        ),
        "compression_ratio": {
            "mean":   round(float(np.mean(ratios)), 3)   if ratios else 0,
            "median": round(float(np.median(ratios)), 3) if ratios else 0,
            "p10":    round(float(np.percentile(ratios, 10)), 3) if ratios else 0,
            "p90":    round(float(np.percentile(ratios, 90)), 3) if ratios else 0,
            "min":    round(float(np.min(ratios)), 3)    if ratios else 0,
            "max":    round(float(np.max(ratios)), 3)    if ratios else 0,
            "pct_budget_enforced":  round(100 * sum(e["budget_enforced"]  for e in edges) / len(edges), 1) if edges else 0,
            "pct_over_compressed":  round(100 * sum(e["over_compressed"]  for e in edges) / len(edges), 1) if edges else 0,
        },
        "crs_gate": {
            "mean_score":   round(float(np.mean(crs_scores)), 3) if crs_scores else 0,
            "pct_below_0_25": round(100 * sum(1 for s in crs_scores if s < 0.25) / len(crs_scores), 1) if crs_scores else 0,
            "blocked_events": len(crs_blocked),
        },
        "node_quality": {
            "cugcr_triggered":     {"n": len(q_with),    "mean": round(float(np.mean(q_with)), 3)    if q_with    else 0},
            "cugcr_not_triggered": {"n": len(q_without), "mean": round(float(np.mean(q_without)), 3) if q_without else 0},
        },
        "per_agent_compression": {
            ag: {"mean": round(float(np.mean(vs)), 3), "n": len(vs)}
            for ag, vs in by_agent.items()
        },
        "atom_survival": {
            "mean_code_blocks":   round(float(np.mean([e["code_blocks"]   for e in edges])), 2) if edges else 0,
            "mean_named_entities": round(float(np.mean([e["named_entities"] for e in edges])), 2) if edges else 0,
            "mean_prose_claims":  round(float(np.mean([e["prose_claims"]  for e in edges])), 2) if edges else 0,
        },
    }

    return {"summary": summary, "edges": edges, "nodes": node_records}


# ── Part 2: Retention Quality (GPT-as-judge) ──────────────────────────────────

def _load_final_answers(mode: str, queries: list[dict]) -> dict[str, str]:
    """Load the best final_answer per query for the given ablation_mode."""
    best: dict[str, tuple[str, str]] = {}  # query_prefix → (timestamp, answer)
    for fn in TEL_DIR.glob("run_*.json"):
        try:
            data = json.loads(fn.read_text())
        except Exception:
            continue
        if data.get("ablation_mode") != mode:
            continue
        et = data.get("engine_telemetry", {})
        q = et.get("user_query", "").strip()
        ans = et.get("final_answer", "").strip()
        ts = data.get("timestamp", "")
        if not (q and ans):
            continue
        key = q[:120]
        if key not in best or ts > best[key][0]:
            best[key] = (ts, ans)

    result = {}
    for entry in queries:
        q = entry["query"]
        key = q[:120]
        result[q] = best.get(key, ("", ""))[1]
    return result


JUDGE_PROMPT = """You are an expert evaluator of bioinformatics AI systems. Many questions ask the system to retrieve information from a specific knowledge base (KB) of papers. Reward answers that cite paper-specific content; penalize answers that give generic textbook descriptions.

QUESTION: "{query}"

ANSWER TO EVALUATE:
{answer}

Score this answer on each dimension from 0-10. Return ONLY valid JSON, no prose:

{{
  "Correctness":   <0-10>,
  "Completeness":  <0-10>,
  "Grounding":     <0-10>,
  "Specificity":   <0-10>,
  "Code_Quality":  <0-10>,
  "Coherence":     <0-10>,
  "reasoning":     "<one sentence>"
}}

Scoring guide:
  Correctness  — factual accuracy with no hallucinations
  Completeness — covers all parts of the multi-part question
  Grounding    — anchored to paper/KB content (0-3: generic, 7-9: paper-specific, 10: paper-exact)
  Specificity  — concreteness of technical detail
  Code_Quality — runnable code with correct API calls (10 if no code needed)
  Coherence    — logical flow and clear structure"""


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


async def judge_answer(query: str, answer: str, client: AsyncOpenAI) -> dict:
    prompt = JUDGE_PROMPT.format(query=query[:300], answer=answer[:3000])
    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "developer", "content": "You are a strict JSON-only evaluator. Output only valid JSON."},
                {"role": "user",      "content": prompt},
            ],
        )
        raw  = resp.choices[0].message.content.strip()
        data = _extract_json(raw)
        if not data:
            raise ValueError(f"No JSON: {raw[:100]}")
        scores = {d: float(data.get(d, 0)) for d in DIMS}
        scores["mean"]      = round(sum(scores[d] for d in DIMS) / len(DIMS), 2)
        scores["gs_score"]  = round(sum(scores[d] * GS_WEIGHTS[d] for d in DIMS), 2)
        scores["reasoning"] = data.get("reasoning", "")
        return scores
    except Exception as e:
        return {d: 0.0 for d in DIMS} | {"mean": 0.0, "gs_score": 0.0, "reasoning": f"err: {e}"}


async def run_quality_eval(queries: list[dict], client: AsyncOpenAI) -> list[dict]:
    """Judge adaptive vs raw final answers for each query."""
    adaptive_answers = _load_final_answers("adaptive", queries)
    raw_answers      = _load_final_answers("raw",      queries)

    results = []
    for i, entry in enumerate(queries, 1):
        q   = entry["query"]
        qid = entry["qid"]
        print(f"  Q{i:02d}/{len(queries)} ({qid}): {q[:70]}...")

        ans_adap = adaptive_answers.get(q, "")
        ans_raw  = raw_answers.get(q, "")

        if not ans_adap:
            print(f"    ⚠  No adaptive answer found — skipping")
            continue
        if not ans_raw:
            print(f"    ⚠  No raw-mode answer found — skipping")
            continue

        scored_adap, scored_raw = await asyncio.gather(
            judge_answer(q, ans_adap, client),
            judge_answer(q, ans_raw,  client),
        )

        gs_delta = round(scored_adap["gs_score"] - scored_raw["gs_score"], 3)
        winner   = "adaptive" if gs_delta >= 0 else "raw"
        print(f"    adaptive={scored_adap['gs_score']:.2f}  raw={scored_raw['gs_score']:.2f}  "
              f"Δ={gs_delta:+.2f}  [{winner} wins]")

        results.append({
            "qid":   qid,
            "query": q,
            "adaptive": {"answer": ans_adap[:300], "scores": scored_adap},
            "raw":      {"answer": ans_raw[:300],  "scores": scored_raw},
            "gs_delta": gs_delta,
            "topic":    entry.get("topic", ""),
        })

        await asyncio.sleep(0.5)

    return results


# ── Charts ─────────────────────────────────────────────────────────────────────

def _make_compression_chart(summary: dict, edges: list, out_path: Path):
    ratios  = [e["compression_ratio"] for e in edges]
    crs_sc  = [e["crs_score"] for e in edges if e["crs_score"] is not None]
    agents  = sorted(set(e["downstream_agent"] for e in edges))
    by_agent = {ag: [e["compression_ratio"] for e in edges if e["downstream_agent"] == ag]
                for ag in agents}

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Compression ratio histogram
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.linspace(0, max(ratios) + 0.1, 40)
    ax1.hist(ratios, bins=bins, color="#3498db", edgecolor="white", alpha=0.85)
    ax1.axvline(0.15, color="#e74c3c", linewidth=2, linestyle="--", label="Over-compressed threshold (0.15)")
    ax1.axvline(np.median(ratios), color="#2ecc71", linewidth=2, linestyle="-", label=f"Median = {np.median(ratios):.2f}")
    pct_oc = summary["compression_ratio"]["pct_over_compressed"]
    ax1.set_title(f"Compression Ratio Distribution  (N={len(ratios)} edges,  "
                  f"{pct_oc:.1f}% over-compressed)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Compression ratio  (output tokens / input tokens)", fontsize=9)
    ax1.set_ylabel("Edge count", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # 2. Per-agent compression boxplot
    ax2 = fig.add_subplot(gs[0, 2])
    data_ag = [by_agent[ag] for ag in agents]
    bp = ax2.boxplot(data_ag, labels=agents, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 2})
    colors_ag = ["#2ecc71", "#e67e22", "#3498db", "#9b59b6"]
    for patch, c in zip(bp["boxes"], colors_ag[:len(agents)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.axhline(0.15, color="#e74c3c", linewidth=1.5, linestyle="--", label="OC threshold")
    ax2.set_title("Compression by downstream agent", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Compression ratio", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.spines[["top", "right"]].set_visible(False)

    # 3. CRS gate score distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if crs_sc:
        ax3.hist(crs_sc, bins=30, color="#9b59b6", edgecolor="white", alpha=0.85)
        ax3.axvline(0.25, color="#e74c3c", linewidth=2, linestyle="--", label="Gate threshold (0.25)")
        pct_blocked = summary["crs_gate"]["pct_below_0_25"]
        ax3.set_title(f"CRS Gate Score Distribution\n({pct_blocked:.1f}% below threshold)", fontsize=9, fontweight="bold")
        ax3.set_xlabel("CRS score (cosine sim)", fontsize=9)
        ax3.set_ylabel("Edge count", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.spines[["top", "right"]].set_visible(False)

    # 4. Atom type survival counts
    ax4 = fig.add_subplot(gs[1, 1])
    atom_keys = ["code_blocks", "named_entities", "prose_claims"]
    atom_labels = ["Code\nblocks", "Named\nentities", "Prose\nclaims"]
    atom_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    means = [np.mean([e[k] for e in edges]) for k in atom_keys]
    bars = ax4.bar(atom_labels, means, color=atom_colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_title("Mean atom count per compressed edge", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Mean count", fontsize=9)
    ax4.spines[["top", "right"]].set_visible(False)

    # 5. Node quality: CUGCR vs no CUGCR
    ax5 = fig.add_subplot(gs[1, 2])
    nq = summary["node_quality"]
    cats = ["CUGCR\ntriggered", "CUGCR\nnot triggered"]
    vals = [nq["cugcr_triggered"]["mean"], nq["cugcr_not_triggered"]["mean"]]
    ns   = [nq["cugcr_triggered"]["n"],    nq["cugcr_not_triggered"]["n"]]
    bars2 = ax5.bar(cats, vals, color=["#2ecc71", "#3498db"], edgecolor="white", width=0.45)
    for bar, v, n in zip(bars2, vals, ns):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax5.set_ylim(0, 1.1)
    ax5.set_title("Per-node quality score\n(internal 0-1 judge)", fontsize=9, fontweight="bold")
    ax5.set_ylabel("Mean quality score", fontsize=9)
    ax5.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CUGCR Compression Analysis  |  Static evaluation from adaptive telemetry",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Compression chart → {out_path}")


def _make_quality_chart(results: list[dict], out_path: Path):
    systems = ["adaptive", "raw"]
    colors  = {"adaptive": "#2ecc71", "raw": "#e67e22"}
    labels  = {"adaptive": "Adaptive (CUGCR)", "raw": "Raw (no CUGCR)"}

    gs_adap = [r["adaptive"]["scores"]["gs_score"] for r in results]
    gs_raw  = [r["raw"]["scores"]["gs_score"]      for r in results]

    dim_means = {
        "adaptive": {d: np.mean([r["adaptive"]["scores"][d] for r in results]) for d in DIMS},
        "raw":      {d: np.mean([r["raw"]["scores"][d]      for r in results]) for d in DIMS},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # Left: GS_Score bar + per-query scatter
    ax = axes[0]
    x = np.arange(2)
    means = [np.mean(gs_adap), np.mean(gs_raw)]
    stds  = [np.std(gs_adap),  np.std(gs_raw)]
    bars = ax.bar(x, means, 0.45, color=[colors[s] for s in systems],
                  edgecolor="white", yerr=stds, capsize=5,
                  error_kw={"linewidth": 1, "color": "#555"}, zorder=3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    # overlay individual query dots
    jitter = 0.08
    for i, (g_a, g_r) in enumerate(zip(gs_adap, gs_raw)):
        ax.plot([0 + (np.random.rand() - 0.5) * jitter,
                 1 + (np.random.rand() - 0.5) * jitter],
                [g_a, g_r], color="#bbb", linewidth=0.6, zorder=2)
        ax.scatter(0 + (np.random.rand() - 0.5) * jitter, g_a,
                   color=colors["adaptive"], s=18, alpha=0.7, zorder=4)
        ax.scatter(1 + (np.random.rand() - 0.5) * jitter, g_r,
                   color=colors["raw"],      s=18, alpha=0.7, zorder=4)

    delta = (means[0] - means[1]) / means[1] * 100 if means[1] else 0
    ax.annotate(f"{delta:+.1f}%\nvs raw",
                xy=(0, means[0] + stds[0] + 0.4),
                ha="center", fontsize=9, fontweight="bold",
                color="#2ecc71" if delta >= 0 else "#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in systems], fontsize=10)
    ax.set_ylabel("GS_Score (0–10)", fontsize=9)
    ax.set_title(f"GS_Score: Adaptive (CUGCR) vs Raw  |  N={len(results)} queries",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.yaxis.grid(True, color="#eee", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: per-dimension breakdown
    ax2 = axes[1]
    x2 = np.arange(len(DIMS))
    width = 0.35
    for sys_name, offset in zip(systems, [-width / 2, width / 2]):
        vals = [dim_means[sys_name][d] for d in DIMS]
        ax2.bar(x2 + offset, vals, width, label=labels[sys_name],
                color=colors[sys_name], edgecolor="white", zorder=3)
    for xi in [DIMS.index("Grounding"), DIMS.index("Specificity")]:
        ax2.axvspan(xi - 0.45, xi + 0.45, color="#fffde7", zorder=0, alpha=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([d.replace("_", "\n") for d in DIMS], fontsize=8)
    ax2.set_ylabel("Score (0–10)", fontsize=9)
    ax2.set_title("Per-dimension breakdown  (highlighted = GS_Score weighted)",
                  fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, color="#eee", zorder=0); ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    wins_adap = sum(1 for r in results if r["gs_delta"] > 0)
    wins_raw  = sum(1 for r in results if r["gs_delta"] < 0)
    fig.suptitle(f"CUGCR Retention Quality  |  Adaptive wins {wins_adap}/{len(results)} queries  |  Primary: GS_Score",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Quality chart → {out_path}")


# ── Text report ────────────────────────────────────────────────────────────────

def _write_report(summary: dict, quality_results: list, out_path: Path):
    lines = [
        "CUGCR EVALUATION REPORT",
        "=" * 70,
        "",
        "── 1. COMPRESSION ANALYSIS ──────────────────────────────────────────",
        f"  Edges analyzed        : {summary['total_edges_analyzed']}",
        f"  Nodes analyzed        : {summary['total_nodes_analyzed']}",
        "",
        "  Compression ratio (output_tokens / input_tokens):",
        f"    Mean   : {summary['compression_ratio']['mean']:.3f}",
        f"    Median : {summary['compression_ratio']['median']:.3f}",
        f"    P10    : {summary['compression_ratio']['p10']:.3f}",
        f"    P90    : {summary['compression_ratio']['p90']:.3f}",
        f"    Min    : {summary['compression_ratio']['min']:.3f}",
        f"    Max    : {summary['compression_ratio']['max']:.3f}",
        f"    Budget enforced (ratio<0.95): {summary['compression_ratio']['pct_budget_enforced']:.1f}% of edges",
        f"    Over-compressed (<15%):      {summary['compression_ratio']['pct_over_compressed']:.1f}% of edges",
        "",
        "  CRS Gate:",
        f"    Mean score       : {summary['crs_gate']['mean_score']:.3f}",
        f"    % below 0.25     : {summary['crs_gate']['pct_below_0_25']:.1f}%  (would block compression)",
        f"    Blocked events   : {summary['crs_gate']['blocked_events']}",
        "",
        "  Atom survival (mean per compressed edge):",
        f"    Code blocks    : {summary['atom_survival']['mean_code_blocks']:.2f}",
        f"    Named entities : {summary['atom_survival']['mean_named_entities']:.2f}",
        f"    Prose claims   : {summary['atom_survival']['mean_prose_claims']:.2f}",
        "",
        "  Per-agent mean compression ratio:",
    ]
    for ag, v in sorted(summary["per_agent_compression"].items()):
        lines.append(f"    {ag:<10} : {v['mean']:.3f}  (n={v['n']})")

    nq = summary["node_quality"]
    lines += [
        "",
        "  Node quality (internal 0-1 judge):",
        f"    CUGCR triggered     : {nq['cugcr_triggered']['mean']:.3f}  (n={nq['cugcr_triggered']['n']})",
        f"    CUGCR not triggered : {nq['cugcr_not_triggered']['mean']:.3f}  (n={nq['cugcr_not_triggered']['n']})",
    ]

    if quality_results:
        gs_adap = [r["adaptive"]["scores"]["gs_score"] for r in quality_results]
        gs_raw  = [r["raw"]["scores"]["gs_score"]      for r in quality_results]
        delta   = np.mean(gs_adap) - np.mean(gs_raw)
        pct     = delta / np.mean(gs_raw) * 100 if np.mean(gs_raw) else 0
        wins_a  = sum(1 for r in quality_results if r["gs_delta"] > 0)
        wins_r  = sum(1 for r in quality_results if r["gs_delta"] < 0)
        ties    = len(quality_results) - wins_a - wins_r
        lines += [
            "",
            "── 2. RETENTION QUALITY TEST ────────────────────────────────────────",
            f"  Queries evaluated   : {len(quality_results)}",
            f"  Adaptive GS_Score   : {np.mean(gs_adap):.3f}  (std={np.std(gs_adap):.3f})",
            f"  Raw GS_Score        : {np.mean(gs_raw):.3f}  (std={np.std(gs_raw):.3f})",
            f"  GS_Score lift       : {delta:+.3f} pts  ({pct:+.1f}%)",
            f"  Per-query wins      : adaptive={wins_a}  raw={wins_r}  ties={ties}",
            "",
            "  Per-query results:",
            f"  {'QID':<5}  {'Adaptive':>9}  {'Raw':>6}  {'Delta':>7}  Topic",
            f"  {'-'*52}",
        ]
        for r in sorted(quality_results, key=lambda x: x["gs_delta"], reverse=True):
            lines.append(
                f"  {r['qid']:<5}  {r['adaptive']['scores']['gs_score']:>9.2f}  "
                f"{r['raw']['scores']['gs_score']:>6.2f}  "
                f"{r['gs_delta']:>+7.2f}  {r.get('topic','')}"
            )

    lines += ["", "=" * 70]
    out_path.write_text("\n".join(lines))
    print(f"  Report → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_entries = _load_best_queries()
    if args.queries > 0:
        best_entries = best_entries[:args.queries]

    print(f"\n{'═'*65}")
    print(f"  CUGCR EVALUATION")
    print(f"  Best queries : {len(best_entries)}")
    print(f"  Judge calls  : {'0 (--no-judge)' if args.no_judge else f'~{len(best_entries)*2}'}")
    print(f"{'═'*65}\n")

    # ── Part 1: Static compression analysis ──────────────────────────────────
    print("⏳ Analyzing compression from adaptive telemetry...")
    comp = analyze_compression(TEL_DIR)
    summary = comp["summary"]

    print(f"  Edges: {summary['total_edges_analyzed']}  |  "
          f"Mean ratio: {summary['compression_ratio']['mean']:.3f}  |  "
          f"Over-compressed: {summary['compression_ratio']['pct_over_compressed']:.1f}%  |  "
          f"CRS mean: {summary['crs_gate']['mean_score']:.3f}")

    stats_path = RESULTS_DIR / "cugcr_compression_stats.json"
    stats_path.write_text(json.dumps(
        {"summary": summary, "edges": comp["edges"][:200]}, indent=2
    ))
    print(f"  Stats → {stats_path}")

    _make_compression_chart(summary, comp["edges"], RESULTS_DIR / "cugcr_compression_chart.png")

    # ── Part 2: Retention quality test ───────────────────────────────────────
    quality_results = []
    if not args.no_judge:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            print("\n⚠  OPENAI_API_KEY not set — skipping quality eval. Use --no-judge to suppress.")
        else:
            client = AsyncOpenAI(api_key=openai_key)
            print(f"\n⏳ Running quality retention eval on {len(best_entries)} queries...")
            quality_results = await run_quality_eval(best_entries, client)

            qr_path = RESULTS_DIR / "cugcr_quality_results.json"
            qr_path.write_text(json.dumps(quality_results, indent=2))
            print(f"  Quality results → {qr_path}")

            if quality_results:
                _make_quality_chart(quality_results, RESULTS_DIR / "cugcr_quality_chart.png")

    _write_report(summary, quality_results, RESULTS_DIR / "cugcr_eval_report.txt")

    # ── Console summary ───────────────────────────────────────────────────────
    nq = summary["node_quality"]
    print(f"\n{'─'*65}")
    print(f"  COMPRESSION SUMMARY")
    print(f"  Mean ratio: {summary['compression_ratio']['mean']:.3f}  "
          f"  Median: {summary['compression_ratio']['median']:.3f}")
    print(f"  Over-compressed edges: {summary['compression_ratio']['pct_over_compressed']:.1f}%")
    print(f"  CRS gate mean: {summary['crs_gate']['mean_score']:.3f}  "
          f"  blocked: {summary['crs_gate']['blocked_events']}")
    print(f"  Node quality — CUGCR: {nq['cugcr_triggered']['mean']:.3f} "
          f"vs no-CUGCR: {nq['cugcr_not_triggered']['mean']:.3f}")

    if quality_results:
        gs_a = np.mean([r["adaptive"]["scores"]["gs_score"] for r in quality_results])
        gs_r = np.mean([r["raw"]["scores"]["gs_score"]      for r in quality_results])
        print(f"\n  RETENTION QUALITY")
        print(f"  Adaptive GS_Score: {gs_a:.3f}  Raw: {gs_r:.3f}  "
              f"Δ={gs_a-gs_r:+.3f} ({(gs_a-gs_r)/gs_r*100:+.1f}%)" if gs_r else "")
    print(f"{'─'*65}")


def main():
    p = argparse.ArgumentParser(description="CUGCR evaluation: compression + retention quality")
    p.add_argument("--no-judge",  action="store_true",
                   help="Skip GPT-as-judge quality eval (compression analysis only, 0 LLM calls)")
    p.add_argument("--queries",   type=int, default=0,
                   help="Limit quality eval to first N best queries (0 = all 30)")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
