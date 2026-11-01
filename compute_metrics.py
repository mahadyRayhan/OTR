"""
compute_metrics.py
==================
Phase 2 of the evaluation framework.
Reads every run_*.json from raw_telemetry/ and computes ALL metrics
across the 7 evaluation layers. Saves results as JSON + CSV files
ready for visualization.

Outputs (all in metrics/ directory):
  all_metrics.json          — full structured metrics tree
  L1_system.csv             — system-level metrics per run
  L2_agent.csv              — per-agent-type aggregates
  L3_edges.csv              — raw edge data (one row per communication edge)
  L3_communication.csv      — aggregated communication metrics per mode
  L4_knowledge.csv          — knowledge utilization per run
  L5_selfhealing.csv        — self-healing events and outcomes
  L6_cost_latency.csv       — cost/latency per run
  L6_pareto.csv             — (tokens, GTS) pairs for Pareto plot
  L7_novel.csv              — novel metrics (determinism, profile, boundary)
  llm_judge_cache.json      — cached LLM judge calls to avoid re-billing
"""

import asyncio
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Any

import anthropic
from dotenv import load_dotenv
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR     = Path("raw_telemetry")
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

JUDGE_CACHE_FILE = METRICS_DIR / "llm_judge_cache.json"

# ── LLM judge setup ──────────────────────────────────────────────────────────
_client = anthropic.Anthropic()
_judge_cache: dict[str, Any] = {}

if JUDGE_CACHE_FILE.exists():
    with open(JUDGE_CACHE_FILE) as f:
        _judge_cache = json.load(f)


def _flush_cache():
    with open(JUDGE_CACHE_FILE, "w") as f:
        json.dump(_judge_cache, f, indent=2, default=str)


def llm_judge(prompt: str, cache_key: str, max_tokens: int = 512) -> str:
    """Call Claude as judge. Results are cached by cache_key to avoid double billing."""
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]
    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        result = msg.content[0].text.strip()
        _judge_cache[cache_key] = result
        _flush_cache()
        return result
    except Exception as exc:
        print(f"  [judge error] {exc}")
        return ""


def judge_score(prompt: str, cache_key: str) -> float:
    """Ask the judge for a 0-10 numeric score. Returns -1.0 on failure."""
    response = llm_judge(prompt, cache_key)
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b|\bscore[:\s]+(\d+(?:\.\d+)?)", response, re.I)
    if match:
        return float(match.group(1) or match.group(2))
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
    candidates = [float(n) for n in nums if 0 <= float(n) <= 10]
    return candidates[0] if candidates else -1.0


# ── Load all runs ─────────────────────────────────────────────────────────────

def load_all_runs() -> list[dict]:
    runs = []
    for path in sorted(RAW_DIR.glob("run_*.json")):
        with open(path) as f:
            try:
                runs.append(json.load(f))
            except json.JSONDecodeError:
                print(f"  [warn] could not parse {path.name}")
    print(f"Loaded {len(runs)} run files.")
    return runs


def safe(record: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    cur = record
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def tele(run: dict) -> dict:
    """Shorthand for engine_telemetry sub-dict."""
    return run.get("engine_telemetry", {})


def nodes(run: dict) -> list[dict]:
    return tele(run).get("got_dag_nodes", [])


def edges(run: dict) -> list[dict]:
    return tele(run).get("communication_edges", [])


# ── Layer 1: System-level metrics ────────────────────────────────────────────

def compute_L1(runs: list[dict]) -> tuple[list[dict], dict]:
    """
    Metrics: GTS distribution, TSR, pairwise win rate, answer consistency.
    """
    print("\n[L1] Computing system-level metrics...")
    rows = []

    for run in runs:
        t = tele(run)
        failed  = run.get("failed", False)
        mode    = run.get("ablation_mode", "unknown")
        query   = run.get("query", "")
        qid     = run.get("query_id", "")
        proto   = run.get("protocol", "")

        gts          = t.get("final_judge_score", -1)
        total_tokens = t.get("total_tokens", 0)
        # Fix 9: engine stores latency_seconds — multiply by 1000 to get ms.
        # Fallback chain: latency_seconds → observed_latency_ms → wall_latency_ms
        if t.get("latency_seconds") is not None:
            latency_ms = t["latency_seconds"] * 1000
        else:
            latency_ms = t.get("observed_latency_ms", run.get("wall_latency_ms", 0))

        # Task Success Rate — binary: not failed AND no catastrophic output
        final_answer = t.get("final_answer", "")
        has_placeholder_leak = bool(re.search(r"\[Result of Task|\[PLACEHOLDER|\@\@REF\@\@", final_answer))
        is_empty    = len(final_answer.strip()) < 30
        tsr         = 0 if (failed or has_placeholder_leak or is_empty) else 1

        rows.append({
            "run_id":              run.get("run_id"),
            "protocol":            proto,
            "query_id":            qid,
            "ablation_mode":       mode,
            "gts":                 gts,
            "tsr":                 tsr,
            "total_tokens":        total_tokens,
            "latency_ms":          latency_ms,
            "failed":              int(failed),
            "has_placeholder_leak":int(has_placeholder_leak),
            "is_empty":            int(is_empty),
            "diagnostician_triggered": int(t.get("diagnostician_triggered", False)),
            "cugcr_triggered":     int(t.get("cugcr_triggered", False)),
            "depth_budget":        t.get("depth_budget", 0),
            "depth_used":          t.get("depth_used", 0),
        })

    # Pairwise win rate across Protocol A ablation runs
    # Group by query_id, then compare ADAPTIVE vs RAW vs NONE
    ablation_runs = [r for r in rows if r["protocol"] == "A_ablation"]
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in ablation_runs:
        grouped[r["query_id"]][r["ablation_mode"]].append(r["gts"])

    win_rates: dict[str, float] = {}
    total_battles = adaptive_wins = raw_wins = none_wins = 0
    for qid, modes_dict in grouped.items():
        adaptive_gts = statistics.mean(modes_dict.get("adaptive", [-1]))
        raw_gts      = statistics.mean(modes_dict.get("raw",      [-1]))
        none_gts     = statistics.mean(modes_dict.get("none",     [-1]))
        scores = {"adaptive": adaptive_gts, "raw": raw_gts, "none": none_gts}
        winner = max(scores, key=scores.get)
        total_battles += 1
        if winner == "adaptive": adaptive_wins += 1
        elif winner == "raw":    raw_wins      += 1
        else:                    none_wins      += 1

    win_rates = {
        "adaptive_win_rate": adaptive_wins / max(total_battles, 1),
        "raw_win_rate":      raw_wins      / max(total_battles, 1),
        "none_win_rate":     none_wins     / max(total_battles, 1),
        "total_comparisons": total_battles,
    }

    # Consistency: variance of GTS for Protocol B runs
    consistency_runs = [r for r in rows if r["protocol"] == "B_consistency" and r["gts"] >= 0]
    cons_by_qid: dict[str, list] = defaultdict(list)
    for r in consistency_runs:
        cons_by_qid[r["query_id"]].append(r["gts"])
    consistency = {}
    for qid, gtss in cons_by_qid.items():
        consistency[qid] = {
            "mean":   statistics.mean(gtss),
            "stdev":  statistics.stdev(gtss) if len(gtss) > 1 else 0.0,
            "min":    min(gtss),
            "max":    max(gtss),
        }

    # Summary stats per mode
    per_mode: dict[str, dict] = defaultdict(dict)
    for mode in ["none", "raw", "adaptive"]:
        mode_rows = [r for r in ablation_runs if r["ablation_mode"] == mode and r["gts"] >= 0]
        gtss = [r["gts"] for r in mode_rows]
        if gtss:
            per_mode[mode] = {
                "mean_gts":  round(statistics.mean(gtss), 3),
                "stdev_gts": round(statistics.stdev(gtss) if len(gtss) > 1 else 0.0, 3),
                "tsr":       round(statistics.mean([r["tsr"] for r in mode_rows]), 3),
                "n":         len(mode_rows),
            }

    summary = {
        "win_rates":   win_rates,
        "consistency": consistency,
        "per_mode":    dict(per_mode),
    }
    return rows, summary


# ── Layer 2: Agent-level metrics ─────────────────────────────────────────────

def compute_L2(runs: list[dict]) -> tuple[list[dict], dict]:
    """
    Metrics: NQS per agent, first-pass success rate, hallucination rate,
             placeholder leak rate, format compliance.
    """
    print("[L2] Computing agent-level metrics...")
    rows = []

    AI_WARNING_PATTERN    = re.compile(r"⚠️\s*AI EXTRAPOLATION WARNING", re.I)
    PLACEHOLDER_PATTERN   = re.compile(r"\[Result of Task|\[PLACEHOLDER|\@\@REF\@\@")
    DB_REFUSAL_PATTERN    = re.compile(r"# ERROR: The requested core algorithm is not present")

    for run in runs:
        for node in nodes(run):
            output    = node.get("output", "")
            agent     = node.get("agent", "UNKNOWN")
            nqs       = node.get("nqs", node.get("node_quality_score", -1))
            is_refine = node.get("operator", "") == "refine"
            depth     = node.get("depth", 1)

            rows.append({
                "run_id":           run.get("run_id"),
                "protocol":         run.get("protocol"),
                "ablation_mode":    run.get("ablation_mode"),
                "agent_type":       agent,
                "nqs":              nqs,
                "is_refine":        int(is_refine),
                "depth":            depth,
                "tokens_used":      node.get("tokens_used", 0),
                "latency_ms":       node.get("latency_ms", 0),
                "ai_warning":       int(bool(AI_WARNING_PATTERN.search(output))),
                "placeholder_leak": int(bool(PLACEHOLDER_PATTERN.search(output))),
                "db_refusal":       int(bool(DB_REFUSAL_PATTERN.search(output))),
                "format_ok":        int(not bool(PLACEHOLDER_PATTERN.search(output))),
            })

    # Aggregate per agent type
    agent_agg: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["is_refine"] == 0 and row["nqs"] >= 0:   # first-pass only
            agent_agg[row["agent_type"]]["nqs_first"].append(row["nqs"])
        if row["nqs"] >= 0:
            agent_agg[row["agent_type"]]["all_nqs"].append(row["nqs"])
        agent_agg[row["agent_type"]]["ai_warnings"].append(row["ai_warning"])
        agent_agg[row["agent_type"]]["placeholder_leaks"].append(row["placeholder_leak"])
        agent_agg[row["agent_type"]]["db_refusals"].append(row["db_refusal"])
        # First pass success: non-refine node with NQS >= 0.75 (normalised 0.0–1.0)
        if row["is_refine"] == 0:
            agent_agg[row["agent_type"]]["first_pass_success"].append(
                1 if row["nqs"] >= 0.75 else 0
            )

    agent_summary: dict[str, dict] = {}
    for agent, data in agent_agg.items():
        agent_summary[agent] = {
            "mean_nqs":              round(statistics.mean(data["all_nqs"]), 3) if data["all_nqs"] else -1,
            "stdev_nqs":             round(statistics.stdev(data["all_nqs"]), 3) if len(data["all_nqs"]) > 1 else 0,
            "first_pass_success_rate": round(statistics.mean(data["first_pass_success"]), 3) if data["first_pass_success"] else -1,
            "hallucination_rate":    round(statistics.mean(data["ai_warnings"]), 3) if data["ai_warnings"] else 0,
            "placeholder_leak_rate": round(statistics.mean(data["placeholder_leaks"]), 3) if data["placeholder_leaks"] else 0,
            "db_refusal_rate":       round(statistics.mean(data["db_refusals"]), 3) if data["db_refusals"] else 0,
            "n_nodes":               len(data["all_nqs"]),
        }

    return rows, agent_summary


# ── Layer 3: Agent-to-agent communication metrics ────────────────────────────

def compute_L3(runs: list[dict]) -> tuple[list[dict], dict]:
    """
    Metrics: CRS, CUS (+ U-curve data), DSR, ΔQ, CE, token compression ratio,
             protected atom preservation rate, Information Gain per edge.
    """
    print("[L3] Computing communication metrics...")
    rows = []

    for run in runs:
        t    = tele(run)
        mode = run.get("ablation_mode", "unknown")
        gts  = t.get("final_judge_score", -1)

        for edge in edges(run):
            # Fix 6: single token field — what was actually passed on the wire
            context_tokens = edge.get("context_token_count", 0)
            # Fix 7: context_relevance (was context_relevance_score)
            crs            = edge.get("context_relevance",
                             edge.get("crs", -1))
            cus            = edge.get("cus",
                             edge.get("context_utilization_score", -1))
            # Fix 8: dependency_accuracy (was dependency_satisfaction_rate)
            dsr            = edge.get("dependency_accuracy",
                             edge.get("dsr", -1))
            prot_total     = edge.get("protected_atoms_count", 0)
            prot_preserved = edge.get("protected_atoms_preserved", 0)
            compress_ms    = edge.get("compression_time_ms", 0)

            # Compression ratio: compare context_token_count across modes
            # (computed in the mode-aggregation step below; set 1.0 as default here)
            compression_ratio = 1.0

            # Protected atom preservation rate (1.0 = perfect)
            prot_preservation = (
                prot_preserved / prot_total if prot_total > 0 else -1
            )

            rows.append({
                "run_id":              run.get("run_id"),
                "protocol":            run.get("protocol"),
                "ablation_mode":       mode,
                "query_id":            run.get("query_id"),
                "from_agent":          edge.get("from_agent", ""),
                "to_agent":            edge.get("to_agent", ""),
                "edge_pair":           f"{edge.get('from_agent','?')}→{edge.get('to_agent','?')}",
                "context_token_count": context_tokens,
                # keep old column names in output CSV for visualiser compatibility
                "raw_tokens":          context_tokens,
                "injected_tokens":     context_tokens,
                "compression_ratio":   round(compression_ratio, 3),
                "crs":                 crs,
                "cus":                 cus,
                "dsr":                 dsr,
                "run_gts":             gts,
                "protected_atoms_total":     prot_total,
                "protected_atoms_preserved": prot_preserved,
                "prot_preservation_rate":    round(prot_preservation, 3) if prot_preservation >= 0 else -1,
                "compression_latency_ms":    compress_ms,
            })

    # Aggregate per mode for comparison table
    mode_agg: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        m = row["ablation_mode"]
        for metric in ["crs", "cus", "compression_ratio", "crs"]:
            v = row[metric]
            if v is not None and v >= 0:
                mode_agg[m][metric].append(v)
        if row["dsr"] in [0, 1]:
            mode_agg[m]["dsr"].append(row["dsr"])
        if row["prot_preservation_rate"] >= 0:
            mode_agg[m]["prot_preservation_rate"].append(row["prot_preservation_rate"])

    # ── Post-process: compute compression ratio per edge pair ──────────────────
    # Group mean context_token_count by (query_id, edge_pair, mode)
    token_by_key: dict[tuple, list] = defaultdict(list)
    for row in rows:
        token_by_key[(row["query_id"], row["edge_pair"], row["ablation_mode"])].append(
            row["context_token_count"]
        )
    for row in rows:
        raw_mean  = statistics.mean(
            token_by_key.get((row["query_id"], row["edge_pair"], "raw"), [0]) or [0]
        )
        adap_mean = statistics.mean(
            token_by_key.get((row["query_id"], row["edge_pair"], "adaptive"), [0]) or [0]
        )
        if row["ablation_mode"] == "adaptive" and adap_mean > 0 and raw_mean > 0:
            row["compression_ratio"] = round(raw_mean / adap_mean, 3)
        elif row["ablation_mode"] == "raw":
            row["compression_ratio"] = 1.0
        # none mode: no communication, leave as 1.0

    # Compute ΔQ (information gain) per edge
    edge_gts_by_key: dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = (row["query_id"], row["edge_pair"])
        edge_gts_by_key[(key, row["ablation_mode"])].append(row["run_gts"])

    delta_q_rows = []
    qids  = set(r["query_id"]  for r in rows)
    pairs = set(r["edge_pair"] for r in rows)
    for qid in qids:
        for pair in pairs:
            adaptive_gtss = edge_gts_by_key.get(((qid, pair), "adaptive"), [])
            none_gtss     = edge_gts_by_key.get(((qid, pair), "none"),     [])
            if adaptive_gtss and none_gtss:
                delta_q = statistics.mean(adaptive_gtss) - statistics.mean(none_gtss)
                # Fix 6: use context_token_count (not old injected_tokens)
                adaptive_tokens = statistics.mean([
                    r["context_token_count"] for r in rows
                    if r["query_id"] == qid and r["edge_pair"] == pair
                    and r["ablation_mode"] == "adaptive" and r["context_token_count"] > 0
                ] or [1])
                ce = (delta_q / (adaptive_tokens / 1000)) if adaptive_tokens > 0 else 0
                delta_q_rows.append({
                    "query_id":                   qid,
                    "edge_pair":                  pair,
                    "delta_q":                    round(delta_q, 3),
                    "mean_adaptive_tokens":        round(adaptive_tokens, 1),
                    "communication_efficiency_ce": round(ce, 4),
                })

    comm_summary = {}
    for mode, data in mode_agg.items():
        comm_summary[mode] = {
            k: round(statistics.mean(v), 3) for k, v in data.items() if v
        }
    comm_summary["delta_q_by_edge"] = delta_q_rows

    return rows, comm_summary


# ── Layer 4: Knowledge utilization ───────────────────────────────────────────

def compute_L4(runs: list[dict]) -> tuple[list[dict], dict]:
    """
    Metrics: DB hit/miss rate, knowledge coverage, grounding rate (sampled),
             retrieval quality per agent.
    """
    print("[L4] Computing knowledge utilization metrics...")
    rows = []

    DB_REFUSAL    = re.compile(r"# ERROR: The requested core algorithm is not present")
    AI_WARNING    = re.compile(r"⚠️\s*AI EXTRAPOLATION WARNING", re.I)

    for run in runs:
        for node in nodes(run):
            output    = node.get("output", "")
            agent     = node.get("agent", "UNKNOWN")
            retrieved = node.get("retrieved_chunks", node.get("retrieval_hits", -1))

            db_miss     = bool(DB_REFUSAL.search(output))
            ai_warn     = bool(AI_WARNING.search(output))
            db_hit      = not db_miss and agent in ("RAG", "SCHOLAR", "CODE", "CLOUD")

            rows.append({
                "run_id":         run.get("run_id"),
                "protocol":       run.get("protocol"),
                "ablation_mode":  run.get("ablation_mode"),
                "agent_type":     agent,
                "db_hit":         int(db_hit),
                "db_miss":        int(db_miss),
                "ai_extrapolated":int(ai_warn),
                "retrieved_chunks": retrieved,
                "nqs":            node.get("nqs", -1),
            })

    # Coverage rate = hits / (hits + misses)
    total_hits   = sum(r["db_hit"]  for r in rows if r["agent_type"] in ("RAG","SCHOLAR","CODE","CLOUD"))
    total_misses = sum(r["db_miss"] for r in rows if r["agent_type"] in ("RAG","SCHOLAR","CODE","CLOUD"))
    total_ext    = sum(r["ai_extrapolated"] for r in rows)

    summary = {
        "knowledge_coverage_rate": round(total_hits  / max(total_hits + total_misses, 1), 3),
        "db_refusal_rate":         round(total_misses / max(total_hits + total_misses, 1), 3),
        "ai_extrapolation_rate":   round(total_ext    / max(len(rows), 1), 3),
        "total_nodes_evaluated":   len(rows),
    }

    # Grounding rate via LLM judge — sample up to 5 runs
    # Judge checks if final answer claims are traceable to retrieved content
    grounding_scores = []
    sample_runs = [r for r in runs if r.get("protocol") == "A_ablation"
                   and r.get("ablation_mode") == "adaptive"][:5]

    for run in sample_runs:
        t            = tele(run)
        final_answer = t.get("final_answer", "")
        retrieved    = " ".join([n.get("retrieved_context", "") for n in nodes(run)])
        if not final_answer or not retrieved:
            continue
        cache_key = f"grounding_{run.get('run_id')}"
        prompt = (
            "You are a scientific grounding evaluator.\n"
            f"RETRIEVED CONTEXT (truncated):\n{retrieved[:1500]}\n\n"
            f"FINAL ANSWER (truncated):\n{final_answer[:1500]}\n\n"
            "Score 0–10: How well are the factual claims in the answer "
            "supported by the retrieved context? "
            "Reply with just a number like: 7.5/10"
        )
        score = judge_score(prompt, cache_key)
        if score >= 0:
            grounding_scores.append(score)

    if grounding_scores:
        summary["mean_grounding_score"]  = round(statistics.mean(grounding_scores), 3)
        summary["grounding_sample_size"] = len(grounding_scores)

    return rows, summary


# ── Layer 5: Self-healing evaluation ─────────────────────────────────────────

def compute_L5(runs: list[dict]) -> tuple[list[dict], dict]:
    """
    Metrics: Recovery rate, recovery depth distribution, Reflexion quality,
             false positive REFINE rate, cascading failure rate,
             Diagnostician targeting accuracy.
    """
    print("[L5] Computing self-healing metrics...")
    rows = []
    reflexion_quality_scores = []

    for run in runs:
        t   = tele(run)
        all_nodes = nodes(run)

        # Map node_id → nqs for cascade detection
        nqs_map = {n.get("node_id", ""): n.get("nqs", -1) for n in all_nodes}

        # Find refine nodes and their parent failures
        refine_nodes = [n for n in all_nodes if n.get("operator", "") == "refine"]
        first_nodes  = [n for n in all_nodes if n.get("operator", "") != "refine"]

        for node in first_nodes:
            nqs  = node.get("nqs", -1)
            fail = (nqs >= 0 and nqs < 0.75)   # quality gate threshold (NQS is 0.0–1.0)

            # Was there a successful REFINE after this node?
            agent    = node.get("agent", "")
            matching = [r for r in refine_nodes if r.get("agent") == agent]
            recovered_nqs = max([r.get("nqs", -1) for r in matching], default=-1)

            recovery_attempted = len(matching) > 0
            recovery_success   = recovered_nqs >= 0.8 if recovery_attempted else False
            recovery_depth     = len(matching)

            # False positive: REFINE triggered but original NQS was actually acceptable (>=7.5)
            false_positive_refine = recovery_attempted and not fail

            rows.append({
                "run_id":                run.get("run_id"),
                "protocol":              run.get("protocol"),
                "ablation_mode":         run.get("ablation_mode"),
                "agent_type":            agent,
                "node_id":               node.get("node_id", ""),
                "original_nqs":          nqs,
                "failed_quality_gate":   int(fail),
                "recovery_attempted":    int(recovery_attempted),
                "recovery_success":      int(recovery_success),
                "recovered_nqs":         recovered_nqs,
                "recovery_depth":        recovery_depth,
                "false_positive_refine": int(false_positive_refine),
            })
            # Reflexion quality: judge the <reflection> block content
            if recovery_attempted:
                for refine_node in matching:
                    reflection_text = _extract_reflection(refine_node.get("output", ""))
                    if reflection_text:
                        ck   = f"reflexion_quality_{refine_node.get('node_id','')}"
                        orig_critique = node.get("judge_critique", "unknown")
                        prompt = (
                            "You are evaluating the quality of a self-reflection block.\n"
                            f"ORIGINAL JUDGE CRITIQUE:\n{orig_critique}\n\n"
                            f"AGENT REFLECTION:\n{reflection_text}\n\n"
                            "Score 0-10: Does the reflection correctly identify the "
                            "error described in the critique and propose a concrete fix? "
                            "Reply with a number like: 7.5/10"
                        )
                        score = judge_score(prompt, ck)
                        if score >= 0:
                            reflexion_quality_scores.append({
                                "run_id":    run.get("run_id"),
                                "agent":     agent,
                                "rq_score":  score,
                                "recovered": int(recovery_success),
                            })

        # Cascading failure: a node fails AND the downstream dependent node also fails
        cascades = 0
        for edge in edges(run):
            from_nqs = nqs_map.get(edge.get("from_node", ""), -1)
            to_nqs   = nqs_map.get(edge.get("to_node",   ""), -1)
            if from_nqs >= 0 and from_nqs < 0.75 and to_nqs >= 0 and to_nqs < 0.75:
                cascades += 1

        # Diagnostician
        diag_triggered = t.get("diagnostician_triggered", False)
        diag_target    = t.get("diagnostician_target_agent", "")
        final_gts      = t.get("final_judge_score", -1)
        diag_success   = final_gts >= 0.8 if diag_triggered else None

        rows.append({   # summary row for this run
            "run_id":                    run.get("run_id"),
            "protocol":                  run.get("protocol"),
            "ablation_mode":             run.get("ablation_mode"),
            "agent_type":                "SYSTEM",
            "node_id":                   "GLOBAL",
            "original_nqs":              final_gts,
            "failed_quality_gate":       int(final_gts >= 0 and final_gts < 0.8),
            "recovery_attempted":        int(diag_triggered),
            "recovery_success":          int(diag_success) if diag_success is not None else -1,
            "recovered_nqs":             final_gts,
            "recovery_depth":            0,
            "false_positive_refine":     0,
            "cascading_failure_count":   cascades,
            "diagnostician_target":      diag_target,
        })

    # Aggregate stats
    agent_rows = [r for r in rows if r["agent_type"] != "SYSTEM"]
    fail_rows  = [r for r in agent_rows if r["failed_quality_gate"] == 1]
    rec_rows   = [r for r in fail_rows  if r["recovery_attempted"] == 1]

    summary = {
        "total_node_failures":      len(fail_rows),
        "recovery_rate":            round(
            len([r for r in rec_rows if r["recovery_success"] == 1]) / max(len(fail_rows), 1), 3
        ),
        "recovery_depth_distribution": {
            str(d): len([r for r in agent_rows if r["recovery_depth"] == d])
            for d in range(0, 5)
        },
        "false_positive_refine_rate": round(
            sum(r["false_positive_refine"] for r in agent_rows) / max(len(agent_rows), 1), 3
        ),
        "mean_reflexion_quality": (
            round(statistics.mean([s["rq_score"] for s in reflexion_quality_scores]), 3)
            if reflexion_quality_scores else -1
        ),
        "reflexion_quality_scores": reflexion_quality_scores,
        "cascading_failure_total":  sum(r.get("cascading_failure_count", 0) for r in rows if r["agent_type"] == "SYSTEM"),
    }
    return rows, summary


def _extract_reflection(text: str) -> str:
    match = re.search(r"<reflection>(.*?)</reflection>", text, re.S | re.I)
    return match.group(1).strip() if match else ""


# ── Layer 6: Cost and latency trade-off ──────────────────────────────────────

def compute_L6(runs: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """
    Metrics: Token cost per query per mode, latency breakdown, Pareto data,
             CUGCR overhead tax, depth budget utilization.
    """
    print("[L6] Computing cost/latency metrics...")
    rows      = []
    pareto    = []

    for run in runs:
        t    = tele(run)
        mode = run.get("ablation_mode", "unknown")
        gts  = t.get("final_judge_score", -1)

        total_tokens  = t.get("total_tokens", 0)
        # Fix 9: same latency_seconds → ms conversion as L1
        if t.get("latency_seconds") is not None:
            latency_ms = t["latency_seconds"] * 1000
        else:
            latency_ms = t.get("observed_latency_ms", run.get("wall_latency_ms", 0))
        depth_budget  = t.get("depth_budget", 0)
        depth_used    = t.get("depth_used",   0)
        compress_ms   = sum(e.get("compression_time_ms", 0) for e in edges(run))
        judge_tokens  = t.get("judge_tokens", 0)

        cugcr_triggered = t.get("cugcr_triggered", False)
        cugcr_overhead_pct = (
            (compress_ms / latency_ms * 100) if cugcr_triggered and latency_ms > 0 else 0
        )

        rows.append({
            "run_id":               run.get("run_id"),
            "protocol":             run.get("protocol"),
            "query_id":             run.get("query_id"),
            "ablation_mode":        mode,
            "gts":                  gts,
            "total_tokens":         total_tokens,
            "judge_tokens":         judge_tokens,
            "latency_ms":           latency_ms,
            "depth_budget":         depth_budget,
            "depth_used":           depth_used,
            "depth_utilization_pct":round(depth_used / max(depth_budget, 1) * 100, 1),
            "cugcr_triggered":      int(cugcr_triggered),
            "cugcr_compress_ms":    compress_ms,
            "cugcr_overhead_pct":   round(cugcr_overhead_pct, 2),
        })

        if gts >= 0 and total_tokens > 0:
            pareto.append({
                "run_id":        run.get("run_id"),
                "ablation_mode": mode,
                "query_id":      run.get("query_id"),
                "total_tokens":  total_tokens,
                "gts":           gts,
                "latency_ms":    latency_ms,
            })

    # Summary per mode (ablation runs only)
    ablation_rows = [r for r in rows if r["protocol"] == "A_ablation"]
    mode_summary  = {}
    for mode in ["none", "raw", "adaptive"]:
        mr = [r for r in ablation_rows if r["ablation_mode"] == mode]
        if mr:
            mode_summary[mode] = {
                "mean_tokens":   round(statistics.mean([r["total_tokens"] for r in mr]), 1),
                "mean_latency":  round(statistics.mean([r["latency_ms"]   for r in mr]), 1),
                "mean_overhead_pct": round(statistics.mean([r["cugcr_overhead_pct"] for r in mr]), 2),
                "mean_depth_util":   round(statistics.mean([r["depth_utilization_pct"] for r in mr]), 1),
            }

    summary = {"per_mode": mode_summary}
    return rows, summary, pareto


# ── Layer 7: Novel metrics ────────────────────────────────────────────────────

def compute_L7(runs: list[dict]) -> tuple[list[dict], dict]:
    """
    Metrics: Router determinism score, query complexity scaling,
             profile sensitivity, context boundary behaviour.
    """
    print("[L7] Computing novel metrics...")
    rows    = []
    summary = {}

    # --- Router determinism (Protocol D) ---
    det_runs = [r for r in runs if r.get("protocol") == "D_determinism"]
    dag_structures = []
    for run in det_runs:
        structure = _dag_fingerprint(tele(run).get("got_dag_nodes", []))
        dag_structures.append(structure)
        rows.append({
            "run_id":    run.get("run_id"),
            "metric":    "router_determinism",
            "rep":       run.get("rep", -1),
            "value":     structure,
            "numeric":   0,    # placeholder, filled below
        })

    if dag_structures:
        unique_structures   = len(set(dag_structures))
        determinism_score   = 1.0 - (unique_structures - 1) / max(len(dag_structures) - 1, 1)
        summary["router_determinism_score"] = round(determinism_score, 3)
        summary["unique_dag_structures"]    = unique_structures
        summary["total_determinism_runs"]   = len(dag_structures)
        for row in rows:
            if row["metric"] == "router_determinism":
                row["numeric"] = determinism_score

    # --- Query complexity scaling (Protocol A) ---
    ablation_runs = [r for r in runs if r.get("protocol") == "A_ablation"
                     and r.get("ablation_mode") == "adaptive"]
    group_gts: dict[str, list] = defaultdict(list)
    for run in ablation_runs:
        group = run.get("query_group", "unknown")
        gts   = tele(run).get("final_judge_score", -1)
        if gts >= 0:
            group_gts[group].append(gts)
            rows.append({
                "run_id":  run.get("run_id"),
                "metric":  "complexity_scaling",
                "rep":     run.get("query_id"),
                "value":   group,
                "numeric": gts,
            })

    complexity_summary = {}
    for group, gtss in group_gts.items():
        complexity_summary[group] = {
            "mean_gts": round(statistics.mean(gtss), 3),
            "n":        len(gtss),
        }
    summary["complexity_scaling"] = complexity_summary

    # --- Profile sensitivity (Protocol E) ---
    prof_runs = [r for r in runs if r.get("protocol") == "E_profile_sensitivity"]
    prof_gts: dict[str, float] = {}
    for run in prof_runs:
        profile = run.get("profile_name", "unknown")
        gts     = tele(run).get("final_judge_score", -1)
        prof_gts[profile] = gts
        rows.append({
            "run_id":  run.get("run_id"),
            "metric":  "profile_sensitivity",
            "rep":     profile,
            "value":   profile,
            "numeric": gts,
        })

    if len(prof_gts) > 1:
        gtss = [v for v in prof_gts.values() if v >= 0]
        summary["profile_sensitivity"] = {
            "gts_by_profile":    prof_gts,
            "gts_range":         round(max(gtss) - min(gtss), 3) if gtss else -1,
            "mean_gts":          round(statistics.mean(gtss), 3) if gtss else -1,
        }

    # --- Context boundary (Protocol F) ---
    bound_runs = [r for r in runs if r.get("protocol") == "F_boundary"]
    for run in bound_runs:
        cugcr_triggered = tele(run).get("cugcr_triggered", False)
        gts             = tele(run).get("final_judge_score", -1)
        rows.append({
            "run_id":  run.get("run_id"),
            "metric":  "boundary_behavior",
            "rep":     run.get("ablation_mode"),
            "value":   "cugcr" if cugcr_triggered else "raw_passthrough",
            "numeric": gts,
        })

    # --- Adversarial failure floor (Protocol C) ---
    adv_runs = [r for r in runs if r.get("protocol") == "C_adversarial"]
    adv_gtss = [tele(r).get("final_judge_score", -1) for r in adv_runs if not r.get("failed")]
    adv_gtss = [g for g in adv_gtss if g >= 0]
    if adv_gtss:
        summary["adversarial_failure_floor"] = {
            "mean_gts":  round(statistics.mean(adv_gtss), 3),
            "min_gts":   min(adv_gtss),
            "max_gts":   max(adv_gtss),
            "n":         len(adv_gtss),
        }
        for run, gts in zip(adv_runs, adv_gtss):
            rows.append({
                "run_id":  run.get("run_id"),
                "metric":  "adversarial_robustness",
                "rep":     run.get("query_id"),
                "value":   "adversarial",
                "numeric": gts,
            })

    return rows, summary


def _dag_fingerprint(dag_nodes: list[dict]) -> str:
    """Create a hashable string fingerprint of DAG agent types and depths."""
    sig = sorted(
        f"{n.get('agent', '?')}@{n.get('depth', 0)}"
        for n in dag_nodes if n.get("operator", "") != "refine"
    )
    return "|".join(sig)


# ── CSV writer helper ─────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"  [skip] no data for {path.name}")
        return
    fieldnames = list(rows[0].keys())
    # Collect any extra keys from other rows
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✔  {path.name}  ({len(rows)} rows)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          FULL SYSTEM EVALUATION — METRICS COMPUTATION   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Reading from : {RAW_DIR.resolve()}")
    print(f"Output to    : {METRICS_DIR.resolve()}")

    runs = load_all_runs()
    if not runs:
        print("No run files found. Run run_full_evaluation.py first.")
        return

    # Compute all layers
    L1_rows, L1_summary  = compute_L1(runs)
    L2_rows, L2_summary  = compute_L2(runs)
    L3_rows, L3_summary  = compute_L3(runs)
    L4_rows, L4_summary  = compute_L4(runs)
    L5_rows, L5_summary  = compute_L5(runs)
    L6_rows, L6_summary, pareto_rows = compute_L6(runs)
    L7_rows, L7_summary  = compute_L7(runs)

    # Write CSVs
    print("\nWriting CSV files...")
    write_csv(L1_rows,    METRICS_DIR / "L1_system.csv")
    write_csv(L2_rows,    METRICS_DIR / "L2_agent.csv")
    write_csv(L3_rows,    METRICS_DIR / "L3_edges.csv")
    write_csv(L4_rows,    METRICS_DIR / "L4_knowledge.csv")
    write_csv(L5_rows,    METRICS_DIR / "L5_selfhealing.csv")
    write_csv(L6_rows,    METRICS_DIR / "L6_cost_latency.csv")
    write_csv(pareto_rows,METRICS_DIR / "L6_pareto.csv")
    write_csv(L7_rows,    METRICS_DIR / "L7_novel.csv")

    # Communication aggregate (from summary)
    comm_agg_rows = []
    for mode, data in L3_summary.items():
        if mode == "delta_q_by_edge":
            continue
        row = {"ablation_mode": mode}
        row.update(data)
        comm_agg_rows.append(row)
    write_csv(comm_agg_rows, METRICS_DIR / "L3_communication.csv")

    # Write master metrics JSON
    all_metrics = {
        "computed_at": datetime.utcnow().isoformat(),
        "total_runs":  len(runs),
        "L1_system":          L1_summary,
        "L2_agent":           L2_summary,
        "L3_communication":   L3_summary,
        "L4_knowledge":       L4_summary,
        "L5_selfhealing":     L5_summary,
        "L6_cost_latency":    L6_summary,
        "L7_novel":           L7_summary,
    }
    with open(METRICS_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  ✔  all_metrics.json")

    # Print high-level summary to console
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\n[L1] System-level")
    for mode, stats in L1_summary.get("per_mode", {}).items():
        print(f"  {mode:10s} | GTS={stats.get('mean_gts','?'):.2f}  TSR={stats.get('tsr','?'):.2f}  n={stats.get('n','?')}")
    wr = L1_summary.get("win_rates", {})
    print(f"  Win rates → ADAPTIVE:{wr.get('adaptive_win_rate',0):.0%}  "
          f"RAW:{wr.get('raw_win_rate',0):.0%}  NONE:{wr.get('none_win_rate',0):.0%}")

    print("\n[L2] Agent first-pass success rates")
    for agent, stats in L2_summary.items():
        print(f"  {agent:10s} | NQS={stats.get('mean_nqs','?'):.2f}  "
              f"FPS={stats.get('first_pass_success_rate','?'):.2f}  "
              f"Hall={stats.get('hallucination_rate','?'):.2f}")

    print("\n[L3] Communication (adaptive mode)")
    adap = L3_summary.get("adaptive", {})
    print(f"  CRS={adap.get('crs','?')}  CUS={adap.get('cus','?')}  "
          f"Compression×={adap.get('compression_ratio','?')}")

    print("\n[L5] Self-healing")
    print(f"  Recovery rate  : {L5_summary.get('recovery_rate','?'):.1%}")
    print(f"  FP REFINE rate : {L5_summary.get('false_positive_refine_rate','?'):.1%}")
    print(f"  Reflexion qual : {L5_summary.get('mean_reflexion_quality','?')}")
    print(f"  Cascades       : {L5_summary.get('cascading_failure_total','?')}")

    print("\n[L7] Novel metrics")
    print(f"  Router determinism : {L7_summary.get('router_determinism_score','?'):.3f}")
    print(f"  Adversarial GTS    : {L7_summary.get('adversarial_failure_floor',{}).get('mean_gts','?')}")

    print(f"\n✅ Done. All metrics saved to {METRICS_DIR.resolve()}")


if __name__ == "__main__":
    main()