"""
eval_routing.py — Decomposer + Router evaluation (Structural + LLM-as-Judge)
=============================================================================
Structural checks  (free, from telemetry):
  - Agent validity: assigned agent is a known agent
  - Dependency soundness: all parent_ids reference existing nodes
  - Placeholder leakage: no unresolved [result of goal N] in outputs
  - Failure modes: distribution across nodes
  - Node quality scores per agent

LLM-as-Judge (one call per run):
  - Was each subtask routed to the correct agent?
  - Is the decomposition complete — all query aspects covered?
  - Overall routing score 0–10

Usage:
    python scripts/eval/eval_routing.py
    python scripts/eval/eval_routing.py --dir historic_telemetry/raw_telemetry-lite --no-llm
    python scripts/eval/eval_routing.py --out results/routing_eval.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import Config

KNOWN_AGENTS = {"code", "cloud", "rag", "scholar"}

ROUTING_JUDGE_PROMPT = """You are evaluating the task decomposition and agent routing of a multi-agent bioinformatics AI pipeline.

Original query: "{original_query}"

The system decomposed this into {n_tasks} subtask(s) and assigned each to an agent:
{tasks_list}

Available agents and their roles:
- code    : writes Python/Bash code, implements algorithms
- cloud   : designs AWS/cloud infrastructure, writes Terraform/IaC
- rag     : retrieves and explains bioinformatics knowledge from a local knowledge base
- scholar : finds real researchers and papers via Semantic Scholar API

Evaluate each subtask:
1. Was the correct agent assigned for this specific subtask?
2. Does the full set of subtasks collectively address ALL aspects of the original query?

Return ONLY this JSON (no extra text):
{{
  "tasks": [
    {{"task_id": <id>, "assigned": "<agent>", "correct": <true/false>, "suggested": "<agent>", "reason": "<one phrase>"}}
  ],
  "decomposition_complete": <true/false>,
  "missing_aspects": ["<aspect if any>"],
  "routing_score": <0-10>
}}
"""


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def check_structural(run: dict) -> dict:
    et    = run.get("engine_telemetry", {})
    tasks = et.get("initial_tasks", [])
    nodes = et.get("got_dag_nodes", [])
    edges = et.get("communication_edges", [])
    if isinstance(edges, dict):
        edges = edges.get("edges", edges.get("logs", []))

    valid_task_ids  = {t["id"] for t in tasks}
    valid_node_ids  = {n["node_id"] for n in nodes}

    # 1. Agent validity
    invalid_agents = [t for t in tasks if t.get("agent") not in KNOWN_AGENTS]

    # 2. Dependency soundness (parent_ids all exist)
    broken_deps = []
    for n in nodes:
        for pid in (n.get("parent_ids") or []):
            if pid not in valid_node_ids:
                broken_deps.append({"node": n["node_id"][:6], "missing_parent": pid[:6]})

    # 3. Placeholder leakage in node outputs
    leaky_nodes = [
        n["node_id"][:6] for n in nodes
        if not n.get("dependency_accuracy", True)
    ]

    # 4. Failure modes
    failure_modes = defaultdict(int)
    for n in nodes:
        fm = n.get("failure_mode") or "none"
        failure_modes[fm] += 1

    # 5. Node quality scores per agent
    quality_by_agent = defaultdict(list)
    for n in nodes:
        qs = n.get("quality_score")
        if qs is not None:
            quality_by_agent[n.get("agent", "?")].append(float(qs))

    # 6. Edge dependency accuracy
    edge_dep_acc = [e.get("dependency_accuracy", True) for e in edges]
    edge_dep_rate = sum(edge_dep_acc) / len(edge_dep_acc) * 100 if edge_dep_acc else 100.0

    return {
        "n_tasks":           len(tasks),
        "n_nodes":           len(nodes),
        "invalid_agents":    invalid_agents,
        "broken_deps":       broken_deps,
        "leaky_nodes":       leaky_nodes,
        "failure_modes":     dict(failure_modes),
        "quality_by_agent":  {a: round(sum(v)/len(v), 3) for a, v in quality_by_agent.items()},
        "edge_dep_acc_pct":  round(edge_dep_rate, 1),
        # Structural pass = no invalid agents, no broken deps, no leaky nodes
        "structural_pass":   not invalid_agents and not broken_deps and not leaky_nodes,
    }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

async def judge_routing(client, original_query: str, tasks: list, sem) -> dict:
    tasks_list = "\n".join(
        f"  Task {t['id']} → [{t['agent'].upper()}]: {t['query'][:120]}"
        for t in tasks
    )
    prompt = ROUTING_JUDGE_PROMPT.format(
        original_query=original_query,
        n_tasks=len(tasks),
        tasks_list=tasks_list,
    )
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=Config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0].strip().lstrip("json")
            return json.loads(text)
        except Exception as e:
            return {"tasks": [], "decomposition_complete": None,
                    "missing_aspects": [], "routing_score": 0,
                    "error": str(e)}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_runs(tel_dir: Path, mode: str = "adaptive") -> list[dict]:
    runs = []
    for f in sorted(tel_dir.glob("run_*.json")):
        try:
            d = json.loads(f.read_text())
            if not mode or d.get("ablation_mode") == mode:
                runs.append(d)
        except Exception:
            pass
    return runs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], use_llm: bool):
    n = len(results)
    if not n:
        return

    # Structural
    struct_pass = sum(r["structural"]["structural_pass"] for r in results)
    dep_acc     = sum(r["structural"]["edge_dep_acc_pct"] for r in results) / n
    leaky       = sum(len(r["structural"]["leaky_nodes"]) for r in results)
    all_fmodes  = defaultdict(int)
    for r in results:
        for fm, c in r["structural"]["failure_modes"].items():
            all_fmodes[fm] += c
    all_quality = defaultdict(list)
    for r in results:
        for agent, q in r["structural"]["quality_by_agent"].items():
            all_quality[agent].append(q)

    print(f"\n{'═'*62}")
    print(f"  ROUTING EVALUATION  (N={n} runs)")
    print(f"{'═'*62}")
    print(f"\n── Structural ───────────────────────────────────────────────")
    print(f"  Structural pass rate : {struct_pass}/{n}  ({struct_pass/n*100:.0f}%)")
    print(f"  Dep. accuracy (edges): {dep_acc:.1f}%")
    print(f"  Placeholder leaks    : {leaky} nodes")
    print(f"\n  Failure modes:")
    for fm, c in sorted(all_fmodes.items(), key=lambda x: -x[1]):
        print(f"    {fm:<20} {c:>4}  ({c/sum(all_fmodes.values())*100:.0f}%)")
    print(f"\n  Node quality score (mean per agent):")
    for agent in sorted(all_quality):
        vals = all_quality[agent]
        print(f"    {agent:<10} {sum(vals)/len(vals):.3f}  (n={len(vals)})")

    if use_llm:
        llm_results = [r for r in results if r.get("llm")]
        if llm_results:
            routing_scores  = [r["llm"]["routing_score"] for r in llm_results]
            comp_flags      = [r["llm"]["decomposition_complete"] for r in llm_results
                               if r["llm"]["decomposition_complete"] is not None]
            # Per-task correctness
            correct_tasks   = []
            wrong_routes    = defaultdict(int)  # (assigned, suggested)
            for r in llm_results:
                for t in r["llm"].get("tasks", []):
                    correct_tasks.append(t.get("correct", False))
                    if not t.get("correct", True):
                        key = f"{t.get('assigned','?')} → {t.get('suggested','?')}"
                        wrong_routes[key] += 1

            print(f"\n── LLM Judge ────────────────────────────────────────────────")
            print(f"  Routing score (mean) : {sum(routing_scores)/len(routing_scores):.1f}/10")
            if correct_tasks:
                acc = sum(correct_tasks)/len(correct_tasks)*100
                print(f"  Task routing accuracy: {sum(correct_tasks)}/{len(correct_tasks)}  ({acc:.0f}%)")
            if comp_flags:
                comp_rate = sum(comp_flags)/len(comp_flags)*100
                print(f"  Decomp. completeness : {sum(comp_flags)}/{len(comp_flags)}  ({comp_rate:.0f}%)")
            if wrong_routes:
                print(f"\n  Misrouting patterns:")
                for pair, c in sorted(wrong_routes.items(), key=lambda x: -x[1]):
                    print(f"    {pair:<25} {c}×")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(tel_dir: Path, out_path: Path, use_llm: bool,
               mode: str, concurrency: int):
    runs = load_runs(tel_dir, mode)
    print(f"\n  {len(runs)} {mode or 'all-mode'} runs  ({tel_dir.name})")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if use_llm else None
    sem    = asyncio.Semaphore(concurrency)

    # Build judge tasks upfront for concurrent execution
    judge_coros = []
    for run in runs:
        et    = run.get("engine_telemetry", {})
        query = run.get("query", et.get("user_query", ""))
        tasks = et.get("initial_tasks", [])
        if use_llm and tasks:
            judge_coros.append(judge_routing(client, query, tasks, sem))
        else:
            judge_coros.append(None)

    if use_llm:
        llm_results = await asyncio.gather(*[c for c in judge_coros if c])
    else:
        llm_results = []

    llm_iter = iter(llm_results)
    results  = []
    for run in runs:
        qid      = run.get("query_id", "?")
        et       = run.get("engine_telemetry", {})
        query    = run.get("query", et.get("user_query", ""))[:80]
        struct   = check_structural(run)
        llm_out  = next(llm_iter) if use_llm and et.get("initial_tasks") else None

        status = "✓" if struct["structural_pass"] else "✗"
        rs     = f"  routing={llm_out['routing_score']}/10" if llm_out else ""
        print(f"  {qid} | struct={status} | dep_acc={struct['edge_dep_acc_pct']}%{rs}")

        results.append({
            "query_id":  qid,
            "query":     query,
            "structural": struct,
            "llm":       llm_out,
        })

    print_summary(results, use_llm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  → {out_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=ROOT / "historic_telemetry" / "raw_telemetry-lite")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "routing_eval.json")
    p.add_argument("--mode", default="adaptive",
                   help="ablation mode to evaluate (default: adaptive, '' for all)")
    p.add_argument("--no-llm", action="store_true",
                   help="structural checks only, skip LLM judge")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()
    asyncio.run(main(args.dir, args.out, not args.no_llm, args.mode, args.concurrency))
