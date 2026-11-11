"""
replay.py — rerun a recorded run's DAG with some inputs or outputs changed.

The plan, the graph the GoT run grew (branch, refine and aggregate nodes) and every node's task are
taken from the trace and held fixed; the judge is not rerun. Each node is executed again with the
engine's own _execute_node, in the order the run executed it, so its input is built exactly as in the
run. With the LLM cache on (llm_cache.py), a node whose prompts did not change is served from the cache:
replaying with no changes reproduces the run's first-pass answer at no cost, and a change costs only the
nodes it reaches.

Overrides:
  outputs={node_id: text}    the node is not run; downstream nodes see this output
  inputs={node_id: text}     the node's agent is called on exactly this input (context routing skipped)
  policy="raw"               the context policy for every routed edge (default: the run's)

The result says which nodes changed and how many LLM calls missed the cache. Retrieval that goes to
live web APIs (Semantic Scholar, Europe PMC) can return different results later; that shows up as
cache misses in nodes nobody changed.

Usage:
    eng = OrchestratorEngine()
    res = await replay(eng, "~/otr_local/traces/<run>.json", outputs={"4b87...": "..."})
    res["answer"], res["changed"], res["llm"]
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from orchestrator import llm_cache, trace
from orchestrator.adaptive_got_dag import GoTOp
from orchestrator.context_layer import ArtifactStore

SKIP_OPS = {"judge", "retry"}


def _graph(t: dict) -> list[dict]:
    """The run's DAG nodes in execution order (no judge or global-retry pseudo-nodes)."""
    return [dict(n, node_id=nid) for nid, n in t["nodes"].items()
            if nid != trace.OUTSIDE_NODE and n.get("op") not in SKIP_OPS and "agent" in n]


def _event(t: dict, kind: str) -> dict:
    return next((e for e in t["events"] if e["kind"] == kind), {})


async def replay(engine, recorded: str | Path | dict, *, outputs: dict | None = None, inputs: dict | None = None,
                 policy: str | None = None, record: bool = True) -> dict:
    from profiler_agent.agent import classify_query

    t = recorded if isinstance(recorded, dict) else trace.load(Path(recorded).expanduser())
    # The agents read their indexes when imported; other indexes would retrieve other documents.
    if os.environ.get("SCIWEAVER_INDEX_DIR") != t["config"].get("SCIWEAVER_INDEX_DIR"):
        raise RuntimeError(f"run used SCIWEAVER_INDEX_DIR={t['config'].get('SCIWEAVER_INDEX_DIR')!r}; set it "
                           f"before importing the engine (now {os.environ.get('SCIWEAVER_INDEX_DIR')!r})")
    outputs, inputs = outputs or {}, inputs or {}
    user_query, mode = t["user_query"], t["config"].get("ablation_mode", "adaptive")
    policy = policy or t["config"].get("context_policy", "cugcr")
    style = _event(t, "synthesis_style").get("text", "")

    # Engine state that _execute_agent and the policies read, as at the start of the run.
    scores = classify_query(user_query)
    engine._user_query, engine._artifacts, engine._node_metadata = user_query, ArtifactStore(), {}
    engine._current_query_dim = max(scores, key=scores.get)

    before = llm_cache.stats()
    if record:
        trace.start_run(user_query, {**t["config"], "replay_of": t["run_id"], "replay_policy": policy,
                                     "replay_outputs": sorted(outputs), "replay_inputs": sorted(inputs)})

    async def synthesize(outs: list) -> str:
        return await engine._synthesize(user_query, outs, style)

    nodes: dict[str, SimpleNamespace] = {}
    changed = []
    for rec in _graph(t):
        nid = rec["node_id"]
        node = SimpleNamespace(node_id=nid, op=GoTOp(rec["op"]), depth=rec.get("depth", 1),
                               parent_ids=list(rec.get("parents", [])), judge_reasoning=rec.get("judge_critique"),
                               task=SimpleNamespace(agent=rec["agent"], query=rec.get("task", "")),
                               dim=rec.get("dim"), result=None)
        parents = [nodes[p] for p in node.parent_ids if p in nodes]
        tok = trace.enter_node(nid, agent=rec["agent"], op=rec["op"], parents=node.parent_ids, task=node.task.query)
        try:
            if nid in outputs:
                node.result = outputs[nid]
            elif nid in inputs:
                node.result, _ = await engine._dispatch_agent(rec["agent"], inputs[nid], rec.get("strategy", "cot"),
                                                              rec.get("depth", 1))
            else:
                node.result = await engine._execute_node(node, parents, mode, policy, user_query, synthesize)
            trace.record_node(output=node.result)
        finally:
            trace.exit_node(tok)
        nodes[nid] = node
        if node.result != rec.get("output"):
            changed.append(nid)

    # The answer as the run formed it before any global retry (engine._execute_pipeline).
    aggregates = [n for n in nodes.values() if n.op == GoTOp.AGGREGATE]
    if aggregates:
        answer = aggregates[-1].result
    else:
        children = {p for n in nodes.values() for p in n.parent_ids}
        leaves = [n for n in nodes.values() if n.node_id not in children and n.result]
        got_answer = "\n\n---\n\n".join(f"[{(n.dim or '').upper()}] {n.result}" for n in leaves)
        answer = await synthesize([got_answer])

    after = llm_cache.stats()
    llm = {"cache_hits": after["hits"] - before["hits"], "cache_misses": after["misses"] - before["misses"],
           "cache_enabled": after["enabled"]}
    path = trace.end_run(final_answer=answer, llm_cache=llm) if record else None
    from orchestrator.evaluation.communication_log import comm_tracker
    comm_tracker.clear()
    return {"answer": answer, "outputs": {k: n.result for k, n in nodes.items()}, "changed": changed,
            "llm": llm, "trace_file": str(path) if path else None,
            "recorded_answer": _event(t, "first_pass").get("answer")}
