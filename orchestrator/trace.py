"""
trace.py — a complete record of one engine run, detailed enough to replay any node.

The engine's telemetry keeps node outputs and judge scores. A trace also keeps what each node *read*:
the exact string sent to its agent, the context atoms offered on each incoming edge and which of them
were passed on, what the agent retrieved, and every LLM call it made. Ablations, cost tables and
error analysis read it; counterfactual replay (drop an atom, rerun downstream) needs it.

One run = one JSON file in SCIWEAVER_TRACE_DIR (default ~/otr_local/traces, outside OneDrive):

  {"run_id", "started", "seconds", "user_query", "config", "final_answer",
   "nodes": {node_id: {"agent", "op", "parents", "task", "agent_input", "edge", "retrievals",
                       "llm_calls", "output", "seconds", ...judge fields}},
   "events": [{"kind": "decomposition" | "retry" | ..., ...}]}

Recording is a no-op outside a run, so agents can call record_* unconditionally. The run and the
current node are context variables: asyncio tasks and asyncio.to_thread inherit them, so calls made
while a node executes are attributed to that node.

Usage (engine):
    run = trace.start_run(user_query, config_snapshot)
    token = trace.enter_node(node_id, agent=..., op=..., parents=[...], task=...)
    ... trace.record_node(agent_input=..., output=...) ...
    trace.exit_node(token)
    trace.end_run(final_answer=...)            # writes the file, returns its path
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path(os.environ.get("SCIWEAVER_TRACE_DIR", Path.home() / "otr_local" / "traces"))
OUTSIDE_NODE = "_run"  # calls made outside any node (decomposition, synthesis, final judge)

_run: contextvars.ContextVar["RunTrace | None"] = contextvars.ContextVar("sciweaver_trace_run", default=None)
_node: contextvars.ContextVar[str] = contextvars.ContextVar("sciweaver_trace_node", default=OUTSIDE_NODE)


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


class RunTrace:
    def __init__(self, user_query: str, config: dict):
        self.run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.started = time.time()
        self.user_query = user_query
        self.config = config
        self.nodes: dict[str, dict] = {}
        self.events: list[dict] = []
        self._lock = threading.Lock()  # LLM calls are recorded from worker threads

    def node(self, node_id: str) -> dict:
        with self._lock:
            return self.nodes.setdefault(node_id, {"retrievals": [], "llm_calls": []})

    def to_dict(self, **extra) -> dict:
        return {"run_id": self.run_id, "started": datetime.fromtimestamp(self.started).isoformat(),
                "seconds": round(time.time() - self.started, 2), "user_query": self.user_query,
                "config": self.config, **extra, "nodes": self.nodes, "events": self.events}


def start_run(user_query: str, config: dict | None = None) -> RunTrace:
    run = RunTrace(user_query, config or {})
    _run.set(run)
    _node.set(OUTSIDE_NODE)
    return run


def current() -> RunTrace | None:
    return _run.get()


def enter_node(node_id: str, **fields) -> contextvars.Token | None:
    run = _run.get()
    if run is None:
        return None
    rec = run.node(node_id)
    rec.update(fields)
    rec["_t0"] = time.time()
    return _node.set(node_id)


def exit_node(token: contextvars.Token | None) -> None:
    run = _run.get()
    if run is None or token is None:
        return
    rec = run.node(_node.get())
    if "_t0" in rec:
        rec["seconds"] = round(time.time() - rec.pop("_t0"), 2)
    _node.reset(token)


def record_node(node_id: str | None = None, **fields) -> None:
    """Merge fields into the current node (or node_id)."""
    run = _run.get()
    if run is not None:
        run.node(node_id or _node.get()).update(fields)


def record_edge(offered: list[dict], kept_ids: list[str], policy: str, text: str, **meta) -> None:
    """The context on the current node's incoming edge: every atom offered, which were passed on."""
    record_node(edge={"policy": policy, "offered": offered, "kept": kept_ids, "text": text, **meta})


def record_retrieval(source: str, query: str, items: list[dict]) -> None:
    """What an agent retrieved: items like {"id", "title", "score"} (no full documents)."""
    run = _run.get()
    if run is not None:
        run.node(_node.get())["retrievals"].append({"source": source, "query": query, "items": items})


def record_tool_call(tool: str, args: dict, result: dict) -> None:
    """A non-LLM tool call made by the current node (sandbox run, database lookup): short summaries only."""
    run = _run.get()
    if run is not None:
        run.node(_node.get()).setdefault("tool_calls", []).append({"tool": tool, "args": args, "result": result})


def record_llm_call(model: str, input_tokens: int, output_tokens: int, prompt: str | None = None) -> None:
    run = _run.get()
    if run is not None:
        call = {"model": model, "input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}
        if prompt is not None:
            call["prompt_hash"] = text_hash(prompt)
        run.node(_node.get())["llm_calls"].append(call)


def event(kind: str, **data) -> None:
    run = _run.get()
    if run is not None:
        with run._lock:
            run.events.append({"kind": kind, "t": round(time.time() - run.started, 2), **data})


def end_run(**extra) -> Path | None:
    """Write the trace and detach it from the context. Returns the file path."""
    run = _run.get()
    if run is None:
        return None
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{run.run_id}.json"
    path.write_text(json.dumps(run.to_dict(**extra), indent=1, default=str))
    _run.set(None)
    return path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
