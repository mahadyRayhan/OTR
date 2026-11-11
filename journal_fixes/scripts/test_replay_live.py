"""
test_replay_live.py — live check of replay.py on a recorded run (LLM calls; a few cents).

  1. replay with no change: same first-pass answer, no cache misses (except live web retrieval)
  2. replace one root node's output: only that node's descendants change, and only they miss the cache

The run must have been recorded with the cache on (SCIWEAVER_LLM_CACHE=1).
Usage: SCIWEAVER_LLM_CACHE=1 python journal_fixes/scripts/test_replay_live.py ~/otr_local/traces/<run>.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCIWEAVER_LLM_CACHE", "1")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


async def main(path: Path) -> None:
    from orchestrator import trace

    t = trace.load(path)
    if t["config"].get("SCIWEAVER_INDEX_DIR"):  # the same indexes as the run, before the agents are imported
        os.environ["SCIWEAVER_INDEX_DIR"] = t["config"]["SCIWEAVER_INDEX_DIR"]
    from orchestrator.engine import OrchestratorEngine
    from orchestrator.replay import _graph, replay
    eng = OrchestratorEngine()

    same = await replay(eng, t)
    print("unchanged replay:", json.dumps({"answer_identical": same["answer"] == same["recorded_answer"],
                                           "changed_nodes": same["changed"], **same["llm"]}))

    graph = _graph(t)
    children = {}
    for n in graph:
        for p in n.get("parents", []):
            children.setdefault(p, set()).add(n["node_id"])
    root = next(n for n in graph if not n.get("parents") and n["node_id"] in children)
    reach, todo = set(), [root["node_id"]]
    while todo:
        for c in children.get(todo.pop(), ()):
            if c not in reach:
                reach.add(c)
                todo.append(c)
    alt = await replay(eng, t, outputs={root["node_id"]: "No information was found for this sub-task."})
    unexpected = [c for c in alt["changed"] if c != root["node_id"] and c not in reach]
    print("override replay:", json.dumps({"overridden": root["node_id"], "descendants": sorted(reach),
                                          "changed_nodes": alt["changed"], "changed_outside_descendants": unexpected,
                                          **alt["llm"]}))
    print("answers:", repr(same["answer"][-80:]), "|", repr(alt["answer"][-80:]))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).expanduser()))
