"""
test_shared_core.py — offline checks of the run trace and the context layer (no LLM calls).

  1. trace: calls made in a node (also from asyncio tasks and worker threads) are attributed to it;
     the file is written and readable
  2. typed atoms: CONSTRAINT, ARTIFACT (pointer + store) and the sensitive flag
  3. policies: none / raw route as before; route_context records offered and kept atoms
  4. the cugcr policy returns exactly what engine._resolve_dependencies returns
  5. sandbox (if Docker is running): runs code, blocks the network, enforces the timeout, keeps the
     file system read-only outside /tmp
  6. lookups (only with --online): NCBI esearch / esummary / efetch

Usage: python journal_fixes/scripts/test_shared_core.py [--online]
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SCIWEAVER_TRACE_DIR"] = tempfile.mkdtemp()

from orchestrator import trace  # noqa: E402
from orchestrator.context_layer import (ArtifactStore, AtomType, get_policy, route_context,  # noqa: E402
                                        typed_atoms)

UPSTREAM = ("The aligner must not be run with more than 8 threads on shared nodes. "
            "BWA-MEM2 indexes the GRCh38 reference in about 1 hour. "
            "The primer ATGCGTACGTTAGCATGCATGCAAGT binds upstream.\n"
            "```python\n" + "x = 1\n" * 500 + "```")


class FakeEngine:
    async def _resolve_dependencies(self, query, depends_on, previous_results, mode, downstream_agent="rag"):
        return f"{query} || CUGCR({','.join(depends_on)})"


def test_trace():
    trace.TRACE_DIR = Path(os.environ["SCIWEAVER_TRACE_DIR"])
    trace.start_run("q", {"MODEL": "m"})
    trace.record_llm_call("planner", 10, 5)  # outside any node

    async def node(nid):
        tok = trace.enter_node(nid, agent="rag")
        trace.record_llm_call("m", 1, 1)
        await asyncio.to_thread(trace.record_llm_call, "m", 2, 2)  # worker thread inherits the node
        trace.record_retrieval("literature", "q", [{"id": "p1"}])
        trace.record_node(output=f"out {nid}")
        trace.exit_node(tok)

    async def main():
        await asyncio.gather(node("n1"), node("n2"))

    asyncio.run(main())
    trace.event("retry", retry_num=1)
    path = trace.end_run(final_answer="A")
    t = trace.load(path)
    assert t["nodes"]["_run"]["llm_calls"][0]["model"] == "planner"
    for nid in ("n1", "n2"):
        n = t["nodes"][nid]
        assert len(n["llm_calls"]) == 2 and n["output"] == f"out {nid}" and n["retrievals"][0]["items"][0]["id"] == "p1"
        assert "seconds" in n and "_t0" not in n
    assert t["final_answer"] == "A" and t["events"][0]["kind"] == "retry" and trace.current() is None
    trace.record_llm_call("m", 1, 1)  # no run: must be a no-op
    print("trace ok:", path.name)


def test_typed_atoms():
    store = ArtifactStore()
    atoms = typed_atoms(UPSTREAM, "node1234abcd", store)
    kinds = {a.type for a in atoms}
    assert AtomType.CONSTRAINT in kinds and AtomType.ARTIFACT in kinds, kinds
    c = next(a for a in atoms if a.type == AtomType.CONSTRAINT)
    assert "must not" in c.content and c.is_protected
    art = next(a for a in atoms if a.type == AtomType.ARTIFACT)
    assert art.content.startswith("[ARTIFACT a1:") and store.get("a1").startswith("```python")
    assert any(a.sensitive for a in atoms) and not c.sensitive
    assert len({a.id for a in atoms}) == len(atoms) and all(a.id.startswith("node1234#") for a in atoms)
    print("typed atoms ok:", sorted(k.value for k in kinds))


def test_policies():
    eng, up = FakeEngine(), {"node1234abcd": UPSTREAM}

    async def run(name):
        trace.start_run("q")
        tok = trace.enter_node("child", agent="code")
        text = await route_context(eng, name, "Write the script.", "code", up, store=ArtifactStore())
        edge = trace.current().nodes["child"]["edge"]
        trace.exit_node(tok)
        trace.end_run()
        return text, edge

    text, edge = asyncio.run(run("none"))
    assert text == "Write the script." and edge["kept"] == [] and edge["policy"] == "none" and edge["offered"]
    text, edge = asyncio.run(run("raw"))
    assert text == f"Write the script.\n\n[UPSTREAM CONTEXT]:\n--- Result ---\n{UPSTREAM}"
    assert set(edge["kept"]) == {a["id"] for a in edge["offered"]}, "raw passes every atom, artifacts by full text"
    text, edge = asyncio.run(run("cugcr"))
    assert text == "Write the script. || CUGCR(node1234abcd)"
    try:
        get_policy("nope")
        raise AssertionError("unknown policy accepted")
    except KeyError:
        pass
    print("policies ok")


def test_sandbox():
    from orchestrator import sandbox
    if not sandbox.available():
        print("sandbox SKIPPED: Docker or the sciweaver-sandbox image is not available")
        return
    r = sandbox.run_python("import Bio, numpy\ns = 'AGATCTTTAGATCT'\nprint(sum(s.startswith('AGATCT', i) for i in range(len(s))))")
    assert r.ok and r.stdout.strip() == "2", r
    r = sandbox.run_python("import urllib.request\nurllib.request.urlopen('https://www.ncbi.nlm.nih.gov', timeout=5)")
    assert not r.ok and ("URLError" in r.stderr or "resolution" in r.stderr.lower()), r.stderr[-200:]
    r = sandbox.run_python("open('/etc/x', 'w')")
    assert not r.ok and "Read-only" in r.stderr, r.stderr[-200:]
    r = sandbox.run_python("open('out.txt', 'w').write('ok'); print(open('data.txt').read())", files={"data.txt": "hi"})
    assert r.ok and r.stdout.strip() == "hi", r
    r = sandbox.run_python("import time\ntime.sleep(30)", timeout=3)
    assert r.timed_out and r.seconds < 20, r
    print("sandbox ok")


def test_lookups():
    from orchestrator import lookups
    ids = lookups.esearch("gene", "BRCA1[sym] AND human[orgn]", 5)
    assert "672" in ids, ids
    assert lookups.esummary("gene", ["672"])["672"]["name"] == "BRCA1"
    fasta = lookups.efetch("nuccore", ["NM_007294.4"], "fasta")
    assert fasta.startswith(">NM_007294"), fasta[:80]
    assert lookups.esearch("gene", "NOSUCHGENEXYZ123[sym]") == []
    print("lookups ok")


if __name__ == "__main__":
    test_trace()
    test_typed_atoms()
    test_policies()
    test_sandbox()
    if "--online" in sys.argv:
        test_lookups()
    print("shared-core self-test passed")
