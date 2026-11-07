"""Offline checks for the paper/code alignment of the PoT judge (no API calls)."""
import asyncio, json, os, sys
sys.path.insert(0, os.getcwd())
from orchestrator import adaptive_got_dag as g

class FakeResp:
    def __init__(self, text): self.text = text

class FakeModel:
    def __init__(self, payload): self.payload, self.prompts = payload, []
    def generate_content(self, prompt):
        self.prompts.append(prompt); return FakeResp(json.dumps(self.payload))

async def main():
    m = FakeModel({"completeness": 6, "depth": 4, "accuracy": 8, "clarity": 7, "failure_mode": "specificity", "reasoning": "too generic"})
    score, why, mode = await g._llm_judge("Which aligner?", "Use an aligner.", m, downstream=["Write a STAR command for GRCh38"])
    assert "Write a STAR command for GRCh38" in m.prompts[0], "downstream spec missing from judge prompt"
    assert mode is g.FailureMode.SPECIFICITY and abs(score - (0.35*6+0.30*4+0.25*8+0.10*7)/10) < 1e-9
    m2 = FakeModel({"completeness": 2, "depth": 2, "accuracy": 5, "clarity": 6, "failure_mode": "knowledge_boundary", "reasoning": "not found"})
    _, _, mode2 = await g._llm_judge("q", "No data available.", m2)
    assert "final answer synthesis" in m2.prompts[0]
    assert mode2 is g.FailureMode.KNOWLEDGE_BOUNDARY and mode2 not in g._EXPANDABLE_MODES
    for fm in (g.FailureMode.SPECIFICITY, g.FailureMode.COVERAGE, g.FailureMode.RELEVANCE):
        assert fm in g._EXPANDABLE_MODES and fm in g._REPAIR_HINT
    assert g.FailureMode("missing_data") is g.FailureMode.MISSING_DATA  # old telemetry still parses
    import config; assert config.Config.PER_NODE_TAU == 0.70
    print("PoT alignment checks passed")

asyncio.run(main())


def test_low_confidence():
    from types import SimpleNamespace
    T = lambda q: SimpleNamespace(query=q)
    mk = lambda nid, q, score, kids=(), fm=None: g.GoTNode(node_id=nid, task=T(q), dim=list(g.Dim)[0], depth=1, result="out",
                                                            quality_score=score, children_ids=list(kids), failure_mode=fm)
    a = mk("a", "repaired", 0.5, kids=["b"]); b = mk("b", "fix", 0.8)
    c = mk("c", "stuck", 0.4, fm=g.FailureMode.KNOWLEDGE_BOUNDARY); d = mk("d", "fine", 0.9)
    orch = g.AdaptiveGoTOrchestrator(quality_threshold=0.7)
    orch._dag = SimpleNamespace(_nodes={n.node_id: n for n in (a, b, c, d)})
    assert [n.node_id for n in orch.low_confidence_nodes()] == ["c"]
    print("low-confidence surfacing check passed")

test_low_confidence()
