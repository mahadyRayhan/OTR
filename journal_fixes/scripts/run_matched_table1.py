"""
run_matched_table1.py — the corrected Table 1: every system answers the SAME 50 queries
(none excluded), every LLM call is metered, and every answer is scored by two judges
from different model families with the same rubric and no truncation.

Systems
  sciweaver       full engine (adaptive: DAG + CUGCR + PoT)
  sciweaver_raw   full engine with raw context instead of CUGCR (routing ablation)
  react_context   engine, linear ReAct with raw context passing
  direct_gemini   one call to the generation model
  direct_strong   one call to Config.STRONG_BASELINE_MODEL (a stronger Gemini; not a judge)
  self_refine     generation model: draft -> critique -> revise, twice (equal-budget baseline)

Judges: Config.EVAL_JUDGES (two Gemini models, neither the system model); scripts/eval/eval_mode.JUDGE_PROMPT.

Output: journal_fixes/results/table1_matched/{answers_<system>.jsonl, judged_<judge>.jsonl, table1.json}

Usage
  python journal_fixes/scripts/run_matched_table1.py answer --systems direct_gemini,direct_strong,self_refine
  python journal_fixes/scripts/run_matched_table1.py answer --systems react_context,sciweaver,sciweaver_raw
  python journal_fixes/scripts/run_matched_table1.py judge
  python journal_fixes/scripts/run_matched_table1.py table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from orchestrator import usage_meter  # noqa: E402

usage_meter.install()
import config  # noqa: E402

OUT = ROOT / "journal_fixes" / "results" / "table1_matched"
QUERIES = ROOT / "journal_fixes" / "data" / "benchmark50.json"
DIMS = ["grounding", "specificity", "correctness", "completeness", "code_quality", "coherence"]
W = dict(zip(DIMS, [0.35, 0.35, 0.15, 0.10, 0.03, 0.02]))
ENGINE_SYSTEMS = {"sciweaver": "adaptive", "sciweaver_raw": "raw", "react_context": "react_cot_context"}


def queries() -> list[dict]:
    if not QUERIES.exists():
        d = json.loads((ROOT / "results/full_comparison/comparison_results.json").read_text())
        QUERIES.write_text(json.dumps([{"id": r["query_id"], "query": r["query"]} for r in d], indent=1))
    return json.loads(QUERIES.read_text())


async def gemini(prompt: str, temperature: float = 0.0, model: str | None = None) -> str:
    from journal_fixes.scripts.llm import generate
    return await generate(model or config.Config.GENERATION_MODEL, prompt, temperature)


async def self_refine(q: str) -> str:
    ans = await gemini(q)
    for _ in range(2):
        crit = await gemini(f"Question:\n{q}\n\nDraft answer:\n{ans}\n\nList the concrete errors, missing parts and vague statements in the draft. Be specific.")
        ans = await gemini(f"Question:\n{q}\n\nDraft answer:\n{ans}\n\nCritique:\n{crit}\n\nWrite the improved final answer.")
    return ans


_ENGINE = None


async def engine(q: str, mode: str) -> str:
    global _ENGINE
    if _ENGINE is None:
        from orchestrator.engine import OrchestratorEngine
        _ENGINE = OrchestratorEngine()
    return (await _ENGINE.run(q, ablation_mode=mode, start_time=time.time()))["final_answer"]


async def answer(systems: list[str], concurrency: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qs = queries()
    for s in systems:
        path = OUT / f"answers_{s}.jsonl"
        done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
        sem = asyncio.Semaphore(1 if s in ENGINE_SYSTEMS else concurrency)
        lock = asyncio.Lock()

        async def one(item):
            async with sem:
                if s in ENGINE_SYSTEMS:
                    usage_meter.reset()
                before = usage_meter.snapshot()
                t0 = time.time()
                try:
                    if s in ENGINE_SYSTEMS:
                        text = await engine(item["query"], ENGINE_SYSTEMS[s])
                    elif s == "direct_gemini":
                        text = await gemini(item["query"])
                    elif s == "direct_strong":
                        text = await gemini(item["query"], model=config.Config.STRONG_BASELINE_MODEL)
                    else:
                        text = await self_refine(item["query"])
                    err = None
                except Exception as exc:
                    if type(exc).__name__ == "CreditsExhausted":
                        raise
                    text, err = "", repr(exc)[:300]
                after = usage_meter.snapshot()
                # The in-loop judge is part of the system, so its tokens count toward cost.
                rec = {"id": item["id"], "query": item["query"], "answer": text, "error": err,
                       "latency_s": round(time.time() - t0, 2),
                       "tokens": (after["input_tokens"] + after["output_tokens"]) - (0 if s in ENGINE_SYSTEMS else before["input_tokens"] + before["output_tokens"]),
                       "calls": after["calls"] - (0 if s in ENGINE_SYSTEMS else before["calls"])}
                async with lock:
                    with path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")

        todo = [q for q in qs if q["id"] not in done]
        print(f"[{s}] {len(todo)} to run")
        await asyncio.gather(*[one(q) for q in todo])


async def judge_all(concurrency: int) -> None:
    from scripts.eval.eval_mode import JUDGE_PROMPT
    for jmodel in config.Config.EVAL_JUDGES:
        jname = jmodel
        jfn = lambda p, m=jmodel: gemini(p, model=m)
        path = OUT / f"judged_{jname}.jsonl"
        done = {(x["system"], x["id"]) for x in map(json.loads, path.open())} if path.exists() else set()
        sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()

        async def one(system, rec):
            async with sem:
                try:
                    text = await jfn(JUDGE_PROMPT.format(query=rec["query"], response=rec["answer"]))
                    s = json.loads(text[text.index("{"): text.rindex("}") + 1])
                    row = {"system": system, "id": rec["id"], **{k: float(s[k]) for k in DIMS}}
                    async with lock:
                        with path.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                except Exception as exc:
                    if type(exc).__name__ == "CreditsExhausted":
                        raise
                    print(jname, "judge error", repr(exc)[:120])

        tasks = []
        for ap in OUT.glob("answers_*.jsonl"):
            system = ap.stem.replace("answers_", "")
            for rec in map(json.loads, ap.open()):
                if rec["answer"] and (system, rec["id"]) not in done:
                    tasks.append(one(system, rec))
        print(f"[judge {jname}] {len(tasks)} to score")
        await asyncio.gather(*tasks)


def table() -> None:
    import numpy as np
    from scipy import stats
    out = {}
    cost = {}
    for ap in OUT.glob("answers_*.jsonl"):
        recs = [json.loads(l) for l in ap.open() if json.loads(l)["answer"]]
        cost[ap.stem.replace("answers_", "")] = {"mean_tokens": round(float(np.mean([r["tokens"] for r in recs])), 0),
                                                "mean_calls": round(float(np.mean([r["calls"] for r in recs])), 1),
                                                "mean_latency_s": round(float(np.mean([r["latency_s"] for r in recs])), 1)}
    for jp in OUT.glob("judged_*.jsonl"):
        rows = [json.loads(l) for l in jp.open()]
        by = {}
        for r in rows:
            by.setdefault(r["system"], {})[r["id"]] = r
        common = set.intersection(*[set(v) for v in by.values()]) if by else set()
        tab = {}
        for s, d in by.items():
            m = {k: round(float(np.mean([d[i][k] for i in common])), 2) for k in DIMS}
            tab[s] = {"n": len(common), "global_score": round(sum(W[k] * m[k] for k in DIMS), 2),
                      "unweighted": round(float(np.mean(list(m.values()))), 2), **m, **cost.get(s, {})}
        if "sciweaver" in by:
            for s in by:
                if s == "sciweaver":
                    continue
                ids = sorted(common)
                d = np.array([sum(W[k] * (by["sciweaver"][i][k] - by[s][i][k]) for k in DIMS) for i in ids])
                tab[s]["sciweaver_minus_this"] = round(float(d.mean()), 3)
                tab[s]["wilcoxon_p"] = round(float(stats.wilcoxon(d).pvalue), 4) if np.any(d) else 1.0
        out[jp.stem.replace("judged_", "judge_")] = tab
    (OUT / "table1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answer", "judge", "table", "dry-run"])
    ap.add_argument("--systems", default="direct_gemini,direct_strong,self_refine")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    if a.step == "answer":
        asyncio.run(answer(a.systems.split(","), a.concurrency))
    elif a.step == "judge":
        asyncio.run(judge_all(a.concurrency))
    elif a.step == "table":
        table()
    else:
        print(len(queries()), "queries; first:", queries()[0]["query"][:120])
