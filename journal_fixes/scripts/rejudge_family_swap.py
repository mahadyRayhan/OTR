"""
rejudge_family_swap.py — re-score the STORED answers (originally judged by GPT-5.4) with a
different judge model, using the exact rubric of scripts/eval/eval_mode.py, and test whether the
paper's conclusions survive the judge change.

Reads answers from
  results/eval_react_cot_context.json, results/eval_react_cot_no_context.json,
  results/eval_adaptive.json                     (30 benchmark queries, GPT-5.4 judged)
  results/full_comparison/comparison_results.json (50 queries: adaptive / gemini / gpt-4o)
and writes journal_fixes/results/rejudge_<model>.json with:
  * per-system means under both judges,
  * per-item agreement (Spearman) between the judges,
  * whether each pairwise conclusion (sign of the paired difference) flips.

The answer is passed in full (the original evaluation truncated it to 5,000 characters);
--truncate 5000 reproduces the original setting.

Usage
  python journal_fixes/scripts/rejudge_family_swap.py --judge-model gemini-3.1-flash-lite-preview
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from orchestrator import usage_meter  # noqa: E402

usage_meter.install()
from scripts.eval.eval_mode import JUDGE_PROMPT  # noqa: E402

OUT = ROOT / "journal_fixes" / "results"
DIMS = ["grounding", "specificity", "correctness", "completeness", "code_quality", "coherence"]
W = dict(zip(DIMS, [0.35, 0.35, 0.15, 0.10, 0.03, 0.02]))


def load_items() -> list[dict]:
    items = []
    for f, system in [("eval_react_cot_context.json", "react_context"), ("eval_react_cot_no_context.json", "react_no_context"),
                      ("eval_adaptive.json", "sciweaver")]:
        for r in json.loads((ROOT / "results" / f).read_text())["runs"]:
            j = r.get("judge") or {}
            if r.get("final_answer") and all(k in j for k in DIMS):
                items.append({"set": "bench30", "system": system, "query": r["query"], "answer": r["final_answer"],
                              "orig": {k: j[k] for k in DIMS}})
    for r in json.loads((ROOT / "results/full_comparison/comparison_results.json").read_text()):
        for system in ["adaptive", "gemini", "gpt"]:
            s = r[system]["scores"]
            items.append({"set": "full50", "system": {"adaptive": "sciweaver", "gpt": "gpt-4o"}.get(system, system),
                          "query": r["query"], "answer": r[system]["answer"],
                          "orig": {k: s[{"code_quality": "Code_Quality"}.get(k, k.capitalize())] for k in DIMS}})
    return items


async def judge(model_name: str, query: str, answer: str) -> dict:
    from journal_fixes.scripts.llm import generate
    text = await generate(model_name, JUDGE_PROMPT.format(query=query, response=answer))
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


async def main(model_name: str, truncate: int | None, concurrency: int) -> None:
    import numpy as np
    from scipy import stats

    items = load_items()
    raw_path = OUT / f"rejudge_{model_name}.jsonl"
    done = {}
    if raw_path.exists():
        for l in raw_path.open():
            x = json.loads(l)
            done[(x["set"], x["system"], x["query"])] = x
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        key = (it["set"], it["system"], it["query"])
        if key in done:
            return
        async with sem:
            try:
                s = await judge(model_name, it["query"], it["answer"][:truncate] if truncate else it["answer"])
                rec = {**{k: it[k] for k in ("set", "system", "query", "orig")}, "new": {k: float(s[k]) for k in DIMS}}
                done[key] = rec
                with raw_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as exc:
                if type(exc).__name__ == "CreditsExhausted":
                    raise
                print("judge error:", repr(exc)[:150])

    OUT.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*[one(it) for it in items])

    gs = lambda d: sum(W[k] * d[k] for k in DIMS)
    # Usage covers this invocation only (earlier, resumed invocations are not included).
    report = {"judge_model": model_name, "truncate": truncate, "llm_usage_this_invocation": usage_meter.snapshot(), "sets": {}}
    for st in ["bench30", "full50"]:
        recs = [r for r in done.values() if r["set"] == st]
        systems = sorted({r["system"] for r in recs})
        by = {s: {r["query"]: r for r in recs if r["system"] == s} for s in systems}
        means = {s: {"orig_gs": round(float(np.mean([gs(r["orig"]) for r in by[s].values()])), 3),
                     "new_gs": round(float(np.mean([gs(r["new"]) for r in by[s].values()])), 3),
                     "new_correctness": round(float(np.mean([r["new"]["correctness"] for r in by[s].values()])), 3),
                     "n": len(by[s])} for s in systems}
        rho = stats.spearmanr([gs(r["orig"]) for r in recs], [gs(r["new"]) for r in recs])[0] if len(recs) > 3 else None
        pairs = {}
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                qs = sorted(set(by[a]) & set(by[b]))
                if len(qs) < 5:
                    continue
                d_orig = np.array([gs(by[a][q]["orig"]) - gs(by[b][q]["orig"]) for q in qs])
                d_new = np.array([gs(by[a][q]["new"]) - gs(by[b][q]["new"]) for q in qs])
                pairs[f"{a} - {b}"] = {"n": len(qs), "orig_diff": round(float(d_orig.mean()), 3),
                                       "new_diff": round(float(d_new.mean()), 3),
                                       "new_wilcoxon_p": round(float(stats.wilcoxon(d_new).pvalue), 4) if np.any(d_new) else 1.0,
                                       "conclusion_flips": bool(np.sign(d_orig.mean()) != np.sign(d_new.mean()))}
        report["sets"][st] = {"system_means": means, "item_spearman_orig_vs_new": None if rho is None else round(float(rho), 3),
                              "pairwise": pairs}
    (OUT / f"rejudge_{model_name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    ap.add_argument("--truncate", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        its = load_items()
        import collections
        print(len(its), collections.Counter((i["set"], i["system"]) for i in its))
    else:
        asyncio.run(main(a.judge_model, a.truncate, a.concurrency))
