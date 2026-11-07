"""
judge_pilot.py — score a pilot run's answers with the paper's LLM-judge rubric (Table 1 dimensions).

Uses the rubric of scripts/eval/eval_mode.py and the Global Score weights of rejudge_family_swap.py.
The judge sees the question (with its format line, as the systems saw it) and the full answer, not the
gold answer, as in the paper. Both systems of every item are judged, so the comparison is paired.
Next to the rubric it reports the exact-answer accuracy (scoring.py) of the same items, and how well
the judge's Correctness tracks it.

Output: <run>/judged_<model>.jsonl (per answer, resumable) and <run>/judged_<model>.json (summary).
Usage: python journal_fixes/scripts/judge_pilot.py ~/otr_local/runs/pilot_bio_v7 [--judge-model gemini-3-flash-preview]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from orchestrator import usage_meter  # noqa: E402

usage_meter.install()
from journal_fixes.scripts.llm import generate  # noqa: E402
from journal_fixes.scripts.scoring import prompt  # noqa: E402
from scripts.eval.eval_mode import JUDGE_PROMPT  # noqa: E402

SYSTEMS = ("sciweaver", "gemini")
# As in rejudge_family_swap.py (not imported: that module changes directory and installs the meter on import).
DIMS = ["grounding", "specificity", "correctness", "completeness", "code_quality", "coherence"]
W = dict(zip(DIMS, [0.35, 0.35, 0.15, 0.10, 0.03, 0.02]))


def items(domain: str) -> dict:
    out = {}
    for f in (ROOT / "journal_fixes/data/testsets" / domain).glob("*.jsonl"):
        for line in open(f):
            x = json.loads(line)
            out[x["id"]] = x
    return out


async def judge(model: str, query: str, answer: str) -> dict:
    text = await generate(model, JUDGE_PROMPT.format(query=query, response=answer), 0.0)
    s = json.loads(text[text.index("{"): text.rindex("}") + 1])
    return {k: float(s[k]) for k in DIMS}


async def main(run: Path, model: str, domain: str, concurrency: int) -> None:
    import numpy as np
    from scipy import stats

    pool = items(domain)
    rows = [json.loads(l) for l in open(run / "answers.jsonl")]
    raw = run / f"judged_{model}.jsonl"
    done = {(r["id"], r["system"]): r for r in map(json.loads, open(raw))} if raw.exists() else {}
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict, system: str) -> None:
        if (r["id"], system) in done or r["id"] not in pool:
            return
        async with sem:
            for attempt in range(2):
                try:
                    s = await judge(model, prompt(pool[r["id"]]), r[system]["answer"] or "(no answer)")
                    break
                except Exception as exc:
                    if type(exc).__name__ == "CreditsExhausted" or attempt:
                        print("judge error:", r["id"][:10], system, repr(exc)[:150])
                        return
        rec = {"id": r["id"], "set": r["set"], "system": system, "exact": bool(r[system]["correct"]), "scores": s}
        done[(r["id"], system)] = rec
        with raw.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    await asyncio.gather(*[one(r, s) for r in rows for s in SYSTEMS])

    gs = lambda d: sum(W[k] * d[k] for k in DIMS)
    ids = [r["id"] for r in rows if all((r["id"], s) in done for s in SYSTEMS)]
    table = {}
    for s in SYSTEMS:
        recs = [done[(i, s)] for i in ids]
        table[s] = {"GTS": round(float(np.mean([gs(x["scores"]) for x in recs])), 2),
                    **{k: round(float(np.mean([x["scores"][k] for x in recs])), 2) for k in DIMS},
                    "exact_accuracy": round(float(np.mean([x["exact"] for x in recs])), 3), "n": len(recs)}
    diff = np.array([gs(done[(i, "sciweaver")]["scores"]) - gs(done[(i, "gemini")]["scores"]) for i in ids])
    by_set = {}
    for st in sorted({done[(i, "sciweaver")]["set"] for i in ids}):
        q = [i for i in ids if done[(i, "sciweaver")]["set"] == st]
        by_set[st] = {s: {"GTS": round(float(np.mean([gs(done[(i, s)]["scores"]) for i in q])), 2),
                          "correctness": round(float(np.mean([done[(i, s)]["scores"]["correctness"] for i in q])), 2),
                          "exact": sum(done[(i, s)]["exact"] for i in q)} for s in SYSTEMS} | {"n": len(q)}
    allrec = [done[(i, s)] for i in ids for s in SYSTEMS]
    summary = {"judge_model": model, "run": str(run), "n_items": len(ids), "table": table,
               "sciweaver_minus_gemini_GTS": round(float(diff.mean()), 3),
               "wilcoxon_p": round(float(stats.wilcoxon(diff).pvalue), 4) if np.any(diff) else 1.0,
               "judge_correctness_vs_exact_spearman": round(float(stats.spearmanr(
                   [x["scores"]["correctness"] for x in allrec], [x["exact"] for x in allrec])[0]), 3),
               "by_set": by_set, "llm_usage_this_invocation": usage_meter.snapshot()}
    (run / f"judged_{model}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "llm_usage_this_invocation"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--judge-model", default="gemini-3-flash-preview")
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    asyncio.run(main(a.run.expanduser(), a.judge_model, a.domain, a.concurrency))
