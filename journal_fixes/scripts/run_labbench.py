"""
run_labbench.py — objective, judge-free evaluation on LAB-Bench (FutureHouse, CC BY-SA 4.0).

Every question has one correct option, so accuracy is exact-match and no LLM
judge is involved. Every system's LLM calls are metered (orchestrator/usage_meter)
so results can be compared at matched token cost.

Systems
  direct            one call to the generation model
  self_consistency  k samples at temperature 0.7, majority vote (equal-budget baseline)
  react_context     SciWeaver engine, linear ReAct mode with raw context passing
  react_no_context  SciWeaver engine, linear ReAct mode, no context passing
  sciweaver         SciWeaver engine, full adaptive mode (CUGCR + PoT)
  sciweaver_raw     full DAG + PoT but raw context instead of CUGCR (routing ablation)

Metrics (LAB-Bench convention): accuracy = correct / all; precision = correct / answered
(an "Insufficient information" choice counts as unanswered); coverage = answered / all.

Output: journal_fixes/results/labbench/<system>.jsonl (resumable) and summary.json

Usage
  python journal_fixes/scripts/run_labbench.py --dry-run
  python journal_fixes/scripts/run_labbench.py --per-subset 50 --systems direct,self_consistency
  python journal_fixes/scripts/run_labbench.py --per-subset 50 --systems react_context,sciweaver
  python journal_fixes/scripts/run_labbench.py --summarize
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import json
import os
import random
import re
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

DATA = ROOT / "journal_fixes" / "data" / "benchmarks"
OUT = ROOT / "journal_fixes" / "results" / "labbench"
SUBSETS = ["LitQA2", "ProtocolQA", "SeqQA", "DbQA"]
UNSURE = "Insufficient information to answer the question"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_items(per_subset: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for sub in SUBSETS:
        rows = json.loads((DATA / f"labbench_{sub}.json").read_text())
        rng.shuffle(rows)
        for r in rows[:per_subset]:
            distractors = r["distractors"]
            if isinstance(distractors, str):
                distractors = ast.literal_eval(distractors)
            options = [r["ideal"], *distractors]
            rng.shuffle(options)
            options.append(UNSURE)
            body = r["question"]
            if sub == "ProtocolQA" and r.get("protocol"):
                body = f"Protocol:\n{r['protocol']}\n\n{body}"
            prompt = (f"{body}\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
                      + "\n\nThink it through, then end your reply with one line of the form 'Answer: <letter>'.")
            items.append({"id": r["id"], "subset": sub, "prompt": prompt, "options": options,
                          "gold": LETTERS[options.index(r["ideal"])], "unsure": LETTERS[len(options) - 1]})
    return items


def parse_letter(text: str, n_options: int) -> str | None:
    valid = LETTERS[:n_options]
    m = re.findall(rf"Answer\s*[:\-]\s*\(?([{valid}])\)?\b", text or "", re.I)
    if m:
        return m[-1].upper()
    m = re.findall(rf"\b([{valid}])\)", (text or "")[-300:])
    return m[-1].upper() if m else None


async def gen(prompt: str, temperature: float = 0.0) -> str:
    import config
    from journal_fixes.scripts.llm import generate
    return await generate(config.Config.GENERATION_MODEL, prompt, temperature)


async def run_direct(item: dict) -> dict:
    text = await gen(item["prompt"])
    return {"answer": parse_letter(text, len(item["options"])), "raw": text[-1500:]}


async def run_self_consistency(item: dict, k: int = 5) -> dict:
    texts = await asyncio.gather(*[gen(item["prompt"], temperature=0.7) for _ in range(k)])
    votes = [parse_letter(t, len(item["options"])) for t in texts]
    counted = collections.Counter(v for v in votes if v)
    return {"answer": counted.most_common(1)[0][0] if counted else None, "votes": votes}


_ENGINE = None


async def run_engine(item: dict, mode: str) -> dict:
    global _ENGINE
    if _ENGINE is None:
        from orchestrator.engine import OrchestratorEngine
        _ENGINE = OrchestratorEngine()
    out = await _ENGINE.run(item["prompt"], ablation_mode=mode, start_time=time.time())
    text = out.get("final_answer", "")
    return {"answer": parse_letter(text, len(item["options"])), "raw": text[-1500:]}


SYSTEMS = {
    "direct": run_direct,
    "self_consistency": run_self_consistency,
    "react_context": lambda it: run_engine(it, "react_cot_context"),
    "react_no_context": lambda it: run_engine(it, "react_cot_no_context"),
    "sciweaver": lambda it: run_engine(it, "adaptive"),
    "sciweaver_raw": lambda it: run_engine(it, "raw"),
}
ENGINE_SYSTEMS = {"react_context", "react_no_context", "sciweaver", "sciweaver_raw"}


async def run_system(name: str, items: list[dict], concurrency: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"[{name}] {len(done)} done, {len(todo)} to run")
    # The engine keeps per-run global state (telemetry, comm tracker), so run it one item at a time.
    sem = asyncio.Semaphore(1 if name in ENGINE_SYSTEMS else concurrency)
    lock = asyncio.Lock()

    async def one(it: dict) -> None:
        async with sem:
            usage_meter.reset() if name in ENGINE_SYSTEMS else None
            before = usage_meter.snapshot()
            t0 = time.time()
            try:
                res = await SYSTEMS[name](it)
                err = None
            except Exception as exc:
                if type(exc).__name__ == "CreditsExhausted":
                    raise
                res, err = {"answer": None}, repr(exc)[:300]
            after = usage_meter.snapshot()
            rec = {"id": it["id"], "subset": it["subset"], "gold": it["gold"], "unsure": it["unsure"],
                   **res, "error": err, "latency_s": round(time.time() - t0, 2),
                   "tokens_in": after["input_tokens"] - before["input_tokens"] if name not in ENGINE_SYSTEMS else after["input_tokens"],
                   "tokens_out": after["output_tokens"] - before["output_tokens"] if name not in ENGINE_SYSTEMS else after["output_tokens"],
                   "usage_by_model": after["by_model"] if name in ENGINE_SYSTEMS else None}
            async with lock:
                with path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

    await asyncio.gather(*[one(it) for it in todo])


def summarize() -> dict:
    import numpy as np
    summary = {}
    for path in sorted(OUT.glob("*.jsonl")):
        recs = [json.loads(l) for l in path.open()]
        recs = [r for r in recs if not r.get("error")]
        by = collections.defaultdict(list)
        for r in recs:
            by[r["subset"]].append(r)
            by["ALL"].append(r)
        sys_sum = {}
        for sub, rs in by.items():
            correct = np.array([r["answer"] == r["gold"] for r in rs])
            answered = np.array([r["answer"] not in (None, r["unsure"]) for r in rs])
            boot = [correct[np.random.default_rng(i).integers(0, len(rs), len(rs))].mean() for i in range(2000)]
            sys_sum[sub] = {"n": len(rs), "accuracy": round(float(correct.mean()), 3),
                            "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                            "precision": round(float(correct.sum() / max(answered.sum(), 1)), 3),
                            "coverage": round(float(answered.mean()), 3),
                            "mean_tokens": round(float(np.mean([r["tokens_in"] + r["tokens_out"] for r in rs])), 1),
                            "mean_latency_s": round(float(np.mean([r["latency_s"] for r in rs])), 2)}
        summary[path.stem] = sys_sum
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for s, v in summary.items():
        a = v.get("ALL", {})
        print(f"{s:18s} acc={a.get('accuracy')} {a.get('ci95')} prec={a.get('precision')} cov={a.get('coverage')} tokens/q={a.get('mean_tokens')}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subset", type=int, default=50)
    ap.add_argument("--systems", default="direct,self_consistency")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    items = load_items(args.per_subset)
    if args.dry_run:
        print(f"{len(items)} items; example:\n{items[0]['prompt']}\nGold: {items[0]['gold']}")
        return
    for name in args.systems.split(","):
        asyncio.run(run_system(name.strip(), items, args.concurrency))
    summarize()


if __name__ == "__main__":
    main()
