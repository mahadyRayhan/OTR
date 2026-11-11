"""
run_pilot.py — run systems on a domain's test sets, interleaved, with a running score and early stop.

Draws --per-set items from the chosen split of every set, interleaves them so the running score covers
every part from the start, and asks each item to every system in --systems:

  sciweaver         the full engine (adaptive) on the benchmark-aligned indexes (SCIWEAVER_INDEX_DIR)
  react             the same planner and agents run as a linear chain with raw upstream context
                    (engine mode react_cot_context: no GoT, judge or retries)
  gemini            one call to the generation model, same prompt, no retrieval
  strong            one call to Config.STRONG_BASELINE_MODEL
  self_consistency  5 samples of the generation model (temperature 0.7), majority vote on the parsed answer

All answers are scored with scoring.py (no LLM judge); code_biocoder answers by biocoder_eval.py, which
runs them against BioCoder's tests in the Docker sandbox. Engine runs write a full trace
(orchestrator/trace.py) whose path is stored with the answer. After every item the running accuracy of
each system is printed (marks: ok, x, abs = abstained, fmt = unparsed), and the run stops early when
something looks broken rather than unlucky (comparing sciweaver with gemini when both run):

  - the engine crashed on 2 items
  - after 6 items, more than 30% of SciWeaver answers could not be parsed (format problem)
  - after 8 items, SciWeaver trails plain Gemini by 0.25 or more and its running accuracy fell over the last 4

Ablations: --set KEY=VALUE overrides a config.Config setting for the whole run (repeatable), e.g.
--set CONTEXT_POLICY='"raw"' --set ATTACH_REQUEST=False. Values are Python literals.

Output (outside OneDrive): ~/otr_local/runs/pilot_<domain><tag>/{answers.jsonl, trend.jsonl, summary.json}
Usage: python journal_fixes/scripts/run_pilot.py --domain bio [--per-set 4] [--sets a,b] [--split dev]
       [--systems sciweaver,gemini] [--set KEY=VALUE ...] [--tag _v1] [--no-stop]
       [--ids-from ~/otr_local/runs/pilot_bio_v5]   (repeat exactly that run's items of the given sets)
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path.home() / "otr_local"
SETS = ["rag_litqa2", "rag_bioasq", "scholar_claimpaper", "scholar_toolpaper", "code_seqqa", "cloud_nfcore", "full_system", "code_biocoder", "lookup_geneturing"]
SYSTEMS = ["sciweaver", "react", "gemini", "strong", "self_consistency"]
ENGINE_MODES = {"sciweaver": "adaptive", "react": "react_cot_context"}
SC_SAMPLES = 5
PRICES = {"flash-lite": (0.25, 1.50), "3.8-flash": (0.50, 3.00), "3-flash": (0.50, 3.00), "3.5-flash": (0.50, 3.00), "pro": (2.00, 12.00)}


def pick(domain: str, per_set: int, sets: list[str], split: str, ids_from: Path | None = None) -> list[dict]:
    """Interleaved items of the split. With ids_from, exactly the items of that earlier run (for repeats)."""
    if ids_from:
        want = [json.loads(l)["id"] for l in open(ids_from / "answers.jsonl")]
        pool = {}
        for name in sets:
            for l in open(ROOT / "journal_fixes/data/testsets" / domain / f"{name}.jsonl"):
                x = json.loads(l)
                pool[x["id"]] = dict(x, set=name)
        return [pool[i] for i in want if i in pool]
    rng = random.Random(0)
    groups = []
    for name in sets:
        rows = [json.loads(l) for l in open(ROOT / "journal_fixes/data/testsets" / domain / f"{name}.jsonl")]
        rows = [x for x in rows if x.get("split") == split]
        groups.append([dict(x, set=name) for x in rng.sample(rows, min(per_set, len(rows)))])
    return [g[i] for i in range(per_set) for g in groups if i < len(g)]  # round-robin over the sets


def cost(snap: dict) -> float:
    total = 0.0
    for model, u in snap["by_model"].items():
        pin, pout = next((p for k, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0])) if k in model),
                         PRICES["pro"])
        total += (u["input_tokens"] * pin + u["output_tokens"] * pout) / 1e6
    return total


def stop_reason(rows: list[dict]) -> str | None:
    n = len(rows)
    if "sciweaver" not in rows[-1]["systems"]:
        return None
    sw = [r["systems"]["sciweaver"] for r in rows]
    errors = sum(s.get("error") is not None for s in sw)
    if errors >= 2:
        return f"engine crashed on {errors} items"
    unparsed = sum(s["unparsed"] for s in sw)  # abstaining is not a format failure
    if n >= 6 and unparsed / n > 0.3:
        return f"{unparsed}/{n} SciWeaver answers could not be parsed (format problem)"
    if n >= 8 and "gemini" in rows[-1]["running"]:
        s, g = rows[-1]["running"]["sciweaver"], rows[-1]["running"]["gemini"]
        if g - s >= 0.25 and s < rows[-5]["running"]["sciweaver"]:
            return f"SciWeaver {s:.2f} trails plain Gemini {g:.2f} and is falling"
    return None


async def main(domain: str, per_set: int, no_stop: bool, sets: list[str], tag: str, ids_from: Path | None,
               systems: list[str], split: str, overrides: dict) -> None:
    os.environ["SCIWEAVER_INDEX_DIR"] = str(LOCAL / "indexes" / domain)
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    out = LOCAL / "runs" / f"pilot_{domain}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # the engine appends experiment_telemetry.json to the working directory; keep it off OneDrive

    from orchestrator import usage_meter
    usage_meter.install()
    import config
    for k, v in overrides.items():
        if not hasattr(config.Config, k):
            raise SystemExit(f"--set {k}: no such config.Config setting")
        setattr(config.Config, k, v)
    from journal_fixes.scripts.biocoder_eval import check
    from journal_fixes.scripts.llm import generate
    from journal_fixes.scripts.scoring import paired_credit, paired_difference, prompt, score, summarize

    items = pick(domain, per_set, sets, split, ids_from)
    done = {json.loads(l)["id"] for l in open(out / "answers.jsonl")} if (out / "answers.jsonl").exists() else set()
    rows = [json.loads(l) for l in open(out / "trend.jsonl")] if (out / "trend.jsonl").exists() else []
    spent = Counter(rows[-1]["usd_so_far"]) if rows else Counter()
    eng = None
    if any(s in ENGINE_MODES for s in systems):
        from orchestrator.engine import OrchestratorEngine
        eng = OrchestratorEngine()
    print(f"pilot: {len(items)} items, {len(done)} already done; systems {systems}; overrides {overrides}; "
          f"index {os.environ['SCIWEAVER_INDEX_DIR']}")

    async def ask(system: str, x: dict, p: str) -> dict:
        grade = check if x["answer_format"] == "code" else score
        usage_meter.reset()
        t0, err, trace_file = time.time(), None, None
        ans = ""
        if system in ENGINE_MODES:
            for attempt in range(2):  # one retry, for dropped connections and similar transient errors
                try:  # the engine resets the meter itself at the start of the run
                    res = await eng.run(p, ablation_mode=ENGINE_MODES[system], start_time=t0)
                    ans, err, trace_file = res["final_answer"] or "", None, res.get("trace_file")
                    break
                except Exception:
                    err = traceback.format_exc(limit=3)
                    print(f"engine error on {x['id']} ({system}, attempt {attempt + 1}): {err.splitlines()[-1]}", flush=True)
        else:
            try:
                if system == "self_consistency":
                    samples = await asyncio.gather(*[generate(config.Config.GENERATION_MODEL, p, 0.7)
                                                     for _ in range(SC_SAMPLES)])
                    parsed = [json.dumps(grade(x, s)["parsed"], sort_keys=True, default=str) for s in samples]
                    votes = Counter(v for v in parsed if v not in ("null", "[]", '""'))
                    ans = samples[parsed.index(votes.most_common(1)[0][0])] if votes else samples[0]
                else:
                    model = config.Config.STRONG_BASELINE_MODEL if system == "strong" else config.Config.GENERATION_MODEL
                    ans = await generate(model, p, 0.0)
            except Exception as e:
                err = repr(e)
        use = usage_meter.snapshot()
        sc = grade(x, ans)
        return {"answer": ans, "error": err, "seconds": round(time.time() - t0, 1), "usage": use,
                "usd": round(cost(use), 5), "trace_file": trace_file, **sc}

    for x in items:
        if x["id"] in done:
            continue
        p = prompt(x)
        res = {s: await ask(s, x, p) for s in systems}
        for s in systems:
            spent[s] += res[s]["usd"]
        rec = {"id": x["id"], "set": x["set"], "answer_format": x["answer_format"], "gold": x["gold"],
               "overrides": overrides, **res}
        with open(out / "answers.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

        row = {"id": x["id"], "set": x["set"],
               "systems": {s: {"correct": res[s]["correct"], "answered": res[s]["answered"],
                               "unparsed": not res[s]["parsed"] and not res[s]["answered"],
                               "error": res[s]["error"] and res[s]["error"].splitlines()[-1]} for s in systems}}
        rows.append(row)
        n = len(rows)
        row["running"] = {s: round(sum(bool(r["systems"][s]["correct"]) for r in rows if s in r["systems"]) / n, 3)
                          for s in systems}
        row["usd_so_far"] = {s: round(v, 4) for s, v in spent.items()}
        with open(out / "trend.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        mark = lambda r: "ok " if r["correct"] else "fmt" if not r["parsed"] and not r["answered"] else "abs" if not r["answered"] else "x  "
        print(f"[{n:2d}] {x['set']:<19} " + " ".join(f"{s[:4]} {mark(res[s])}" for s in systems)
              + " | running " + " ".join(f"{s[:4]} {row['running'][s]:.2f}" for s in systems)
              + f" | ${sum(spent.values()):.3f}", flush=True)

        why = stop_reason(rows)
        if why and not no_stop:
            print(f"\nSTOPPED EARLY: {why}")
            break
    else:
        why = None

    answers = [json.loads(l) for l in open(out / "answers.jsonl")]
    per = {s: [a[s] for a in answers if s in a] for s in systems}
    by_set = {}
    for a in answers:
        b = by_set.setdefault(a["set"], {"n": 0})
        b["n"] += 1
        for s in systems:
            if s in a:
                b[s] = b.get(s, 0) + bool(a[s]["correct"])
    summary = {"domain": domain, "split": split, "systems": systems, "overrides": overrides, "n": len(answers),
               "planned": len(items), "stopped_early": why,
               "usd": {s: round(sum(r["usd"] for r in per[s]), 4) for s in systems},
               "overall": {s: summarize(per[s]) for s in systems}, "by_set": by_set}
    if "sciweaver" in systems:
        both = [a for a in answers if "sciweaver" in a]
        summary["sciweaver_minus"] = {s: paired_difference([a["sciweaver"]["correct"] for a in both if s in a],
                                                           [a[s]["correct"] for a in both if s in a])
                                      for s in systems if s != "sciweaver"}
    checklist = [a for a in answers if a["answer_format"] == "checklist"]
    if checklist:  # multi-part items: share of checks passed, per set and paired against SciWeaver
        summary["checklist"] = {
            name: {s: summarize([a[s] for a in checklist if a["set"] == name and s in a]) for s in systems}
            for name in sorted({a["set"] for a in checklist})}
        if "sciweaver" in systems:
            summary["checklist_sciweaver_minus"] = {
                name: {s: paired_credit([a["sciweaver"]["credit"] for a in checklist if a["set"] == name and s in a],
                                        [a[s]["credit"] for a in checklist if a["set"] == name and s in a])
                       for s in systems if s != "sciweaver"}
                for name in sorted({a["set"] for a in checklist})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if checklist:
        from journal_fixes.scripts.judge_packets import export
        export(out)


def parse_set(arg: str) -> tuple[str, object]:
    key, _, value = arg.partition("=")
    try:
        return key.strip(), ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key.strip(), value  # a bare word is taken as a string


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--per-set", type=int, default=4)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--systems", default="sciweaver,gemini", help=f"comma list of {SYSTEMS}")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config.Config override")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--ids-from", type=Path, help="rerun exactly the items of this earlier run folder")
    a = ap.parse_args()
    systems = a.systems.split(",")
    if unknown := [s for s in systems if s not in SYSTEMS]:
        raise SystemExit(f"unknown systems {unknown}; choose from {SYSTEMS}")
    asyncio.run(main(a.domain, a.per_set, a.no_stop, a.sets.split(","), a.tag, a.ids_from and a.ids_from.expanduser(),
                     systems, a.split, dict(parse_set(s) for s in a.set)))
