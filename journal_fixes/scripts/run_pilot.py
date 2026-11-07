"""
run_pilot.py — a small, interleaved pilot on one domain's test sets, with a running score and early stop.

Draws --per-set items from the dev split of every set, interleaves them so the running score covers every
part from the start, and asks each item to

  sciweaver   the full engine (adaptive mode) on the benchmark-aligned indexes (SCIWEAVER_INDEX_DIR)
  gemini      one call to the generation model, same prompt, no retrieval

Both answers are scored with scoring.py (no LLM judge); code_biocoder answers by biocoder_eval.py, which
runs them against BioCoder's tests in a Docker container (needs Docker; SciWeaver itself runs no code). After every item it prints the running accuracy
of both systems and the format failures (abstentions shown as 'abs'), and stops early when something looks broken rather than unlucky:

  - the engine crashed on 2 items
  - after 6 items, more than 30% of SciWeaver answers could not be parsed (format problem)
  - after 8 items, SciWeaver trails plain Gemini by 0.25 or more and its running accuracy fell over the last 4

Output (outside OneDrive): ~/otr_local/runs/pilot_<domain><tag>/{answers.jsonl, trend.jsonl, summary.json}
Usage: python journal_fixes/scripts/run_pilot.py --domain bio [--per-set 4] [--sets a,b] [--tag _v1] [--no-stop]
       [--ids-from ~/otr_local/runs/pilot_bio_v5]   (repeat exactly that run's items of the given sets)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path.home() / "otr_local"
SETS = ["rag_litqa2", "rag_bioasq", "scholar_claimpaper", "scholar_toolpaper", "code_seqqa", "cloud_nfcore", "full_system", "code_biocoder"]
PRICES = {"flash-lite": (0.25, 1.50), "3-flash": (0.50, 3.00), "pro": (2.00, 12.00)}


def pick(domain: str, per_set: int, sets: list[str], ids_from: Path | None = None) -> list[dict]:
    """Interleaved dev items. With ids_from, exactly the items of that earlier run (for repeats)."""
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
        dev = [json.loads(l) for l in open(ROOT / "journal_fixes/data/testsets" / domain / f"{name}.jsonl")]
        dev = [x for x in dev if x.get("split") == "dev"]
        groups.append([dict(x, set=name) for x in rng.sample(dev, min(per_set, len(dev)))])
    return [g[i] for i in range(per_set) for g in groups if i < len(g)]  # round-robin over the sets


def cost(snap: dict) -> float:
    total = 0.0
    for model, u in snap["by_model"].items():
        pin, pout = next((p for k, p in PRICES.items() if k in model), PRICES["pro"])
        total += (u["input_tokens"] * pin + u["output_tokens"] * pout) / 1e6
    return total


def stop_reason(rows: list[dict]) -> str | None:
    n = len(rows)
    errors = sum(r["sciweaver"].get("error") is not None for r in rows)
    if errors >= 2:
        return f"engine crashed on {errors} items"
    unparsed = sum(r["sciweaver"].get("unparsed", not r["sciweaver"]["answered"]) for r in rows)  # abstaining is not a format failure
    if n >= 6 and unparsed / n > 0.3:
        return f"{unparsed}/{n} SciWeaver answers could not be parsed (format problem)"
    if n >= 8:
        sw, gm = rows[-1]["running"]["sciweaver"], rows[-1]["running"]["gemini"]
        if gm - sw >= 0.25 and sw < rows[-5]["running"]["sciweaver"]:
            return f"SciWeaver {sw:.2f} trails plain Gemini {gm:.2f} and is falling"
    return None


async def main(domain: str, per_set: int, no_stop: bool, sets: list[str], tag: str, ids_from: Path | None) -> None:
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
    from journal_fixes.scripts.llm import generate
    from journal_fixes.scripts.biocoder_eval import check
    from journal_fixes.scripts.scoring import prompt, score
    from orchestrator.engine import OrchestratorEngine

    items = pick(domain, per_set, sets, ids_from)
    done = {json.loads(l)["id"] for l in open(out / "answers.jsonl")} if (out / "answers.jsonl").exists() else set()
    rows = [json.loads(l) for l in open(out / "trend.jsonl")] if (out / "trend.jsonl").exists() else []
    spent = rows[-1]["usd_so_far"] if rows else 0.0
    eng = OrchestratorEngine()
    print(f"pilot: {len(items)} items, {len(done)} already done; index {os.environ['SCIWEAVER_INDEX_DIR']}")

    for x in items:
        if x["id"] in done:
            continue
        p = prompt(x)
        usage_meter.reset()
        t0 = time.time()
        try:
            g_ans, g_err = await generate(config.Config.GENERATION_MODEL, p, 0.0), None
        except Exception as e:
            g_ans, g_err = "", repr(e)
        g_use = usage_meter.snapshot()
        g_sec = time.time() - t0

        t0 = time.time()
        for attempt in range(2):  # one retry, for dropped connections and similar transient errors
            try:  # the engine resets the meter itself at the start of the run
                res = await eng.run(p, ablation_mode="adaptive", start_time=t0)
                s_ans, s_err, s_use = res["final_answer"] or "", None, res.get("llm_usage") or usage_meter.snapshot()
                break
            except Exception:
                s_ans, s_err, s_use = "", traceback.format_exc(limit=3), usage_meter.snapshot()
                print(f"engine error on {x['id']} (attempt {attempt + 1}): {s_err.splitlines()[-1]}", flush=True)
        s_sec = time.time() - t0

        grade = check if x["answer_format"] == "code" else score
        sc_s, sc_g = grade(x, s_ans), grade(x, g_ans)
        spent += cost(g_use) + cost(s_use)
        rec = {"id": x["id"], "set": x["set"], "answer_format": x["answer_format"], "gold": x["gold"],
               "sciweaver": {"answer": s_ans, "error": s_err, "seconds": round(s_sec, 1), "usage": s_use, **sc_s},
               "gemini": {"answer": g_ans, "error": g_err, "seconds": round(g_sec, 1), "usage": g_use, **sc_g}}
        with open(out / "answers.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        row = {"id": x["id"], "set": x["set"],
               "sciweaver": {k: sc_s[k] for k in ("correct", "answered", "parts")} | {"error": s_err and s_err.splitlines()[-1],
                                                                               "unparsed": not sc_s["parsed"]},
               "gemini": {k: sc_g[k] for k in ("correct", "answered")}}
        rows.append(row)
        n = len(rows)
        row["running"] = {s: round(sum(bool(r[s]["correct"]) for r in rows) / n, 3) for s in ("sciweaver", "gemini")}
        row["usd_so_far"] = round(spent, 4)
        with open(out / "trend.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        mark = lambda s: "ok " if s["correct"] else "fmt" if not s["parsed"] else "abs" if not s["answered"] else "x  "
        print(f"[{n:2d}] {x['set']:<19} SW {mark(sc_s)} GM {mark(sc_g)} | running SW {row['running']['sciweaver']:.2f} "
              f"GM {row['running']['gemini']:.2f} | {s_sec:5.0f}s | ${spent:.3f}" + (f" | ERROR {row['sciweaver']['error']}" if s_err else ""),
              flush=True)

        why = stop_reason(rows)
        if why and not no_stop:
            print(f"\nSTOPPED EARLY: {why}")
            break
    else:
        why = None

    by_set = {}
    for r in rows:
        b = by_set.setdefault(r["set"], {"n": 0, "sciweaver": 0, "gemini": 0, "sciweaver_unparsed": 0})
        b["n"] += 1
        b["sciweaver"] += bool(r["sciweaver"]["correct"])
        b["gemini"] += bool(r["gemini"]["correct"])
        b["sciweaver_unparsed"] += r["sciweaver"].get("unparsed", not r["sciweaver"]["answered"])
    summary = {"domain": domain, "n": len(rows), "planned": len(items), "stopped_early": why,
               "running_final": rows[-1]["running"] if rows else None, "usd": round(spent, 4), "by_set": by_set}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--per-set", type=int, default=4)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--ids-from", type=Path, help="rerun exactly the items of this earlier run folder")
    a = ap.parse_args()
    asyncio.run(main(a.domain, a.per_set, a.no_stop, a.sets.split(","), a.tag, a.ids_from and a.ids_from.expanduser()))
