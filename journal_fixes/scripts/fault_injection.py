"""
fault_injection.py — validate the PoT judge's fitness score and failure labels against
faults with KNOWN ground truth (no human labels needed).

For real node outputs from logged runs, builds five versions:
  clean               the original output (control)
  coverage            only the first ~25% of the output is kept
  specificity         code blocks removed and concrete entities (numbers, flags,
                      versioned names, CamelCase tools) replaced by generic words
  relevance           the output of a node from a DIFFERENT query
  knowledge_boundary  an explicit "no information available" reply
and scores each with the per-node judge (orchestrator.adaptive_got_dag._llm_judge).

Reports: detection rate (fitness < tau) per fault, mean fitness drop vs clean,
label accuracy per fault, and the full confusion matrix.

Output: journal_fixes/results/fault_injection.json (+ raw .jsonl)

Usage
  python journal_fixes/scripts/fault_injection.py --n 60
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

OUT = ROOT / "journal_fixes" / "results"
KB_REPLY = "I could not find information about this in the available sources. No relevant data is available for this request."


def load_nodes() -> list[dict]:
    nodes = []
    for f in glob.glob(str(ROOT / "raw_telemetry_best" / "*.json")):
        d = json.loads(Path(f).read_text())
        et = d.get("engine_telemetry", {})
        for n in et.get("got_dag_nodes", []):
            out = n.get("output") or ""
            if len(out.split()) >= 150 and n.get("query"):
                nodes.append({"run": d["run_id"], "user_query": et.get("user_query", ""), "query": n["query"], "output": out})
    return nodes


def despecify(text: str) -> str:
    text = re.sub(r"```.*?```", "(code omitted)", text, flags=re.S)
    text = re.sub(r"(?<![\w-])--?[A-Za-z][\w-]*", "an option", text)
    text = re.sub(r"(?<![\w./])\d+(\.\d+)?\s?(GB|GiB|TB|MB|bp|kb|x|%)?(?![\w/])", "some value", text)
    text = re.sub(r"\b[A-Z][a-z]*[A-Z0-9][\w.-]*\b", "a tool", text)
    return text


def variants(node: dict, other: dict) -> dict[str, str]:
    words = node["output"].split()
    return {
        "clean": node["output"],
        "coverage": " ".join(words[: max(40, len(words) // 4)]),
        "specificity": despecify(node["output"]),
        "relevance": other["output"],
        "knowledge_boundary": KB_REPLY,
    }


async def main(n: int, tau: float) -> None:
    import gemini_client as genai
    import config
    from orchestrator.adaptive_got_dag import _llm_judge

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(config.Config.GENERATION_MODEL)
    rng = random.Random(0)
    nodes = load_nodes()
    rng.shuffle(nodes)
    picked = nodes[:n]
    sem = asyncio.Semaphore(6)
    rows = []

    async def judge(node, fault, text):
        async with sem:
            s, why, mode = await _llm_judge(node["query"], text, model)
            rows.append({"run": node["run"], "query": node["query"][:200], "fault": fault,
                         "score": s, "label": mode.value, "why": why})

    tasks = []
    for node in picked:
        other = rng.choice([x for x in nodes if x["user_query"] != node["user_query"]])
        for fault, text in variants(node, other).items():
            tasks.append(judge(node, fault, text))
    await asyncio.gather(*tasks)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "fault_injection_raw.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by = collections.defaultdict(list)
    for r in rows:
        by[r["fault"]].append(r)
    clean_mean = sum(r["score"] for r in by["clean"]) / max(len(by["clean"]), 1)
    report = {"n_nodes": len(picked), "tau": tau, "per_fault": {}, "confusion": {}}
    for fault, rs in by.items():
        report["per_fault"][fault] = {
            "n": len(rs),
            "mean_fitness": round(sum(r["score"] for r in rs) / len(rs), 3),
            "drop_vs_clean": round(clean_mean - sum(r["score"] for r in rs) / len(rs), 3),
            "detected_below_tau": round(sum(r["score"] < tau for r in rs) / len(rs), 3),
            "label_accuracy": None if fault == "clean" else round(sum(r["label"] == fault for r in rs) / len(rs), 3),
        }
        report["confusion"][fault] = dict(collections.Counter(r["label"] for r in rs))
    (OUT / "fault_injection.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        ns = load_nodes()
        v = variants(ns[0], ns[-1])
        print(f"{len(ns)} eligible nodes")
        for k, t in v.items():
            print(f"--- {k}: {t[:300]!r}")
    else:
        asyncio.run(main(a.n, a.tau))
