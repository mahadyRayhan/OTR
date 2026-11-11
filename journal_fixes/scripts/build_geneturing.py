"""
build_geneturing.py — the lookup_geneturing test set: GeneTuring (9 tasks) + GeneHop's Disease gene location.

Source: the GeneGPT repository (Jin et al., Bioinformatics 2024; github.com/ncbi/GeneGPT, data/
geneturing.json and data/genehop.json, NCBI, public domain), downloaded to ~/otr_local/benchmarks/genegpt.
These are the 10 tasks GeneGPT's evaluate.py scores automatically (the other two GeneHop tasks need
manual grading and are left out). 50 questions per task, 500 in all. Answering them needs database
lookups (NCBI E-utilities, BLAST), which is what orchestrator/lookups.py provides.

Each item: answer_format "geneturing", meta.task = the task name; scored by scoring.py with GeneGPT's
rules (exact answer; partial credit for gene-disease association, disease gene location and human
genome alignment). Dev/test split per task, 20% dev, seeded, as in build_testsets.py.

Usage: python journal_fixes/scripts/build_geneturing.py [--domain bio]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from journal_fixes.scripts.build_testsets import item, split  # noqa: E402

SRC = Path.home() / "otr_local" / "benchmarks" / "genegpt"
GENEHOP_AUTO = "Disease gene location"  # the GeneHop task GeneGPT's evaluate.py scores automatically


def build() -> list[dict]:
    tasks = json.loads((SRC / "geneturing.json").read_text())
    tasks[GENEHOP_AUTO] = json.loads((SRC / "genehop.json").read_text())[GENEHOP_AUTO]
    rng, out = random.Random(0), []
    for task, qa in tasks.items():
        rows = []
        for q, a in qa.items():
            question = re.sub(r"\s*Let's decompose the question.*$", "", q, flags=re.S).strip()  # GeneGPT's own prompt text
            gold = a if isinstance(a, list) else str(a)
            iid = "gt_" + hashlib.sha256(f"{task}|{q}".encode()).hexdigest()[:16]
            rows.append(item(iid, "geneturing" if task != GENEHOP_AUTO else "genehop", "lookup", question,
                             "geneturing", gold, "GeneGPT evaluate.py rules (scoring.py)", task=task))
        out += split(rows, rng)
    return out


def main(domain: str) -> None:
    rows = build()
    f = ROOT / "journal_fixes/data/testsets" / domain / "lookup_geneturing.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in rows))
    n_dev = sum(x["split"] == "dev" for x in rows)
    print(f"{f.name}: {len(rows)} items ({n_dev} dev, {len(rows) - n_dev} test), "
          f"{len({x['meta']['task'] for x in rows})} tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    main(ap.parse_args().domain)
