"""
rescore.py — rescore a run's saved answers with the current scoring.py (no LLM calls;
code answers by biocoder_eval.py, in Docker).

Usage: python journal_fixes/scripts/rescore.py ~/otr_local/runs/pilot_bio_v2 [domain, default bio]
Writes <run>/rescored.json: per set and system, summarize() of the rescored answers.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from journal_fixes.scripts.biocoder_eval import check  # noqa: E402
from journal_fixes.scripts.scoring import score, summarize  # noqa: E402


def main(run: Path, domain: str) -> None:
    items = {}
    for f in (ROOT / "journal_fixes/data/testsets" / domain).glob("*.jsonl"):  # ids repeat across folders
        for line in open(f):
            x = json.loads(line)
            items[x["id"]] = x
    rows = defaultdict(list)
    dropped = []
    for line in open(run / "answers.jsonl"):
        r = json.loads(line)
        if r["id"] not in items:  # removed from the set since the run (refine_paper_gold.py)
            dropped.append(r["id"])
            continue
        for system in [k for k in ("sciweaver", "react", "gemini", "strong", "self_consistency") if k in r]:
            x = items[r["id"]]
            s = (check if x["answer_format"] == "code" else score)(x, r[system]["answer"])
            rows[(r["set"], system)].append(s)
            rows[("ALL", system)].append(s)
    out = {f"{k[0]}|{k[1]}": summarize(v) for k, v in sorted(rows.items())}
    out["dropped_items"] = dropped
    (run / "rescored.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k.ljust(26), v)


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), sys.argv[2] if len(sys.argv) > 2 else "bio")
