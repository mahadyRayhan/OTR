"""
judge_packets.py — blinded judge packets for grading a run's checklist answers by hand in Claude (web chat),
and reading the grades back. No API calls.

export <run>   writes <run>/judge_packets/:
                 README.md          what to do, step by step
                 packets/*.md       one file per question: the judging instructions, the question, its reference
                                    checklist and every system's answer under a random label (A, B, ...), with the
                                    order shuffled per question. multipart_qNN.md = the full question;
                                    parts_qNN.md = its three single-part versions (explain / code / cloud)
                 key.json           label -> system for every question. Never upload it.
                 replies/           save Claude's JSON reply for packets/X.md as replies/X.json
               Answers are blinded: text only SciWeaver produces (the engine's "Low-confidence parts" footer, the code
               agent's [AI-GENERATED] tags and agent banners) is removed. The automatic checklist scores the original.
import <run>   reads replies/*.json, un-blinds them with key.json, writes judge_scores.jsonl and judge_summary.json
               and prints mean scores per system (multi-part and single-part separately).

Usage: python journal_fixes/scripts/judge_packets.py export|import ~/otr_local/runs/pilot_bio_multipart
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ORDER = ["sciweaver", "react", "gemini", "strong", "self_consistency"]
CRITERIA = ["correctness", "completeness", "grounding", "code", "cloud"]

JUDGE_PROMPT = """You are an expert reviewer in bioinformatics, scientific software and cloud computing. You will grade
anonymous answers to a research request. The answers come from different AI systems; their labels (A, B, ...) are
random and their order is shuffled. Do not try to guess which system wrote which answer.

For each answer, grade only what is written. Do not reward length, confidence or formatting; a shorter answer that is
correct and complete beats a longer one with errors. Check claims, code and cloud designs yourself.

Score each answer from 1 (poor) to 5 (excellent) on:
- correctness: the scientific explanation is accurate; no wrong statements
- completeness: every part of the request is addressed, with the specific settings it asks for
- grounding: the papers cited are real, relevant and support the claims (DOIs must exist and match)
- code: the code would run as written, uses real library functions correctly and does what was asked
  (null if the request has no code part)
- cloud: the AWS design fits the workload, uses the requested services and instance, and is sensible on cost
  (null if the request has no cloud part)
Then give overall (1-10), and for each item of the reference checklist say whether the answer satisfies it (true/false).
The checklist is a minimum, not a full answer key; judge correctness yourself.
Finally rank all answers from best to worst.

Reply with ONLY one JSON object, no other text, in exactly this shape:
{"packet": "<packet id>",
 "items": [{"item": "<item id>",
            "grades": {"<label>": {"correctness": 1-5, "completeness": 1-5, "grounding": 1-5, "code": 1-5 or null,
                                   "cloud": 1-5 or null, "overall": 1-10, "checklist": {"<check id>": true/false},
                                   "main_error": "<one sentence, or empty>"}},
            "ranking": ["<best label>", "...", "<worst label>"]}]}
Give one entry in "items" per item in the packet, and grade every label of that item."""

README = """# Judge packets: grading by hand in Claude (web)

For each file in `packets/`:
1. Open a NEW chat in Claude (one chat per packet, so earlier answers do not influence later grades).
2. Upload the packet file (or paste its full text) and send it. The instructions are at the top of the file.
3. Copy Claude's reply (a single JSON object) into `replies/<same name>.json`,
   e.g. `packets/multipart_q05.md` -> `replies/multipart_q05.json`.

Do not upload `key.json`: it says which system wrote which answer.
Use the same Claude model for every packet, and note the model name and date here:

- Model:
- Date:

When all replies are saved, run:
    python journal_fixes/scripts/judge_packets.py import {run}
"""

BANNERS = re.compile(r"^.*\[(?:AI-GENERATED|PROVENANCE|Provenance-Aware[^\]]*|[A-Za-z ]*Agent)\].*$\n?", re.M)


def blind(text: str) -> str:
    text = re.split(r"\n\n\*\*Low-confidence parts\*\*", text or "")[0]
    text = re.sub(r"#\s*\[AI-GENERATED\]", "#", text)
    text = re.sub(r"CATALOG-VERIFIED(?: INSTANCE| choice| Instance)?:?\s*", "", text)  # the cloud agent's catalog tag
    return BANNERS.sub("", text).strip()


def export(run: Path) -> Path:
    answers = [json.loads(l) for l in open(run / "answers.jsonl")]
    answers = [a for a in answers if a["answer_format"] == "checklist"]
    items = {}
    for f in (ROOT / "journal_fixes/data/testsets/bio").glob("multipart_*.jsonl"):
        for l in open(f):
            x = json.loads(l)
            items[x["id"]] = x
    found = {k for a in answers for k, v in a.items() if isinstance(v, dict) and "answer" in v}
    systems = [s for s in SYSTEM_ORDER if s in found] + sorted(found - set(SYSTEM_ORDER))  # e.g. merged model variants
    out = run / "judge_packets"
    (out / "packets").mkdir(parents=True, exist_ok=True)
    (out / "replies").mkdir(exist_ok=True)
    rng = random.Random(0)
    key, packets = {}, defaultdict(list)
    for a in answers:
        x = items[a["id"]]
        parent = x["meta"].get("parent")
        packets[f"parts_{parent[3:]}" if parent else f"multipart_{a['id'][3:]}"].append((x, a))
    order = {"explain": 0, "code": 1, "cloud": 2}
    for pid, group in sorted(packets.items()):
        group.sort(key=lambda xa: order.get(xa[0]["meta"].get("section"), 0))
        lines = [f"# Judge packet {pid}", "", JUDGE_PROMPT, "", f'Packet id: "{pid}"', ""]
        for x, a in group:
            shown = [s for s in systems if s in a]
            rng.shuffle(shown)
            labels = {chr(65 + i): s for i, s in enumerate(shown)}
            key[x["id"]] = labels
            lines += ["---", f'## Item "{x["id"]}"', "", "### Request", "", x["question"], "",
                      "### Reference checklist (minimum; ids in brackets)", ""]
            lines += [f"- [{c['id']}] {c['desc']}" for c in x["gold"]["checks"]]
            for lab, s in labels.items():
                lines += ["", f"### Answer {lab}", "", blind(a[s].get("answer", "")) or "(empty answer)"]
            lines.append("")
        (out / "packets" / f"{pid}.md").write_text("\n".join(lines))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "README.md").write_text(README.format(run=run))
    print(f"judge packets: {len(packets)} files in {out / 'packets'} ({len(key)} items, systems {systems})")
    return out


def load_reply(path: Path) -> dict:
    text = path.read_text()
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())
    # the judge sometimes sends a corrected object after the first one: keep the last complete reply object
    found, dec = None, json.JSONDecoder()
    for m in re.finditer(r'\{\s*"packet"', text):
        try:
            found = dec.raw_decode(text, m.start())[0]
        except json.JSONDecodeError:
            pass
    return found if found is not None else json.loads(text[text.find("{"):text.rfind("}") + 1])


def import_(run: Path) -> None:
    out = run / "judge_packets"
    key = json.loads((out / "key.json").read_text())
    rows, missing = [], []
    for f in sorted((out / "packets").glob("*.md")):
        rp = out / "replies" / f"{f.stem}.json"
        if not rp.exists():
            missing.append(f.stem)
            continue
        reply = load_reply(rp)
        for it in reply["items"]:
            labels = key[it["item"]]
            ranking = it.get("ranking", [])
            for lab, g in it["grades"].items():
                ck = g.get("checklist") or {}
                rows.append({"packet": f.stem, "item": it["item"], "system": labels[lab], "label": lab,
                             **{c: g.get(c) for c in CRITERIA + ["overall", "main_error"]},
                             "checklist_share": round(sum(map(bool, ck.values())) / len(ck), 3) if ck else None,
                             "rank": ranking.index(lab) + 1 if lab in ranking else None})
    with open(run / "judge_scores.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {}
    for kind in ("multipart", "parts"):
        per = defaultdict(list)
        for r in rows:
            if r["packet"].startswith(kind):
                per[r["system"]].append(r)
        summary[kind] = {s: {m: round(sum(v[m] for v in rs if v[m] is not None) /
                                      max(1, sum(v[m] is not None for v in rs)), 2)
                             for m in CRITERIA + ["overall", "checklist_share", "rank"]} | {"n": len(rs)}
                         for s, rs in per.items()}
    (run / "judge_summary.json").write_text(json.dumps(summary, indent=1))
    if missing:
        print(f"no reply yet for {len(missing)} packets: {', '.join(missing)}")
    for kind, per in summary.items():
        print(f"\n{kind}: mean judge grades (rank: 1 = best)")
        print(f"  {'system':18}" + "".join(f"{m[:11]:>12}" for m in CRITERIA + ["overall", "checklist_share", "rank"]))
        for s, m in sorted(per.items(), key=lambda kv: -kv[1]["overall"]):
            print(f"  {s:18}" + "".join(f"{m[k]:>12}" for k in CRITERIA + ["overall", "checklist_share", "rank"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    (export if a.action == "export" else import_)(a.run.expanduser())
    sys.exit(0)
