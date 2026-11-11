"""
node_signal_analysis.py — does SciWeaver's internal LLM judge recognise a node that found the gold answer?
Offline: reads a pilot run's answers and SciWeaver traces, no LLM calls.

For every DAG node that states an answer we can check (MCQ / yes-no letter, factoid, DOIs, cloud resources),
the label is "this node's answer is the gold answer". Signals available at run time are then compared by
how well they separate good from bad nodes (AUROC, 0.5 = chance):

  judge      the node's LLM-judge quality score (what triggers REFINE / BRANCH below 0.75)
  agreement  share of the other answering nodes in the same run that give the same answer
  support    the node's answer is found in the documents the node retrieved (cloud: the nf-core label or
             CPU/memory; MCQ: the chosen option's words vs. the other options' words, in the retrieved text)
  root       the node is a root node (not REFINE / BRANCH / AGGREGATE)

Then, per item, what the final answer would have been under simple selection rules (majority vote,
highest judge score, ...) and how often a node had the gold answer but the final answer did not.
BioCoder items are left out (checking a node's code needs the test harness).

Usage: python journal_fixes/scripts/node_signal_analysis.py [--run ~/otr_local/runs/pilot_bio_shared_small]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from journal_fixes.scripts.scoring import DOI_RE, _norm, _norm_doi, score  # noqa: E402

LOCAL = Path.home() / "otr_local"
THRESHOLD = 0.75  # adaptive_got_dag quality_threshold: below it a node is expanded
LABEL_RE = re.compile(r"process_(?:single|low|medium|high_memory|high|long)", re.I)
CPU_RE = re.compile(r"\b(\d{1,3})\s*(?:v?CPUs?|vCPUs?|cores?)\b", re.I)
MEM_RE = re.compile(r"\b(\d{1,4}(?:\.\d+)?)\s*(?:GB|GiB)\b", re.I)
LETTER_RE = re.compile(r"(?:ANSWER|RESULT)\s*[:：]\s*\**\(?([A-Z])\)?\b")
YESNO_RE = re.compile(r"(?:ANSWER|RESULT)\s*[:：]\s*\**(yes|no)\b", re.I)
STOP = set("the a an of and or to in for on with by is are be as at from that this which its it".split())


# ── documents a node retrieved ────────────────────────────────────────────────

class Docs:
    def __init__(self, index_dir: Path):
        import chromadb
        self.cloud = chromadb.PersistentClient(str(index_dir / "cloud")).get_collection("cloud_architecture_vault")
        self.rag = chromadb.PersistentClient(str(index_dir / "rag")).get_collection("rag_paper_vault")
        self.cache: dict[tuple, str] = {}

    def text(self, source: str, item_id: str) -> str:
        key = (source, item_id)
        if key not in self.cache:
            if source == "cloud_index":
                g = self.cloud.get(ids=[item_id], include=["documents"])
            elif source == "literature":
                g = self.rag.get(where={"pmid": item_id}, include=["documents"])
            else:
                g = {"documents": []}
            self.cache[key] = "\n".join(d or "" for d in g["documents"])
        return self.cache[key]

    def for_node(self, node: dict) -> str:
        return "\n".join(self.text(r["source"], it["id"]) for r in node.get("retrievals", [])
                         for it in r["items"] if r["source"] in ("cloud_index", "literature"))


# ── a node's answer and whether it is the gold answer ─────────────────────────

NFCORE = {"process_single": (1, 6), "process_low": (2, 12), "process_medium": (6, 36), "process_high": (12, 72),
          "process_high_memory": (12, 200)}  # nf-core base.config defaults (process_long sets only time)


def cloud_value(text: str):
    """(cpus, memory_gb): the first nf-core label's defaults, else the first CPU and memory figures."""
    for label in LABEL_RE.findall(text):
        if label.lower() in NFCORE:
            return NFCORE[label.lower()]
    cpu, mem = CPU_RE.search(text), MEM_RE.search(text)
    return (int(cpu.group(1)), float(mem.group(1))) if cpu and mem else None


def cloud_correct(value, gold: dict) -> bool:
    """The scorer's resources rule: CPU and memory both within a factor of 2 of the gold."""
    cpus, mem = value
    return gold["cpus"] / 2 <= cpus <= gold["cpus"] * 2 and gold["memory_gb"] / 2 <= mem <= gold["memory_gb"] * 2


def node_answer(item: dict, node: dict):
    """(kind, value, correct) for a node that states a checkable answer, else None."""
    text, fmt, gold = node.get("output") or "", item["answer_format"], item["gold"]
    if fmt == "mcq":
        m = LETTER_RE.findall(text)
        return ("mcq", m[-1], m[-1] == gold) if m else None
    if fmt == "yesno":
        m = YESNO_RE.findall(text)
        return ("yesno", m[-1].lower(), m[-1].lower() == gold) if m else None
    if fmt == "factoid":
        m = re.findall(r"(?:ANSWER|RESULT)\s*[:：]\s*(.+)", text, re.I)
        return ("factoid", _norm(m[-1]), score(item, f"ANSWER: {m[-1]}")["correct"]) if m else None
    if fmt == "doi" or (fmt == "structured" and item["agent"] != "cloud" and node.get("agent") in ("scholar", "rag")):
        found = {_norm_doi(d) for d in DOI_RE.findall(text)}
        golds = {_norm_doi(g) for g in (gold if fmt == "doi" else gold["literature"]["dois"])}
        return ("doi", frozenset(found), bool(found & golds)) if found else None
    if fmt == "structured":
        g = gold if item["agent"] == "cloud" else gold["cloud"]
        if item["agent"] != "cloud" and node.get("agent") != "cloud":
            return None
        v = cloud_value(text)
        return ("cloud", v, cloud_correct(v, g)) if v is not None else None
    return None


def same(a, b) -> bool:
    return bool(a & b) if isinstance(a, frozenset) else a == b


# ── signals ───────────────────────────────────────────────────────────────────

def support(item: dict, kind: str, value, docs_text: str) -> float | None:
    if not docs_text:
        return None
    low = docs_text.lower()
    if kind == "cloud":
        cpus, mem = value
        label = next((k for k, v in NFCORE.items() if v == value), None)
        if label and label in low:
            return 1.0
        return float(bool(re.search(rf"\b{cpus}\s*(?:v?cpus?|cores?)", low)) and
                     bool(re.search(rf"\b{mem:g}\s*(?:gb|gib|\.gb)", low)))
    if kind == "mcq" and item.get("meta", {}).get("choices"):
        words = set(re.findall(r"[a-z0-9]{3,}", low))
        def cover(opt):
            w = [x for x in re.findall(r"[a-z0-9]{3,}", opt.lower()) if x not in STOP]
            return sum(x in words for x in w) / len(w) if w else 0.0
        ch = item["meta"]["choices"]
        i = ord(value) - 65
        if i >= len(ch):
            return None
        others = [cover(c) for j, c in enumerate(ch) if j != i]
        return cover(ch[i]) - (max(others) if others else 0.0)
    if kind == "factoid":
        return float(bool(value) and value in _norm(docs_text))
    return None


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# ── main ──────────────────────────────────────────────────────────────────────

def load_items(domain: str) -> dict[str, dict]:
    items = {}
    for f in (ROOT / "journal_fixes/data/testsets" / domain).glob("*.jsonl"):
        for l in open(f):
            x = json.loads(l)
            items[x["id"]] = x
    return items


def main(run: Path, domain: str) -> None:
    items = load_items(domain)
    docs = Docs(LOCAL / "indexes" / domain)
    nodes, per_item = [], []
    for l in open(run / "answers.jsonl"):
        r = json.loads(l)
        sw = r["sciweaver"]
        if r["answer_format"] == "code" or not sw.get("trace_file"):
            continue
        item = {**items[r["id"]], "gold": r["gold"]}
        t = json.load(open(sw["trace_file"]))
        rows = []
        for nid, n in t["nodes"].items():
            if nid == "_run" or "/judge" in nid or nid.startswith("retry"):
                continue
            a = node_answer(item, n)
            if a is None:
                continue
            kind, value, ok = a
            rows.append({"item": r["id"], "set": r["set"], "node": nid, "op": n.get("op"), "agent": n.get("agent"),
                         "kind": kind, "value": value, "correct": bool(ok), "judge": n.get("quality"),
                         "support": support(item, kind, value, docs.for_node(n))})
        for x in rows:
            others = [y for y in rows if y is not x and y["kind"] == x["kind"]]
            x["agreement"] = sum(same(x["value"], y["value"]) for y in others) / len(others) if others else None
            x["root"] = float(x["op"] == "root")
        nodes += rows
        per_item.append((r, rows))

    out = {"run": str(run), "nodes": len(nodes), "items": len(per_item)}
    print(f"{len(per_item)} items, {len(nodes)} nodes with a checkable answer "
          f"({sum(x['correct'] for x in nodes)} correct)\n")

    # 1. how well each signal separates nodes with the gold answer from nodes without it
    print("AUROC (0.5 = chance, 1.0 = perfect) — does the signal rank gold-answer nodes above the others?")
    groups = {"all": nodes, **{k: [x for x in nodes if x["kind"] == k] for k in sorted({x["kind"] for x in nodes})}}
    out["auroc"] = {}
    print(f"  {'nodes':10} {'n':>4} {'n_ok':>5}  " + "".join(f"{s:>11}" for s in ("judge", "agreement", "support", "root")))
    for g, xs in groups.items():
        row = {}
        for s in ("judge", "agreement", "support", "root"):
            have = [x for x in xs if x.get(s) is not None]
            row[s] = (auroc([x[s] for x in have if x["correct"]], [x[s] for x in have if not x["correct"]]), len(have))
        out["auroc"][g] = row
        fmt = lambda v: f"{v[0]:.2f} ({v[1]:>3})" if v[0] is not None else f"{'—':>4} ({v[1]:>3})"
        print(f"  {g:10} {len(xs):>4} {sum(x['correct'] for x in xs):>5}  " + "".join(f"{fmt(row[s]):>11}" for s in row))

    # 2. the judge's decisions at its own threshold
    judged = [x for x in nodes if x["judge"] is not None]
    ok, bad = [x for x in judged if x["correct"]], [x for x in judged if not x["correct"]]
    low_ok, high_bad = sum(x["judge"] < THRESHOLD for x in ok), sum(x["judge"] >= THRESHOLD for x in bad)
    out["judge_at_threshold"] = {"correct_nodes": len(ok), "correct_judged_low": low_ok,
                                 "wrong_nodes": len(bad), "wrong_judged_high": high_bad}
    print(f"\nJudge at its threshold {THRESHOLD}: {low_ok}/{len(ok)} gold-answer nodes scored LOW (sent to refine/branch); "
          f"{high_bad}/{len(bad)} wrong nodes scored HIGH (accepted)")
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"  mean judge score: gold-answer nodes {mean([x['judge'] for x in ok]):.2f}, "
          f"wrong nodes {mean([x['judge'] for x in bad]):.2f}")

    # 3. per item: found then lost, and what simple selection rules would have answered
    rules = defaultdict(list)
    lost = []
    for r, rows in per_item:
        final_ok = bool(r["sciweaver"].get("correct"))
        rules["actual final answer"].append(final_ok)
        rules["react (for reference)"].append(bool(r["react"].get("correct")))
        if r["set"] == "full_system":  # composite answer: only the final answer is scored as a whole
            continue
        rules["oracle: any node had gold"].append(any(x["correct"] for x in rows))
        if any(x["correct"] for x in rows) and not final_ok:
            lost.append((r["set"], r["id"]))

        def pick(xs, key):
            return max(xs, key=key)["correct"] if xs else False

        def vote(xs):
            if not xs:
                return False
            if xs[0]["kind"] == "doi":
                c = Counter(d for x in xs for d in x["value"])
                best = c.most_common(1)[0][0]
                return any(best in x["value"] and x["correct"] for x in xs) or best in {
                    _norm_doi(g) for g in (r["gold"] if isinstance(r["gold"], list) else [])}
            c = Counter(x["value"] for x in xs)
            top = c.most_common(1)[0][0]
            return next(x["correct"] for x in xs if x["value"] == top)
        non_agg = [x for x in rows if x["op"] != "aggregate"]
        roots = [x for x in rows if x["op"] == "root"]
        rules["majority vote, all nodes"].append(vote(non_agg))
        rules["majority vote, root nodes"].append(vote(roots))
        rules["highest judge score"].append(pick([x for x in non_agg if x["judge"] is not None], lambda x: x["judge"]))
        rules["first root node"].append(roots[0]["correct"] if roots else False)
    out["found_then_lost"] = lost
    out["selection_rules"] = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)} for k, v in rules.items()}
    n_single = len(rules["oracle: any node had gold"])
    print(f"\nItems where some node had the gold answer but the final answer was wrong: {len(lost)}/{n_single} "
          f"(full_system excluded)")
    for s, i in lost:
        print(f"  {s:20} {i}")
    print("\nAccuracy if the final answer were chosen by a simple rule (items with a checkable answer):")
    for k, v in rules.items():
        print(f"  {k:30} {sum(v) / len(v):.3f}  (n={len(v)})")

    for x in nodes:
        x["value"] = sorted(x["value"]) if isinstance(x["value"], frozenset) else x["value"]
    (run / "node_signals.json").write_text(json.dumps({**out, "node_rows": nodes}, indent=1, default=str))
    print(f"\nwritten: {run / 'node_signals.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=LOCAL / "runs/pilot_bio_shared_small")
    ap.add_argument("--domain", default="bio")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main(a.run.expanduser(), a.domain)
