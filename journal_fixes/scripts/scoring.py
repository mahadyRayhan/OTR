"""
scoring.py — judge-free scoring for the test sets in journal_fixes/data/testsets/.

  prompt(item)          the question plus a format instruction for its answer type
  score(item, answer)   {"correct": bool | None, "answered": bool, "parts": {...}, "parsed": ...}
                        correct is None when the format needs an external harness (code execution)
  summarize(rows)       accuracy, precision (correct / answered), coverage (answered / total), and
                        per-part accuracy for structured items

Formats
  mcq           choices lettered A..; an "Insufficient information" option is added last (as in
                LAB-Bench), so a system can abstain. Abstaining counts as not answered.
  yesno         yes / no
  factoid       the answer contains any gold synonym, a gold synonym contains the answer, or their
                word-overlap F1 is >= 0.8 (case, punctuation and hyphens normalised)
  doi           any DOI in the answer equals a gold DOI (case-insensitive, "https://doi.org/" stripped)
  structured    cloud: CPU and memory within a factor of 2 of the nf-core default, and the instance is a
                valid catalog instance that fits the answer's own CPU and memory at <= 1.25x the cheapest
                that does (exact nf-core match and the gold instance are reported, not required).
                Full system: DOI, command (executable and, if any, subcommand) and the two cloud parts;
                the item is correct only if all four are. Every part is reported in summarize().
  code          needs a test harness; scored as None here (BioCoder: biocoder_eval.check).

Self-test: python journal_fixes/scripts/scoring.py
"""

from __future__ import annotations

import json
import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNSURE = "Insufficient information to answer the question"
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;)\]]+", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower().replace("-", " "))).strip()


def _norm_doi(s: str) -> str:
    return s.lower().replace("https://doi.org/", "").replace("doi:", "").strip().rstrip(".")


def _token_f1(a: str, b: str) -> float:
    """SQuAD-style word-overlap F1 of two normalised strings."""
    ta, tb = a.split(), b.split()
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    return 2 * common / (len(ta) + len(tb)) if common else 0.0


def _choices(item: dict) -> list[str]:
    return item["meta"]["choices"] + [UNSURE]


def _last_answer_line(text: str) -> str | None:
    hits = re.findall(r"ANSWER\s*[:：]\s*(.+)", text or "", re.I)
    return hits[-1].strip().strip("*`").strip() if hits else None


def _last_json(text: str, keys: tuple[str, ...]) -> dict | None:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or "", re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and any(k in d for k in keys):
            return d
    return None


# ── prompts ──────────────────────────────────────────────────────────────────

def prompt(item: dict) -> str:
    q, fmt = item["question"], item["answer_format"]
    if fmt == "mcq":
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(_choices(item)))
        return (f"{q}\n\nOptions:\n{opts}\n\nAnswer the question. End your response with a final line of the form "
                f"'ANSWER: <letter>'.")
    if fmt == "yesno":
        return f"{q}\n\nEnd your response with a final line of the form 'ANSWER: yes' or 'ANSWER: no'."
    if fmt == "factoid":
        return f"{q}\n\nEnd your response with a final line of the form 'ANSWER: <the entity or short phrase>'."
    if fmt == "doi":
        return f"{q}\n\nEnd your response with a final line of the form 'ANSWER: <DOI>'."
    if fmt == "structured" and item["agent"] == "cloud":
        return (f"{q}\n\nEnd your response with one JSON object on its own line: "
                '{"cpus": <integer>, "memory_gb": <number>, "instance": "<EC2 instance type>"}')
    if fmt == "structured":
        return (f"{q}\n\nEnd your response with one JSON object on its own line: "
                '{"doi": "<DOI>", "command": "<the command line>", "cpus": <integer>, '
                '"memory_gb": <number>, "instance": "<EC2 instance type>"}')
    if fmt == "code":
        return f"{q}\n\nGive the complete function, with the imports it needs, in one ```python code block."
    return q


# ── scoring ──────────────────────────────────────────────────────────────────

_CATALOG = None


def _catalog() -> dict[str, dict]:
    """Valid answers: current-generation x86 EC2 instances without GPU, burstable t-family excluded."""
    global _CATALOG
    if _CATALOG is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from journal_fixes.scripts.build_testsets import ec2_catalog
        _CATALOG = {r["instance"].lower(): r for r in ec2_catalog()}
    return _CATALOG


def _num(x) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None


def _cloud_parts(gold: dict, d: dict | None) -> dict:
    """resources: CPU and memory both within a factor of 2 of the nf-core default (the default is a
    convention, so an exact match is reported only as nfcore_exact). instance: a valid catalog instance
    that fits the answer's own CPU and memory and costs <= 1.25x the cheapest instance that does."""
    keys = ("resources", "instance", "nfcore_exact", "gold_instance")
    if not d:
        return dict.fromkeys(keys, False)
    cpus, mem = _num(d.get("cpus")), _num(d.get("memory_gb"))
    within = lambda p, g: p is not None and g / 2 <= p <= g * 2
    cat = _catalog()
    row = cat.get(str(d.get("instance") or "").strip().lower())
    instance = False
    if row and cpus and mem and row["vcpu"] >= cpus and row["mem"] >= mem:
        best = min(r["usd"] for r in cat.values() if r["vcpu"] >= cpus and r["mem"] >= mem)
        instance = row["usd"] <= best * 1.25
    return {"resources": within(cpus, gold["cpus"]) and within(mem, gold["memory_gb"]), "instance": instance,
            "nfcore_exact": cpus == gold["cpus"] and mem == gold["memory_gb"],
            "gold_instance": bool(row) and row["instance"].lower() in {a.lower() for a in gold["acceptable"]}}


_CLOUD_REQUIRED = ("resources", "instance")


def _command_ok(gold: list[str], cmd: str) -> bool:
    toks = [t.split("/")[-1].lower() for t in re.split(r"[\s|;&]+", cmd or "") if t]
    exe = gold[0].lower()
    if exe not in toks:
        return False
    return len(gold) < 2 or gold[1].lower() in toks[toks.index(exe) + 1:]


def score(item: dict, answer: str) -> dict:
    fmt, gold = item["answer_format"], item["gold"]
    if fmt == "mcq":
        line = _last_answer_line(answer) or ""
        m = re.match(r"\(?([A-Z])\)?\b", line.upper())
        letter = m.group(1) if m else None
        unsure = LETTERS[len(_choices(item)) - 1]
        answered = letter is not None and letter != unsure
        return {"correct": letter == gold, "answered": answered, "parsed": letter, "parts": {}}
    if fmt == "yesno":
        line = (_last_answer_line(answer) or "").lower()
        p = "yes" if line.startswith("yes") else "no" if line.startswith("no") else None
        return {"correct": p == gold, "answered": p is not None, "parsed": p, "parts": {}}
    if fmt == "factoid":
        line = _last_answer_line(answer)
        syn = [_norm(s) for g in gold for s in (g if isinstance(g, list) else [g])]
        p = _norm(line) if line else ""
        within = lambda a, b: f" {a} " in f" {b} "  # whole-word containment, so "9" does not match "19"
        ok = bool(p) and any(s and (within(s, p) or (len(p) >= 3 and within(p, s)) or _token_f1(p, s) >= 0.8) for s in syn)
        return {"correct": ok, "answered": bool(p), "parsed": line, "parts": {}}
    if fmt == "doi":
        line = _last_answer_line(answer) or ""
        found = {_norm_doi(d) for d in DOI_RE.findall(line)}
        return {"correct": bool(found & {_norm_doi(g) for g in gold}), "answered": bool(found),
                "parsed": sorted(found), "parts": {}}
    if fmt == "structured" and item["agent"] == "cloud":
        d = _last_json(answer, ("cpus", "instance"))
        parts = _cloud_parts(gold, d)
        return {"correct": all(parts[k] for k in _CLOUD_REQUIRED), "answered": d is not None, "parsed": d, "parts": parts}
    if fmt == "structured":
        d = _last_json(answer, ("doi", "command", "instance"))
        parts = {"doi": bool(d) and _norm_doi(str(d.get("doi") or "")) in {_norm_doi(x) for x in gold["literature"]["dois"]},
                 "command": bool(d) and _command_ok(gold["code"]["command"], str(d.get("command") or "")),
                 **{f"cloud_{k}": v for k, v in _cloud_parts(gold["cloud"], d).items()}}
        required = ("doi", "command") + tuple(f"cloud_{k}" for k in _CLOUD_REQUIRED)
        return {"correct": all(parts[k] for k in required), "answered": d is not None, "parsed": d, "parts": parts}
    return {"correct": None, "answered": bool(answer), "parsed": None, "parts": {},
            "note": "needs the official benchmark harness"}


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    answered = sum(r["answered"] for r in scored)
    correct = sum(bool(r["correct"]) for r in scored)
    out = {"n": n, "accuracy": round(correct / n, 3) if n else None,
           "precision": round(correct / answered, 3) if answered else None,
           "coverage": round(answered / n, 3) if n else None}
    keys = sorted({k for r in scored for k in r["parts"]})
    if keys:
        out["parts"] = {k: round(sum(bool(r["parts"].get(k)) for r in scored) / n, 3) for k in keys}
    return out


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    T = Path(__file__).resolve().parents[2] / "journal_fixes" / "data" / "testsets" / "bio"
    first = lambda f: json.loads(open(T / f).readline())
    mcq = first("rag_litqa2.jsonl")
    assert score(mcq, f"blah\nANSWER: {mcq['gold']}")["correct"]
    unsure = LETTERS[len(_choices(mcq)) - 1]
    r = score(mcq, f"ANSWER: {unsure}")
    assert not r["correct"] and not r["answered"]
    bq = [json.loads(l) for l in open(T / "rag_bioasq.jsonl")]
    yn = next(x for x in bq if x["answer_format"] == "yesno")
    assert score(yn, f"ANSWER: {yn['gold']}.")["correct"]
    fa = next(x for x in bq if x["answer_format"] == "factoid")
    g0 = fa["gold"][0][0] if isinstance(fa["gold"][0], list) else fa["gold"][0]
    assert score(fa, f"ANSWER: **{g0.upper()}**")["correct"], (fa["gold"], g0)
    assert not score(fa, "ANSWER: zzzz unrelated")["correct"]
    doi = first("scholar_toolpaper.jsonl")
    assert score(doi, f"ANSWER: https://doi.org/{doi['gold'][0].upper()}")["correct"]
    cl = first("cloud_nfcore.jsonl")
    g = cl["gold"]
    ans = f'text\n{{"cpus": {g["cpus"]}, "memory_gb": "{g["memory_gb"]} GB", "instance": "{g["cheapest"]}"}}'
    assert score(cl, ans)["correct"], score(cl, ans)
    full = first("full_system.jsonl")
    g = full["gold"]
    cmd = " ".join(g["code"]["command"]) + " --threads 4 in.bam"
    ans = json.dumps({"doi": g["literature"]["dois"][0], "command": f"/usr/bin/{cmd}", "cpus": g["cloud"]["cpus"],
                      "memory_gb": g["cloud"]["memory_gb"], "instance": g["cloud"]["cheapest"]})
    r = score(full, "Low-confidence parts: none\n" + ans)
    assert r["correct"], r
    bad = json.loads(ans)
    bad["cpus"] = 999
    r = score(full, json.dumps(bad))
    assert not r["correct"] and r["parts"]["doi"] and not r["parts"]["cloud_resources"]
    over = dict(json.loads(ans), cpus=g["cloud"]["cpus"] * 16, memory_gb=g["cloud"]["memory_gb"] * 16)
    assert not score(full, json.dumps(over))["parts"]["cloud_resources"]  # 16x over-provisioned
    print("scoring self-test passed;", summarize([score(mcq, f"ANSWER: {mcq['gold']}"), score(mcq, f"ANSWER: {unsure}")]))
