"""
leakage_check.py — does any agent's knowledge base already contain a public benchmark's answers?

Knowledge bases scanned
  rag_agent/knowledge_base     (Europe PMC full texts + original PDFs)
  code_agent/knowledge_base    (GitHub notebooks + PDFs)
  cloud_agent/knowledge_base   (markdown)
  scholar_agent/scholar_cache.db (cached Semantic Scholar titles and abstracts)

Benchmarks: LAB-Bench (LitQA2, ProtocolQA, SeqQA, DbQA), PubMedQA (labelled), MatSciBench,
SciCode, BioCoder, BixBench.

Checks
  1. identifiers  KB documents whose DOI / PMID / PMCID is a benchmark source paper
  2. repositories code-KB notebooks from a repository that a code benchmark is drawn from
  3. canary       benchmark canary strings present in any KB file
  4. 13-grams     word 13-grams shared by a benchmark item and the KB (the contamination test of
                  Brown et al. 2020). An item is flagged when its gold fields (answer, key passage,
                  solution, reference code, PubMedQA abstract) share >= 1 13-gram, or its question
                  shares >= 3. Each hit is kept with the KB file it came from, for manual review.

Output: journal_fixes/results/leakage_check.json, journal_fixes/data/leakage_flags.json
        (benchmark -> list of flagged item ids; the test-set builder excludes them).

No LLM calls. Usage: python journal_fixes/scripts/leakage_check.py [--bench-dir ~/otr_local/benchmarks]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = 13
TOKEN = re.compile(r"[a-z0-9]+")
DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.I)


def tokens(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def grams(text: str) -> set[int]:
    t = tokens(text)
    return {hash(" ".join(t[i:i + N])) for i in range(len(t) - N + 1)}


def norm_doi(s: str) -> str:
    m = DOI.search(s or "")
    return m.group(0).lower().rstrip(".,;)") if m else ""


# ── knowledge bases ──────────────────────────────────────────────────────────

def kb_documents() -> list[tuple[str, str, str]]:
    """(agent, file label, text) for every KB document."""
    import fitz
    docs = []
    for agent in ("rag", "code", "cloud"):
        for f in sorted((ROOT / f"{agent}_agent" / "knowledge_base").rglob("*")):
            if f.suffix in (".txt", ".md"):
                docs.append((agent, f.name, f.read_text(errors="ignore")))
            elif f.suffix == ".ipynb":
                nb = json.loads(f.read_text(errors="ignore"))
                docs.append((agent, f.name, "\n".join("".join(c.get("source", "")) for c in nb.get("cells", []))))
            elif f.suffix == ".pdf":
                with fitz.open(f) as pdf:
                    docs.append((agent, f.name, "\n".join(p.get_text() for p in pdf)))
    con = sqlite3.connect(ROOT / "scholar_agent" / "scholar_cache.db")
    for pid, title, abstract in con.execute("select paper_id, title, abstract from papers"):
        docs.append(("scholar", f"s2:{pid}", f"{title or ''}\n{abstract or ''}"))
    return docs


def kb_identifiers() -> dict[str, set[str]]:
    man = json.loads((ROOT / "rag_agent/knowledge_base/oa_corpus/manifest.json").read_text())
    ids = {"doi": set(), "pmid": set(), "pmcid": set()}
    for m in man.values():
        ids["doi"].add(norm_doi(m.get("doi") or ""))
        ids["pmid"].add(str(m.get("pmid") or ""))
        ids["pmcid"].add(str(m.get("pmcid") or "").upper())
    return {k: v - {""} for k, v in ids.items()}


def kb_repos() -> set[str]:
    man = json.loads((ROOT / "code_agent/knowledge_base/notebook_manifest.json").read_text())
    return {m["repo"].lower() for m in man.values()}


# ── benchmarks: (id, question text, gold text, source ids) ───────────────────

def load_benchmarks(bench: Path) -> dict[str, list[dict]]:
    import pandas as pd
    out = {}
    lb = ROOT / "journal_fixes/data/benchmarks"
    for sub in ("LitQA2", "ProtocolQA", "SeqQA", "DbQA"):
        out[f"labbench_{sub}"] = [
            {"id": x["id"], "q": x["question"], "gold": f'{x["ideal"]}\n{x.get("key-passage") or ""}',
             "dois": [norm_doi(s) for s in x.get("sources") or []], "canary": x.get("canary")}
            for x in json.loads((lb / f"labbench_{sub}.json").read_text())]
    p = pd.read_parquet(bench / "pubmedqa_labeled.parquet")
    out["pubmedqa"] = [{"id": str(r.pubid), "q": r.question,
                        "gold": "\n".join(r.context["contexts"]) + "\n" + r.long_answer, "pmids": [str(r.pubid)]}
                       for r in p.itertuples()]
    m = pd.read_parquet(bench / "matscibench.parquet")
    out["matscibench"] = [{"id": r.qid, "q": r.question, "gold": f"{r.solution}\n{r.answer}"} for r in m.itertuples()]
    sc = [json.loads(l) for f in ("scicode_test.jsonl", "scicode_dev.jsonl") for l in open(bench / f)]
    out["scicode"] = [{"id": x["problem_id"],
                       "q": x["problem_description_main"] + "\n" + "\n".join(s["step_description_prompt"] for s in x["sub_steps"]),
                       "gold": "\n".join(s.get("ground_truth_code") or "" for s in x["sub_steps"])} for x in sc]
    bc = json.loads((bench / "biocoder.json").read_text())
    out["biocoder"] = [{"id": x["test_case_id"], "q": f'{x["signature"]}\n{x["comment"]}', "gold": x["goldenCode"],
                        "repo": x["repository"].lower()} for x in bc]
    bx = [json.loads(l) for l in open(bench / "bixbench.jsonl")]
    out["bixbench"] = [{"id": x["id"], "q": x["question"], "gold": str(x["ideal"]),
                        "dois": [norm_doi(x.get("paper") or "")], "canary": x.get("canary")} for x in bx]
    return out


def main(bench: Path) -> None:
    docs = kb_documents()
    print(f"KB documents: {len(docs)}", collections.Counter(a for a, _, _ in docs))
    gram_src: dict[int, str] = {}
    for agent, name, text in docs:
        for g in grams(text):
            gram_src.setdefault(g, f"{agent}:{name}")
    print(f"KB 13-grams: {len(gram_src):,}")
    ids, repos = kb_identifiers(), kb_repos()
    benches = load_benchmarks(bench)

    canaries = {x["canary"] for items in benches.values() for x in items if x.get("canary")}
    canary_hits = sorted({f"{a}:{n}" for a, n, t in docs for c in canaries if c and c in t})

    report, flags = {"kb_documents": dict(collections.Counter(a for a, _, _ in docs)), "canary_hits": canary_hits,
                     "rule": "flag if gold fields share >= 1 13-gram with the KB, or the question shares >= 3, "
                             "or a source DOI/PMID is a KB document, or a code item's repository is in the code KB",
                     "benchmarks": {}}, {}
    for name, items in benches.items():
        flagged, examples = [], []
        id_hits = repo_hits = 0
        for x in items:
            why = []
            if any(d and d in ids["doi"] for d in x.get("dois", [])):
                why.append("source DOI is a KB document")
                id_hits += 1
            if any(p in ids["pmid"] for p in x.get("pmids", [])):
                why.append("source PMID is a KB document")
                id_hits += 1
            if x.get("repo") and x["repo"] in repos:
                why.append("repository is in the code KB")
                repo_hits += 1
            gold_hits = [gram_src[g] for g in grams(x["gold"]) if g in gram_src]
            q_hits = [gram_src[g] for g in grams(x["q"]) if g in gram_src]
            if gold_hits:
                why.append(f"{len(gold_hits)} gold 13-gram(s) in KB")
            if len(q_hits) >= 3:
                why.append(f"{len(q_hits)} question 13-grams in KB")
            if why:
                flagged.append(x["id"])
                examples.append({"id": x["id"], "why": why, "kb_sources": sorted(set(gold_hits + q_hits))[:5],
                                 "question": x["q"][:200]})
        flags[name] = flagged
        report["benchmarks"][name] = {"items": len(items), "flagged": len(flagged),
                                      "flagged_pct": round(100 * len(flagged) / max(1, len(items)), 2),
                                      "identifier_hits": id_hits, "repository_hits": repo_hits,
                                      "examples": examples[:25]}
        print(f"{name:22s} items {len(items):5d}  flagged {len(flagged):4d}")
    (ROOT / "journal_fixes/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "journal_fixes/results/leakage_check.json").write_text(json.dumps(report, indent=1))
    (ROOT / "journal_fixes/data/leakage_flags.json").write_text(json.dumps(flags, indent=1))
    print("canary hits:", canary_hits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(Path.home() / "otr_local" / "benchmarks"))
    main(Path(ap.parse_args().bench_dir))
