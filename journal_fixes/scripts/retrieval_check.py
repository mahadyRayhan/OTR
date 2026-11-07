"""
retrieval_check.py — are the benchmark-aligned indexes and the test questions really aligned?

For every question, query the agent's Chroma collection with the question text alone and check
whether a gold document is among the top-k chunks (recall@k). No LLM is involved, so this is a
lower bound on what the agents can retrieve (they also rewrite queries and rerank).

  rag_litqa2     gold = chunks of the item's source paper
  rag_bioasq     gold = any of the item's gold abstracts
  code_biocoder  gold = a chunk from the item's file (file hit) or repository (repo hit)
  cloud_nfcore   gold = the tested tool's README (questions name the tool)
  full_system    gold = the tested tool's README

Output: journal_fixes/results/retrieval_check_<domain>.json
Usage: python journal_fixes/scripts/retrieval_check.py --domain bio [--k 10]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path.home() / "otr_local"


def collection(path: Path, name: str):
    import chromadb
    from chromadb.utils import embedding_functions
    return chromadb.PersistentClient(path=str(path)).get_collection(
        name=name, embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))


def top(col, text: str, k: int) -> tuple[list[str], list[dict]]:
    r = col.query(query_texts=[text], n_results=k)
    return r["ids"][0], r["metadatas"][0]


def main(domain: str, k: int) -> None:
    idx = LOCAL / "indexes" / domain
    sets = ROOT / "journal_fixes" / "data" / "testsets" / domain
    man = json.loads((LOCAL / "bench_kb" / domain / "manifest.json").read_text())
    load = lambda n: [json.loads(l) for l in open(sets / f"{n}.jsonl")]
    report = {}

    os.environ["SCIWEAVER_INDEX_DIR"] = str(idx)
    import sys
    sys.path.insert(0, str(ROOT))
    from rag_agent.rag_config import settings
    rag = collection(Path(settings.chroma_db_path), settings.chroma_collection_name)

    # File stems as collect_bio_bench.py names them: the PMCID, else "biorxiv_<doi>".
    stem_of = {doi: v.get("pmcid") or "biorxiv_" + re.sub(r"\W", "_", doi)
               for doi, v in man["litqa2"].items() if v["status"] == "full text"}
    hits = []
    for x in load("rag_litqa2"):
        stems = {stem_of.get(s.lower().replace("https://doi.org/", "")) for s in x["meta"]["sources"]} - {None}
        ids, _ = top(rag, x["question"], k)
        hits.append(any(i.split(":")[1] in stems for i in ids if i.startswith("BENCH:")))
    report["rag_litqa2"] = {"n": len(hits), f"recall@{k}": round(sum(hits) / len(hits), 3)}

    hits = []
    for x in load("rag_bioasq"):
        ids, _ = top(rag, x["question"], k)
        hits.append(bool({f"PMID:{p}" for p in x["meta"]["gold_pmids"]} & set(ids)))
    report["rag_bioasq"] = {"n": len(hits), f"recall@{k}": round(sum(hits) / len(hits), 3)}

    code = collection(idx / "code", "scanpy_code_vault")
    file_hits, repo_hits = [], []
    for x in load("code_biocoder"):
        _, metas = top(code, x["question"], k)
        srcs = [m.get("source", "") for m in metas]
        repo = x["meta"]["repository"]
        file_hits.append(f'{repo}/{x["meta"]["file"]}' in srcs)
        repo_hits.append(any(s.startswith(repo + "/") for s in srcs))
    report["code_biocoder"] = {"n": len(file_hits), f"file_recall@{k}": round(sum(file_hits) / len(file_hits), 3),
                               f"repo_recall@{k}": round(sum(repo_hits) / len(repo_hits), 3)}

    cloud = collection(idx / "cloud", "cloud_architecture_vault")
    docs = {t for t, v in man["tool_docs"].items() if v["status"] == "ok"}
    for name in ("cloud_nfcore", "full_system"):
        hits = []
        for x in load(name):
            if x["meta"]["tool"] not in docs:
                continue
            _, metas = top(cloud, x["question"], k)
            hits.append(any(m.get("source") == f'tool_docs/{x["meta"]["tool"]}.md' for m in metas))
        report[name] = {"n_with_readme": len(hits), f"readme_recall@{k}": round(sum(hits) / max(1, len(hits)), 3)}

    out = ROOT / "journal_fixes" / "results" / f"retrieval_check_{domain}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    main(a.domain, a.k)
