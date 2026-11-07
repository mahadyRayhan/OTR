"""
build_bench_index.py — Chroma indexes for the benchmark-aligned setting of one domain.

Each index holds the deployed knowledge base (the haystack, chunked exactly as the deployed
builders do) plus the documents collect_<domain>_bench.py added for the test questions:

  rag    deployed PDFs + Europe PMC corpus, LitQA2 source papers, BioASQ gold abstracts
  code   deployed notebooks + PDFs, BioCoder repositories (target functions removed), tool READMEs
  cloud  deployed markdown, tool READMEs, nf-core analogue modules (one per document, defaults resolved)

Collections, embedding models and chunk formats match the deployed agents, so the agents load these
indexes unchanged when SCIWEAVER_INDEX_DIR=<out>/<domain> is set.

No LLM calls. Usage: python journal_fixes/scripts/build_bench_index.py --domain bio [--only rag,code,cloud]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAG_PDF_EXTRACTOR", "fitz")
ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path.home() / "otr_local"
sys.path.insert(0, str(ROOT))


def header_fields(text: str) -> dict:
    """The 'key: value' lines that collect_*_bench.py writes under each document title."""
    return dict(re.findall(r"^(\w+): (.*)$", text.split("\n\n", 1)[0], re.M))


# ── RAG ──────────────────────────────────────────────────────────────────────

def build_rag(src: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import rag_agent.build_index as bi  # reads SCIWEAVER_INDEX_DIR through rag_config
    from rag_agent.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.load()
    kb.chroma.reset()
    kb.chroma.load()
    texts, metas, ids = [], [], []

    def add(cid, title, chunk, rec, source):
        texts.append(f"{title} {chunk}")
        metas.append({"title": title, "authors": "", "journal": rec.get("journal") or "",
                      "year": int(rec.get("year") or 0) if str(rec.get("year") or "0").isdigit() else 0,
                      "pmid": str(rec.get("pmid") or cid), "abstract": chunk, "url": rec.get("url") or "",
                      "source": source})
        ids.append(cid)

    # Deployed haystack, chunked as rag_agent/build_index.py does.
    for pdf in sorted(bi.KB_DIR.glob("*.pdf")):
        text = bi.clean_text(bi.extract_pdf_text(pdf))
        meta = bi.parse_metadata(text, pdf.stem)
        for i, ch in enumerate(bi.chunk_text(text)):
            add(f"LOCAL:{pdf.stem[:28]}:c{i}", meta["title"], ch, {"year": meta["year"], "journal": meta["journal"]}, "offline_pdf")
    oa = json.loads((bi.OA_DIR / "manifest.json").read_text())
    for pmcid, rec in sorted(oa.items()):
        f = bi.OA_DIR / f"{pmcid}.txt"
        if f.exists():
            for i, ch in enumerate(bi.chunk_text(bi.clean_text(f.read_text(encoding="utf-8")))):
                add(f"PMC:{pmcid}:c{i}", rec.get("title") or "", ch, rec, "europepmc_oa")
    n_deployed = len(texts)

    # Added documents.
    for f in sorted((src / "rag" / "litqa2").glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        rec = header_fields(text)
        for i, ch in enumerate(bi.chunk_text(bi.clean_text(text))):
            add(f"BENCH:{f.stem}:c{i}", rec.get("title", ""), ch, rec, "bench_litqa2_source")
    for f in sorted((src / "rag" / "bioasq").glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        rec = header_fields(text)
        abstract = bi.clean_text(text.split("## Abstract", 1)[-1])
        if len(abstract.split()) >= 20:  # abstracts are one chunk each, even when shorter than MIN_CHUNK_WORDS
            add(f"PMID:{f.stem}", f"PubMed {f.stem}", abstract, rec, "bench_bioasq_abstract")

    print(f"[rag] deployed chunks {n_deployed}, added {len(texts) - n_deployed}; embedding...")
    kb.chroma.add(texts, metas, ids)
    print(f"[rag] collection size {kb.chroma.count()}")


# ── code / cloud ─────────────────────────────────────────────────────────────

def _collection(path: Path, name: str):
    import chromadb
    from chromadb.utils import embedding_functions
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.create_collection(name=name, embedding_function=ef)


def _deployed_docs(builder: str) -> tuple[list, list, list]:
    """Run a deployed builder against a throwaway Chroma path and keep the docs it prepared."""
    import chromadb
    real = chromadb.PersistentClient
    tmp = LOCAL / "indexes" / "_scratch"
    chromadb.PersistentClient = lambda path: real(path=str(tmp))
    try:
        g = runpy.run_path(str(ROOT / builder), run_name="__deployed__")
    finally:
        chromadb.PersistentClient = real
    return g["docs"], g["metadatas"], g["ids"]


def _add(col, docs, metas, ids) -> None:
    for i in range(0, len(docs), 1000):
        col.add(documents=docs[i:i + 1000], metadatas=metas[i:i + 1000], ids=ids[i:i + 1000])


def _readme_pieces(f: Path) -> list[str]:
    """README split by headings, long sections split further at 1,500 characters; each piece keeps the title."""
    text = f.read_text(errors="ignore")
    title = text.splitlines()[0]
    out = []
    for sec in re.split(r"\n(?=#{1,3} )", text):
        for j in range(0, len(sec), 1500):
            piece = sec[j:j + 1500].strip()
            if len(piece) > 80:
                out.append(piece if piece.startswith(title) else f"{title}\n{piece}")
    return out


def build_code(src: Path, out: Path) -> None:
    docs, metas, ids = _deployed_docs("code_agent/build_vector_db.py")
    n_deployed = len(docs)
    for repo in sorted((src / "code" / "repos").iterdir()):
        for f in sorted(repo.rglob("*")):
            if not f.is_file() or f.suffix not in (".py", ".md", ".rst"):
                continue
            rel = f"{repo.name.replace('__', '/')}/{f.relative_to(repo)}"
            text = f.read_text(errors="ignore")
            if f.suffix == ".py":  # 60-line windows, 10 lines of overlap, labelled with the file path
                lines = text.splitlines()
                pieces = ["\n".join(lines[i:i + 60]) for i in range(0, max(1, len(lines)), 50)]
                pieces = [f"### File: {rel}\n```python\n{p}\n```" for p in pieces if p.strip()]
            else:
                pieces = [f"### File: {rel}\n{text[i:i + 800]}" for i in range(0, len(text), 700) if text[i:i + 800].strip()]
            for k, p in enumerate(pieces):
                docs.append(p)
                metas.append({"source": rel, "chunk_index": k})
                ids.append(f"bench:{rel}:{k}")
    for f in sorted((src / "cloud" / "tool_docs").glob("*.md")):  # tool READMEs document the command line
        for k, piece in enumerate(_readme_pieces(f)):
            docs.append(piece)
            metas.append({"source": f"tool_docs/{f.name}", "chunk_index": k})
            ids.append(f"bench:code:tool_docs/{f.stem}:{k}")
    col = _collection(out / "code", "scanpy_code_vault")
    print(f"[code] deployed chunks {n_deployed}, added {len(docs) - n_deployed}; embedding...")
    _add(col, docs, metas, ids)
    print(f"[code] collection size {col.count()}")


def build_cloud(src: Path, out: Path) -> None:
    docs, metas, ids = _deployed_docs("cloud_agent/build_cloud_db.py")
    n_deployed = len(docs)
    for f in sorted((src / "cloud" / "tool_docs").glob("*.md")):
        for k, piece in enumerate(_readme_pieces(f)):
            docs.append(piece)
            metas.append({"source": f"tool_docs/{f.name}"})
            ids.append(f"bench:tool_docs/{f.stem}:{k}")
    # One document per analogue module, with its label resolved to the template defaults, so a query about
    # one tool retrieves the modules of similar tools (a 15-row chunk resembles no tool in particular).
    from journal_fixes.scripts.build_testsets import NFCORE_LABELS
    for row in (src / "cloud" / "nfcore_analogues.md").read_text().splitlines():
        if not row.startswith("| ") or row.startswith(("| module", "| ---")):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split(" | ")]
        mod, tool, labels, desc = cells[0], cells[1], cells[-1], " | ".join(cells[2:-1])
        labs = [l.strip() for l in labels.split(",")]
        main = "process_high_memory" if "process_high_memory" in labs else next(
            (l for l in ("process_high", "process_medium", "process_low", "process_single") if l in labs), None)
        res = f" Default request ({main}, nf-core base.config): {NFCORE_LABELS[main][0]} CPUs, {NFCORE_LABELS[main][1]} GB memory." if main else ""
        docs.append(f"nf-core module {mod} (tool: {tool}): {desc}. Resource labels: {labels}.{res}")
        metas.append({"source": "nfcore_analogues.md", "module": mod})
        ids.append(f"bench:nfcore_analogues:{mod}")
    col = _collection(out / "cloud", "cloud_architecture_vault")
    print(f"[cloud] deployed chunks {n_deployed}, added {len(docs) - n_deployed}; embedding...")
    _add(col, docs, metas, ids)
    print(f"[cloud] collection size {col.count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--only", default="rag,code,cloud")
    a = ap.parse_args()
    src = LOCAL / "bench_kb" / a.domain
    out = LOCAL / "indexes" / a.domain
    os.environ["SCIWEAVER_INDEX_DIR"] = str(out)
    parts = a.only.split(",")
    if "code" in parts:
        build_code(src, out)
    if "cloud" in parts:
        build_cloud(src, out)
    if "rag" in parts:
        build_rag(src)
