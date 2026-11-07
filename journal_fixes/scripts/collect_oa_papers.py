"""
collect_oa_papers.py — build a real, citable literature corpus for the RAG agent.

Replaces hand-picked PDFs with open-access full texts from Europe PMC:
  * only articles in the Europe PMC open-access subset with full text,
  * only Creative Commons licences (recorded per paper),
  * most-cited first within each topic.

Output
  rag_agent/knowledge_base/oa_corpus/<PMCID>.txt   (title, metadata header, abstract, body)
  rag_agent/knowledge_base/oa_corpus/manifest.json (one record per paper)

Usage
  python journal_fixes/scripts/collect_oa_papers.py --per-topic 8
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "rag_agent" / "knowledge_base" / "oa_corpus"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# (domain, topic query). Covers the benchmark's topics plus broader areas so the
# corpus is not tailored to the 30 evaluation queries.
TOPICS = [
    ("bioinformatics", '"community detection" AND (Leiden OR Louvain)'),
    ("bioinformatics", '"single-cell RNA-seq" AND clustering'),
    ("bioinformatics", '"differential expression" AND "RNA-seq" AND (DESeq2 OR edgeR OR limma)'),
    ("bioinformatics", '"adenine base editor" OR "adenine base editing"'),
    ("bioinformatics", '"cytosine base editor" OR "cytosine base editing"'),
    ("bioinformatics", '"prime editing"'),
    ("bioinformatics", '"CRISPR" AND "off-target" AND prediction'),
    ("bioinformatics", '"CITE-seq" OR "multimodal single-cell" AND integration'),
    ("bioinformatics", '"RNA-seq" AND "read alignment" AND (STAR OR HISAT2)'),
    ("bioinformatics", '"gene fusion" AND detection AND RNA-seq'),
    ("bioinformatics", '"batch correction" AND "single-cell"'),
    ("bioinformatics", '"trajectory inference" AND "single-cell"'),
    ("bioinformatics", '"spatial transcriptomics" AND analysis'),
    ("bioinformatics", '"ATAC-seq" AND "peak calling"'),
    ("bioinformatics", '"variant calling" AND (GATK OR DeepVariant)'),
    ("bioinformatics", '"metagenomic" AND "taxonomic classification"'),
    ("bioinformatics", '"protein structure prediction" AND AlphaFold'),
    ("bioinformatics", '"genome assembly" AND "long reads"'),
    ("bioinformatics", '"cell type annotation" AND "reference mapping"'),
    ("bioinformatics", '"gene set enrichment analysis"'),
    ("bioinformatics", '"doublet detection" AND "single-cell"'),
    ("bioinformatics", '"workflow management" AND (Snakemake OR Nextflow)'),
    ("bioinformatics", '"single-cell foundation model" OR "transformer" AND "single-cell" AND pretrained'),
    ("bioinformatics", '"genome-wide association" AND "fine-mapping"'),
    ("materials_science", '"density functional theory" AND "high-throughput" AND materials'),
    ("materials_science", '"machine learning interatomic potential" OR "machine-learned interatomic potentials"'),
    ("materials_science", '"molecular dynamics" AND materials AND simulation AND GPU'),
    ("materials_science", '"materials informatics" AND "machine learning"'),
    ("health_science", '"electronic health records" AND phenotyping AND "machine learning"'),
    ("health_science", '"clinical natural language processing"'),
    ("health_science", '"medical image segmentation" AND "deep learning"'),
    ("health_science", '"federated learning" AND healthcare'),
]

session = requests.Session()
session.headers["User-Agent"] = "SciWeaver-research-corpus/1.0 (University of Missouri)"


def search(query: str, n: int) -> list[dict]:
    q = f'({query}) AND OPEN_ACCESS:y AND HAS_FT:y AND (LICENSE:"cc by" OR LICENSE:"cc0" OR LICENSE:"cc by-sa" OR LICENSE:"cc by-nc" OR LICENSE:"cc by-nc-sa" OR LICENSE:"cc by-nd" OR LICENSE:"cc by-nc-nd")'
    r = session.get(f"{API}/search", params={
        "query": q, "format": "json", "pageSize": max(n * 3, 25),
        "sort": "CITED desc", "resultType": "core",
    }, timeout=60)
    r.raise_for_status()
    hits = r.json().get("resultList", {}).get("result", [])
    return [h for h in hits if h.get("pmcid")]


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def fetch_fulltext(pmcid: str) -> tuple[str, str] | None:
    r = session.get(f"{API}/{pmcid}/fullTextXML", timeout=90)
    if r.status_code != 200 or not r.text.strip():
        return None
    root = ET.fromstring(r.content)
    abstract = _text(root.find(".//abstract"))
    body = root.find(".//body")
    if body is None:
        return None
    parts = []
    for sec in body.iter("sec"):
        title = _text(sec.find("title"))
        paras = [_text(p) for p in sec.findall("p")]
        paras = [p for p in paras if p]
        if paras:
            parts.append((f"## {title}\n" if title else "") + "\n".join(paras))
    if not parts:
        parts = [_text(body)]
    return abstract, "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for domain, topic in TOPICS:
        try:
            hits = search(topic, args.per_topic)
        except Exception as exc:
            print(f"[search failed] {topic}: {exc}")
            continue
        kept = 0
        for h in hits:
            if kept >= args.per_topic:
                break
            pmcid = h["pmcid"]
            if pmcid in manifest:
                kept += 1
                continue
            try:
                ft = fetch_fulltext(pmcid)
            except Exception as exc:
                print(f"[fulltext failed] {pmcid}: {exc}")
                ft = None
            time.sleep(0.2)
            if not ft or len(ft[1].split()) < 1500:
                continue
            abstract, body = ft
            rec = {
                "pmcid": pmcid, "pmid": h.get("pmid"), "doi": h.get("doi"),
                "title": h.get("title"), "journal": h.get("journalTitle"),
                "year": h.get("pubYear"), "cited_by": h.get("citedByCount"),
                "license": h.get("license"), "domain": domain, "topic_query": topic,
                "url": f"https://europepmc.org/article/PMC/{pmcid}",
            }
            header = "\n".join(f"{k}: {v}" for k, v in rec.items() if k != "topic_query")
            (OUT_DIR / f"{pmcid}.txt").write_text(
                f"# {rec['title']}\n{header}\n\n## Abstract\n{abstract}\n\n{body}\n", encoding="utf-8")
            manifest[pmcid] = rec
            kept += 1
        print(f"{domain:18s} {kept:2d} papers  <- {topic}")
        manifest_path.write_text(json.dumps(manifest, indent=1))

    print(f"Total papers: {len(manifest)}")


if __name__ == "__main__":
    main()
