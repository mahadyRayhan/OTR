"""
build_scholar_cache.py — Pre-fetch researchers from Semantic Scholar and cache locally
=======================================================================================
Searches Semantic Scholar for top researchers across all bioinformatics topics
from the benchmark + eval query sets. Stores results in:

  1. scholar_agent/scholar_cache.db  (SQLite)
       • authors      — name, h-index, citation count, affiliations
       • papers       — title, abstract, year, citations
       • author_papers — junction
       • topic_authors — which topics each author covers
       • search_topics — fetch log

  2. scholar_agent/scholar_chroma_db/ (ChromaDB)
       • Collection: scholar_cache_vault
       • Each doc: author bio + their top papers
       • For semantic search: "find experts in X" queries

Usage:
    python scripts/build_scholar_cache.py
    python scripts/build_scholar_cache.py --papers-per-topic 75 --max-authors 300
    python scripts/build_scholar_cache.py --skip-chroma
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# macOS Python.org builds lack system certs — disable verification for S2 API
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

SQLITE_OUT = ROOT / "scholar_agent" / "scholar_cache.db"
CHROMA_OUT = ROOT / "scholar_agent" / "scholar_chroma_db"
API_KEY    = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
BASE_URL   = "https://api.semanticscholar.org/graph/v1"

# ---------------------------------------------------------------------------
# Topics — covers ALL benchmark queries (BQ01-BQ35) + eval queries (SQ01-SQ10)
# ---------------------------------------------------------------------------
SEARCH_TOPICS = [
    # ── Benchmark G1: Full-Stack Pipeline ────────────────────────────────
    ("Leiden community detection single-cell genomics",         "BQ01,BQ06,SQ-leiden"),
    ("STAR RNA-seq splice junction alignment",                  "BQ02,BQ08,BQ27"),
    ("DESeq2 differential expression RNA-seq",                  "BQ03,BQ10,BQ26"),
    ("SAM BAM pysam sequencing alignment format",               "BQ04,BQ29"),
    ("Seurat weighted nearest neighbor multimodal single-cell", "BQ05,BQ09"),
    # ── Benchmark G2: Coreference / Knowledge Transfer ────────────────────
    ("CRISPR base editing cytosine adenine",                    "BQ07,BQ30,SQ09"),
    # ── Benchmark G4: Missing Data ────────────────────────────────────────
    ("Monocle3 pseudotime trajectory inference single-cell",    "BQ16"),
    ("Salmon Kallisto RNA-seq pseudoalignment quantification",  "BQ17"),
    ("MAGIC imputation single-cell gene expression",            "BQ18"),
    ("GATK HaplotypeCaller variant calling whole genome",       "BQ20"),
    # ── Eval queries (SQ01-SQ10) ─────────────────────────────────────────
    ("single-cell RNA sequencing trajectory analysis",          "SQ01"),
    ("spatial transcriptomics gene expression imaging",         "SQ02"),
    ("chromatin accessibility ATAC-seq epigenomics",            "SQ03"),
    ("multi-omics data integration single-cell",                "SQ04"),
    ("proteomics transcriptomics integration mass spectrometry","SQ05"),
    ("deep learning neural network genomics DNA sequence",      "SQ06"),
    ("graph neural network protein structure prediction",       "SQ07"),
    ("cancer single-cell sequencing tumor microenvironment",    "SQ08"),
    ("Oxford Nanopore long-read sequencing bioinformatics",     "SQ10"),
    # ── Extra broad coverage ──────────────────────────────────────────────
    ("Scanpy single-cell analysis Python",                      "scanpy"),
    ("scRNA-seq cell clustering dimensionality reduction UMAP", "scrna-cluster"),
    ("metagenomics microbiome 16S rRNA sequencing",             "metagenomics"),
    ("AlphaFold protein structure deep learning",               "alphafold"),
    ("CRISPR screen functional genomics genome-wide",           "crispr-screen"),
    ("comparative genomics phylogenomics evolution",            "phylogenomics"),
]

# ---------------------------------------------------------------------------
# Rate-limited HTTP fetch
# ---------------------------------------------------------------------------
_LAST_REQUEST = 0.0
_MIN_INTERVAL = 0.12  # ~8 req/s with API key (staying under 10 req/s limit)


def _fetch(url: str) -> dict:
    global _LAST_REQUEST
    wait = _MIN_INTERVAL - (time.monotonic() - _LAST_REQUEST)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST = time.monotonic()

    req = urllib.request.Request(url)
    if API_KEY:
        req.add_header("x-api-key", API_KEY)

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                wait = 2 ** attempt
                print(f"      [HTTP {e.code}] retry in {wait}s…")
                time.sleep(wait)
            else:
                print(f"      [HTTP {e.code}] {url[-60:]}")
                return {}
        except Exception as e:
            print(f"      [ERR] {e}")
            return {}
    return {}


def search_papers(query: str, limit: int = 75) -> list[dict]:
    safe = urllib.parse.quote(query)
    url = (
        f"{BASE_URL}/paper/search?query={safe}&limit={limit}"
        f"&fields=paperId,title,abstract,year,citationCount,"
        f"authors,s2FieldsOfStudy,publicationTypes"
    )
    data = _fetch(url)
    return data.get("data", [])


def fetch_author_detail(author_id: str) -> dict:
    url = (
        f"{BASE_URL}/author/{author_id}"
        f"?fields=name,hIndex,citationCount,paperCount,affiliations,"
        f"papers.paperId,papers.title,papers.year,papers.citationCount,papers.abstract"
    )
    return _fetch(url)


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS authors (
    author_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    h_index         INTEGER,
    citation_count  INTEGER,
    paper_count     INTEGER,
    affiliations    TEXT,   -- JSON list of affiliation strings
    fetched_at      TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    abstract        TEXT,
    year            INTEGER,
    citation_count  INTEGER,
    fields          TEXT    -- JSON list of field-of-study strings
);

CREATE TABLE IF NOT EXISTS author_papers (
    author_id   TEXT REFERENCES authors(author_id),
    paper_id    TEXT REFERENCES papers(paper_id),
    PRIMARY KEY (author_id, paper_id)
);

CREATE TABLE IF NOT EXISTS search_topics (
    topic       TEXT PRIMARY KEY,
    query_tags  TEXT,
    papers_found INTEGER,
    authors_found INTEGER,
    fetched_at  TEXT
);

CREATE TABLE IF NOT EXISTS topic_authors (
    topic       TEXT REFERENCES search_topics(topic),
    author_id   TEXT REFERENCES authors(author_id),
    relevance   INTEGER DEFAULT 0,  -- total citations from matched papers
    PRIMARY KEY (topic, author_id)
);

-- FTS on author name + affiliation for quick PI lookup
CREATE VIRTUAL TABLE IF NOT EXISTS authors_fts USING fts5(
    author_id UNINDEXED,
    name,
    affiliations,
    content='authors',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS authors_ai AFTER INSERT ON authors BEGIN
    INSERT INTO authors_fts(rowid, author_id, name, affiliations)
    VALUES (new.rowid, new.author_id, new.name, new.affiliations);
END;

CREATE INDEX IF NOT EXISTS idx_authors_cite  ON authors(citation_count DESC);
CREATE INDEX IF NOT EXISTS idx_papers_year   ON papers(year);
CREATE INDEX IF NOT EXISTS idx_ta_author     ON topic_authors(author_id);
"""


def build_sqlite(all_authors: dict, topic_map: dict, paper_store: dict) -> None:
    print(f"\n── Building SQLite: {SQLITE_OUT} ──────────────────────────")
    SQLITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    if SQLITE_OUT.exists():
        SQLITE_OUT.unlink()

    con = sqlite3.connect(SQLITE_OUT)
    con.executescript(SCHEMA)

    now = datetime.utcnow().isoformat()

    # Insert papers
    for paper_id, p in paper_store.items():
        con.execute(
            "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?)",
            (
                paper_id,
                p.get("title") or "",
                (p.get("abstract") or "")[:3000],
                p.get("year"),
                p.get("citationCount") or 0,
                json.dumps(p.get("fields", [])),
            ),
        )

    # Insert authors + their papers
    for author_id, a in all_authors.items():
        raw_affils = a.get("affiliations") or []
        affils = [
            af.get("name", "") if isinstance(af, dict) else str(af)
            for af in raw_affils
        ]
        con.execute(
            "INSERT OR REPLACE INTO authors VALUES (?,?,?,?,?,?,?)",
            (
                author_id,
                a.get("name") or "",
                a.get("hIndex"),
                a.get("citationCount") or 0,
                a.get("paperCount") or 0,
                json.dumps(affils),
                now,
            ),
        )
        for p in (a.get("papers") or []):
            pid = p.get("paperId")
            if not pid:
                continue
            con.execute(
                "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?)",
                (
                    pid,
                    p.get("title") or "",
                    (p.get("abstract") or "")[:3000],
                    p.get("year"),
                    p.get("citationCount") or 0,
                    "[]",
                ),
            )
            con.execute(
                "INSERT OR IGNORE INTO author_papers VALUES (?,?)",
                (author_id, pid),
            )

    # Insert topics + topic_authors
    for topic, (tags, authors_rel) in topic_map.items():
        total_authors = len(authors_rel)
        con.execute(
            "INSERT OR REPLACE INTO search_topics VALUES (?,?,?,?,?)",
            (topic, tags, None, total_authors, now),
        )
        for author_id, relevance in authors_rel.items():
            con.execute(
                "INSERT OR IGNORE INTO topic_authors VALUES (?,?,?)",
                (topic, author_id, relevance),
            )

    con.commit()
    con.close()

    db_mb = SQLITE_OUT.stat().st_size / 1_048_576
    print(f"  Authors   : {len(all_authors):,}")
    print(f"  Papers    : {len(paper_store):,}")
    print(f"  DB size   : {db_mb:.1f} MB  →  {SQLITE_OUT}")


# ---------------------------------------------------------------------------
# ChromaDB build
# ---------------------------------------------------------------------------
def build_chroma(all_authors: dict, topic_map: dict) -> None:
    print(f"\n── Building ChromaDB: {CHROMA_OUT} ──────────────────────")
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("  [SKIP] chromadb not installed.")
        return

    CHROMA_OUT.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_OUT))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        client.delete_collection("scholar_cache_vault")
    except Exception:
        pass
    col = client.create_collection("scholar_cache_vault", embedding_function=ef)

    # Build topic string per author
    author_topics: dict[str, list[str]] = {}
    for topic, (_, authors_rel) in topic_map.items():
        for author_id in authors_rel:
            author_topics.setdefault(author_id, []).append(topic)

    docs, metas, ids = [], [], []
    BATCH = 100

    for author_id, a in all_authors.items():
        name  = a.get("name") or "Unknown"
        raw_affils = a.get("affiliations") or []
        affils = "; ".join(
            af.get("name", "") if isinstance(af, dict) else str(af)
            for af in raw_affils
        )
        topics = "; ".join(author_topics.get(author_id, []))

        top_papers = sorted(
            (a.get("papers") or []),
            key=lambda p: p.get("citationCount") or 0,
            reverse=True,
        )[:5]
        paper_text = "\n".join(
            f"- [{p.get('year','')}] {p.get('title','')} ({p.get('citationCount',0)} citations)"
            for p in top_papers
        )

        doc = (
            f"Researcher: {name}\n"
            f"Affiliations: {affils or 'Unknown'}\n"
            f"Research topics: {topics}\n"
            f"h-index: {a.get('hIndex') or '?'}  "
            f"Total citations: {a.get('citationCount') or 0}  "
            f"Papers: {a.get('paperCount') or 0}\n"
            f"Top papers:\n{paper_text}"
        )

        docs.append(doc)
        metas.append({
            "author_id":     author_id,
            "name":          name[:200],
            "affiliations":  affils[:300],
            "h_index":       int(a.get("hIndex") or 0),
            "citation_count": int(a.get("citationCount") or 0),
            "paper_count":   int(a.get("paperCount") or 0),
            "topics":        topics[:500],
        })
        ids.append(f"author_{author_id}")

        if len(docs) >= BATCH:
            col.add(documents=docs, metadatas=metas, ids=ids)
            docs, metas, ids = [], [], []

    if docs:
        col.add(documents=docs, metadatas=metas, ids=ids)

    print(f"  Embedded {col.count():,} authors  →  {CHROMA_OUT}")


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------
def run(papers_per_topic: int, max_authors: int, skip_chroma: bool, skip_sqlite: bool):
    if not API_KEY:
        print("[ERROR] SEMANTIC_SCHOLAR_API_KEY not set in .env")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Semantic Scholar Researcher Cache Builder            ║")
    print(f"║     Topics: {len(SEARCH_TOPICS):<3}  Papers/topic: {papers_per_topic:<3}  Max authors: {max_authors:<3}  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    all_authors: dict[str, dict] = {}    # author_id → full detail
    paper_store: dict[str, dict] = {}    # paper_id  → paper record
    topic_map:   dict[str, tuple] = {}   # topic → (tags, {author_id: relevance})

    for i, (topic, tags) in enumerate(SEARCH_TOPICS):
        print(f"\n[{i+1:02d}/{len(SEARCH_TOPICS)}] {topic}")
        papers = search_papers(topic, limit=papers_per_topic)
        print(f"  Papers found: {len(papers)}")

        # Collect author candidates from papers
        candidate_authors: dict[str, int] = {}  # author_id → total citations
        for paper in papers:
            # Store paper
            pid = paper.get("paperId")
            if pid:
                fos = [f.get("category", "") for f in (paper.get("s2FieldsOfStudy") or [])]
                paper["fields"] = fos
                paper_store[pid] = paper

            for author in (paper.get("authors") or []):
                aid = author.get("authorId")
                if not aid:
                    continue
                cite = paper.get("citationCount") or 0
                candidate_authors[aid] = candidate_authors.get(aid, 0) + cite

        # Sort by relevance, fetch top N author details
        top_candidates = sorted(
            candidate_authors.items(), key=lambda x: x[1], reverse=True
        )[:max_authors]

        topic_authors: dict[str, int] = {}

        for j, (author_id, relevance) in enumerate(top_candidates):
            if author_id in all_authors:
                # Already fetched — just record topic link
                topic_authors[author_id] = relevance
                continue

            detail = fetch_author_detail(author_id)
            if not detail or not detail.get("name"):
                continue

            all_authors[author_id] = detail
            topic_authors[author_id] = relevance

            name = detail.get("name", "?")
            hix  = detail.get("hIndex") or "?"
            cite = detail.get("citationCount") or 0
            raw_affils = detail.get("affiliations") or []
            affils = [
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in raw_affils
            ]
            inst = affils[0][:50] if affils else "?"
            print(f"  [{j+1:03d}] {name:<30} h={hix:<4} cite={cite:>8,}  {inst}")

        topic_map[topic] = (tags, topic_authors)
        print(f"  → {len(topic_authors)} authors for this topic  |  Total unique: {len(all_authors)}")

    print(f"\n── Fetch complete ──────────────────────────────────────────")
    print(f"  Unique authors : {len(all_authors):,}")
    print(f"  Papers cached  : {len(paper_store):,}")

    if not skip_sqlite:
        build_sqlite(all_authors, topic_map, paper_store)

    if not skip_chroma:
        build_chroma(all_authors, topic_map)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                        DONE                              ║")
    if not skip_sqlite:
        print(f"║  SQLite   → scholar_agent/scholar_cache.db               ║")
    if not skip_chroma:
        print(f"║  ChromaDB → scholar_agent/scholar_chroma_db/             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-per-topic", type=int, default=75,
                        help="Papers to fetch per topic (default 75)")
    parser.add_argument("--max-authors", type=int, default=15,
                        help="Top N authors to detail-fetch per topic (default 15)")
    parser.add_argument("--skip-sqlite",  action="store_true")
    parser.add_argument("--skip-chroma",  action="store_true")
    args = parser.parse_args()

    run(args.papers_per_topic, args.max_authors, args.skip_chroma, args.skip_sqlite)


if __name__ == "__main__":
    main()
