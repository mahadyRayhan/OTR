"""
build_scholar_cache.py — Pre-fetch researchers from Semantic Scholar and cache locally
=======================================================================================
Searches Semantic Scholar for top researchers across the seed topics of one or
more research domains (see scholar_agent/domains.py). Stores results in:

  1. scholar_agent/scholar_cache.db  (SQLite)
       • authors       — name, h-index, citation count, affiliations
       • papers        — title, abstract, year, citations
       • author_papers — junction
       • search_topics — fetch log, tagged with the owning `domain`
       • topic_authors — which topics each author covers

  2. scholar_agent/scholar_chroma_db/ (ChromaDB)
       • Collection: scholar_cache_vault
       • Each doc: author bio + their top papers
       • Metadata carries `domain` (primary) and `dom_<name>` boolean flags so
         retrieval can be filtered per domain.

Domain membership lives on `search_topics.domain`, NOT on `authors`. An author
reached from both a bioinformatics topic and a health-science topic is a member
of both — which the boolean Chroma flags preserve.

The build is INCREMENTAL: running one domain merges into the existing DB and
leaves other domains untouched. Only the target domain's topic_authors rows are
refreshed.

ChromaDB is always rebuilt from SQLite (not from the in-memory fetch), so the
domain flags stay globally consistent no matter which order domains were built.

Usage:
    python scholar_agent/build_scholar_cache.py --list-domains
    python scholar_agent/build_scholar_cache.py --domain materials_science
    python scholar_agent/build_scholar_cache.py --domain health_science --skip-chroma
    python scholar_agent/build_scholar_cache.py --rebuild-chroma        # no API calls
    python scholar_agent/build_scholar_cache.py --domain all --papers-per-topic 75
"""

import argparse
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

from scholar_agent.domains import (  # noqa: E402
    ALL_DOMAINS,
    DEFAULT_DOMAIN,
    DOMAIN_TOPICS,
    flag_key,
    keywords_of,
    normalize,
    topic_relevance,
    topics_for,
)

SQLITE_OUT = ROOT / "scholar_agent" / "scholar_cache.db"
CHROMA_OUT = ROOT / "scholar_agent" / "scholar_chroma_db"
API_KEY    = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
BASE_URL   = "https://api.semanticscholar.org/graph/v1"

COLLECTION_NAME = "scholar_cache_vault"
EMBED_MODEL     = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Rate-limited HTTP fetch
# ---------------------------------------------------------------------------
_LAST_REQUEST = 0.0

# Semantic Scholar's shared authenticated pool is ~1 req/s. The original 0.12s
# (~8 req/s) drew a wall of 429s, and because a fully-retried request returned
# {} the affected researchers were dropped from the cache with no trace in the
# summary. Start conservative and self-tune upward on sustained throttling.
_MIN_INTERVAL     = 1.1
_MAX_INTERVAL     = 6.0
_THROTTLE_STEP    = 0.25
_MAX_ATTEMPTS     = 6

# Throttling that only ratchets upward would keep the whole run slow after a
# single bad patch. Ease back toward the configured base after a clean streak.
_BASE_INTERVAL    = _MIN_INTERVAL
_DECAY_STEP       = 0.10
_DECAY_AFTER      = 25          # consecutive successful requests
_clean_streak     = 0

# Visibility counters — reported per domain so silent data loss is impossible.
_STATS = {"requests": 0, "throttled": 0, "failed": 0}


def set_rate_delay(seconds: float) -> None:
    global _MIN_INTERVAL, _BASE_INTERVAL
    _MIN_INTERVAL = _BASE_INTERVAL = max(0.05, float(seconds))


def _fetch(url: str) -> dict:
    global _LAST_REQUEST, _MIN_INTERVAL, _clean_streak

    wait = _MIN_INTERVAL - (time.monotonic() - _LAST_REQUEST)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST = time.monotonic()
    _STATS["requests"] += 1

    req = urllib.request.Request(url)
    if API_KEY:
        req.add_header("x-api-key", API_KEY)

    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            _clean_streak += 1
            if _clean_streak >= _DECAY_AFTER and _MIN_INTERVAL > _BASE_INTERVAL:
                _MIN_INTERVAL = max(_BASE_INTERVAL, _MIN_INTERVAL - _DECAY_STEP)
                _clean_streak = 0
                print(f"      [rate] {_DECAY_AFTER} clean requests → easing to "
                      f"{_MIN_INTERVAL:.2f}s/req")
            return payload
        except urllib.error.HTTPError as e:
            # 408/504 are transient gateway timeouts — S2 returns them under load
            # on the author-detail endpoint. Retrying recovers the researcher;
            # giving up drops them from the cache permanently.
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < _MAX_ATTEMPTS - 1:
                _clean_streak = 0
                if e.code == 429:
                    _STATS["throttled"] += 1
                    # Back the sustained rate off, not just this one request.
                    if _MIN_INTERVAL < _MAX_INTERVAL:
                        _MIN_INTERVAL = min(_MAX_INTERVAL, _MIN_INTERVAL + _THROTTLE_STEP)
                        print(f"      [HTTP 429] throttling → {_MIN_INTERVAL:.2f}s/req")
                backoff = min(2 ** attempt, 30)
                print(f"      [HTTP {e.code}] retry {attempt + 1}/{_MAX_ATTEMPTS} in {backoff}s…")
                time.sleep(backoff)
                _LAST_REQUEST = time.monotonic()
            else:
                _STATS["failed"] += 1
                print(f"      [HTTP {e.code}] GAVE UP → {url[-70:]}")
                return {}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # Transient network failures on a ~1000-request run are expected;
            # treat them like the retryable HTTP codes above.
            if attempt < _MAX_ATTEMPTS - 1:
                backoff = min(2 ** attempt, 30)
                print(f"      [NET] {e} — retry {attempt + 1}/{_MAX_ATTEMPTS} in {backoff}s…")
                time.sleep(backoff)
                _LAST_REQUEST = time.monotonic()
            else:
                _STATS["failed"] += 1
                print(f"      [NET] GAVE UP → {e}")
                return {}
        except Exception as e:
            _STATS["failed"] += 1
            print(f"      [ERR] {e}")
            return {}
    _STATS["failed"] += 1
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
    topic        TEXT PRIMARY KEY,
    query_tags   TEXT,
    papers_found INTEGER,
    authors_found INTEGER,
    fetched_at   TEXT,
    domain       TEXT NOT NULL DEFAULT 'bioinformatics'
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

# Indexes that touch columns added by _migrate must be created after it runs,
# otherwise executescript() fails on a pre-domain database.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_st_domain ON search_topics(domain);
"""


def _migrate(con: sqlite3.Connection) -> None:
    """
    Bring a pre-domain database up to the current schema.

    The original cache had no `domain` column. Every row in it came from the
    bioinformatics topic list, so backfilling that value is exact, not a guess.
    """
    cols = {r[1] for r in con.execute("PRAGMA table_info(search_topics)")}
    if not cols or "domain" in cols:
        return

    print("  [migrate] Adding search_topics.domain — backfilling existing rows "
          f"as '{DEFAULT_DOMAIN}'…")
    con.execute(
        "ALTER TABLE search_topics ADD COLUMN domain TEXT NOT NULL "
        f"DEFAULT '{DEFAULT_DOMAIN}'"
    )
    con.execute("UPDATE search_topics SET domain = ? WHERE domain IS NULL",
                (DEFAULT_DOMAIN,))
    n = con.execute("SELECT COUNT(*) FROM search_topics").fetchone()[0]
    con.commit()
    print(f"  [migrate] {n} existing topics tagged '{DEFAULT_DOMAIN}'.")


def build_sqlite(domain: str, all_authors: dict, topic_map: dict, paper_store: dict) -> None:
    """Merge a single domain's fetch into the cache, leaving other domains intact."""
    print(f"\n── Updating SQLite [{domain}]: {SQLITE_OUT} ──────────────")
    SQLITE_OUT.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(SQLITE_OUT)
    con.executescript(SCHEMA)
    _migrate(con)
    con.executescript(INDEXES)

    now = datetime.utcnow().isoformat()

    # Clear only THIS domain's topic links so a re-run is idempotent per domain.
    old_topics = [r[0] for r in con.execute(
        "SELECT topic FROM search_topics WHERE domain = ?", (domain,))]
    if old_topics:
        qs = ",".join("?" * len(old_topics))
        con.execute(f"DELETE FROM topic_authors WHERE topic IN ({qs})", old_topics)
        con.execute(f"DELETE FROM search_topics WHERE topic IN ({qs})", old_topics)
        print(f"  Cleared {len(old_topics)} previous '{domain}' topics.")

    # Insert papers harvested from search results
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

    # Insert topics + topic_authors, tagged with the domain
    for topic, (tags, authors_rel) in topic_map.items():
        con.execute(
            "INSERT OR REPLACE INTO search_topics VALUES (?,?,?,?,?,?)",
            (topic, tags, None, len(authors_rel), now, domain),
        )
        for author_id, relevance in authors_rel.items():
            con.execute(
                "INSERT OR IGNORE INTO topic_authors VALUES (?,?,?)",
                (topic, author_id, relevance),
            )

    # INSERT OR REPLACE bypasses the AFTER INSERT trigger's assumptions —
    # rebuild the external-content FTS index so it matches `authors` exactly.
    con.execute("INSERT INTO authors_fts(authors_fts) VALUES('rebuild')")

    con.commit()

    total_authors = con.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    total_papers  = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    per_domain = con.execute(
        """SELECT st.domain, COUNT(DISTINCT ta.author_id)
           FROM search_topics st
           LEFT JOIN topic_authors ta ON ta.topic = st.topic
           GROUP BY st.domain ORDER BY st.domain"""
    ).fetchall()
    con.close()

    db_mb = SQLITE_OUT.stat().st_size / 1_048_576
    print(f"  New this run : {len(all_authors):,} authors, {len(paper_store):,} papers")
    print(f"  DB totals    : {total_authors:,} authors, {total_papers:,} papers "
          f"({db_mb:.1f} MB)")
    for dom, cnt in per_domain:
        print(f"    • {dom:<20} {cnt:>4} authors")


# ---------------------------------------------------------------------------
# Paper pruning
# ---------------------------------------------------------------------------
DEFAULT_MAX_PAPERS_PER_AUTHOR = 10


def prune_papers(max_per_author: int = DEFAULT_MAX_PAPERS_PER_AUTHOR,
                 verbose: bool = True) -> None:
    """
    Trim every author down to their `max_per_author` most relevant papers.

    S2's author endpoint returns up to 500 papers, which bloats the cache and
    makes domains collected at different times incomparable. The agent only
    ever displays 5 per author, so the tail is dead weight.

    Selection is by overlap with the topics that surfaced the author, with
    citations as the tie-break — NOT by citations alone. A materials scientist's
    most-cited papers are their career-best work, which is frequently unrelated
    to the topic they were found under; pruning on citations would delete
    exactly the topically relevant papers the agent needs.
    """
    if not SQLITE_OUT.exists():
        print("  [prune] No SQLite cache.")
        return

    con = sqlite3.connect(SQLITE_OUT)
    before_ap = con.execute("SELECT COUNT(*) FROM author_papers").fetchone()[0]
    before_p  = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    # author_id → keywords drawn from every topic that found them
    author_kw: dict[str, set] = {}
    for aid, topic in con.execute(
        "SELECT ta.author_id, ta.topic FROM topic_authors ta"
    ):
        author_kw.setdefault(aid, set()).update(keywords_of(topic))

    keep: list[tuple[str, str]] = []
    rows_by_author: dict[str, list] = {}
    for aid, pid, title, abstract, cites in con.execute(
        """SELECT ap.author_id, p.paper_id, p.title, p.abstract, p.citation_count
           FROM author_papers ap JOIN papers p ON p.paper_id = ap.paper_id"""
    ):
        rows_by_author.setdefault(aid, []).append((pid, title, abstract, cites))

    for aid, papers in rows_by_author.items():
        if len(papers) <= max_per_author:
            keep.extend((aid, p[0]) for p in papers)
            continue
        kws = author_kw.get(aid, set())

        def _score(p, _kws=kws):
            return topic_relevance(p[1], p[2], _kws, p[3])

        for p in sorted(papers, key=_score, reverse=True)[:max_per_author]:
            keep.append((aid, p[0]))

    con.execute("CREATE TEMP TABLE _keep (author_id TEXT, paper_id TEXT)")
    con.executemany("INSERT INTO _keep VALUES (?,?)", keep)
    con.execute("CREATE INDEX _keep_idx ON _keep(author_id, paper_id)")

    con.execute(
        """DELETE FROM author_papers
           WHERE NOT EXISTS (SELECT 1 FROM _keep k
                             WHERE k.author_id = author_papers.author_id
                               AND k.paper_id  = author_papers.paper_id)"""
    )
    # Papers no longer reachable through any author are unreadable by the agent,
    # which only ever queries `papers` through the author_papers join.
    con.execute(
        """DELETE FROM papers
           WHERE NOT EXISTS (SELECT 1 FROM author_papers ap
                             WHERE ap.paper_id = papers.paper_id)"""
    )
    con.commit()

    after_ap = con.execute("SELECT COUNT(*) FROM author_papers").fetchone()[0]
    after_p  = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    con.close()

    size_before = SQLITE_OUT.stat().st_size / 1_048_576
    con = sqlite3.connect(SQLITE_OUT)
    con.execute("VACUUM")
    con.close()
    size_after = SQLITE_OUT.stat().st_size / 1_048_576

    if verbose:
        print(f"\n── Pruned to max {max_per_author} papers/author ──────────────")
        print(f"  author_papers : {before_ap:,} → {after_ap:,}")
        print(f"  papers        : {before_p:,} → {after_p:,}")
        print(f"  DB size       : {size_before:.1f} MB → {size_after:.1f} MB")


# ---------------------------------------------------------------------------
# ChromaDB build — always sourced from SQLite for global domain consistency
# ---------------------------------------------------------------------------
def build_chroma_from_sqlite() -> None:
    print(f"\n── Rebuilding ChromaDB from SQLite: {CHROMA_OUT} ─────────")
    if not SQLITE_OUT.exists():
        print("  [SKIP] No SQLite cache to build from.")
        return
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("  [SKIP] chromadb not installed. Re-run later with --rebuild-chroma "
              "in an environment that has chromadb + sentence-transformers.")
        return

    con = sqlite3.connect(SQLITE_OUT)
    con.row_factory = sqlite3.Row

    authors = con.execute(
        """SELECT author_id, name, h_index, citation_count, paper_count, affiliations
           FROM authors"""
    ).fetchall()

    # author_id → {domain: True}  and  author_id → [topic, …]
    author_domains: dict[str, set] = {}
    author_topics:  dict[str, list] = {}
    for row in con.execute(
        """SELECT ta.author_id, ta.topic, st.domain
           FROM topic_authors ta JOIN search_topics st ON st.topic = ta.topic"""
    ):
        author_domains.setdefault(row[0], set()).add(row[2])
        author_topics.setdefault(row[0], []).append(row[1])

    # author_id → top 5 papers by citation
    author_papers: dict[str, list] = {}
    for row in con.execute(
        """SELECT ap.author_id, p.title, p.year, p.citation_count
           FROM author_papers ap JOIN papers p ON p.paper_id = ap.paper_id
           ORDER BY p.citation_count DESC"""
    ):
        bucket = author_papers.setdefault(row[0], [])
        if len(bucket) < 5:
            bucket.append((row[1], row[2], row[3]))
    con.close()

    CHROMA_OUT.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_OUT))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Cosine space so the runtime similarity floor is well-defined.
    col = client.create_collection(
        COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs, metas, ids = [], [], []
    BATCH = 100
    skipped = 0

    for a in authors:
        author_id = a["author_id"]
        doms = author_domains.get(author_id, set())
        if not doms:
            # Orphan author with no topic link — cannot be domain-filtered, skip.
            skipped += 1
            continue

        name   = a["name"] or "Unknown"
        try:
            affils = "; ".join(json.loads(a["affiliations"] or "[]"))
        except Exception:
            affils = ""
        topics = "; ".join(author_topics.get(author_id, []))

        paper_text = "\n".join(
            f"- [{yr or ''}] {title} ({cites or 0} citations)"
            for title, yr, cites in author_papers.get(author_id, [])
        )

        docs.append(
            f"Researcher: {name}\n"
            f"Affiliations: {affils or 'Unknown'}\n"
            f"Research topics: {topics}\n"
            f"h-index: {a['h_index'] or '?'}  "
            f"Total citations: {a['citation_count'] or 0}  "
            f"Papers: {a['paper_count'] or 0}\n"
            f"Top papers:\n{paper_text}"
        )

        meta = {
            "author_id":      author_id,
            "name":           name[:200],
            "affiliations":   affils[:300],
            "h_index":        int(a["h_index"] or 0),
            "citation_count": int(a["citation_count"] or 0),
            "paper_count":    int(a["paper_count"] or 0),
            "topics":         topics[:500],
            # Primary domain (deterministic) for display, plus one boolean per
            # domain so multi-domain authors survive a `where` filter.
            "domain":         sorted(doms)[0],
            "domains":        "; ".join(sorted(doms)),
        }
        for d in ALL_DOMAINS:
            meta[flag_key(d)] = d in doms
        metas.append(meta)
        ids.append(f"author_{author_id}")

        if len(docs) >= BATCH:
            col.add(documents=docs, metadatas=metas, ids=ids)
            docs, metas, ids = [], [], []

    if docs:
        col.add(documents=docs, metadatas=metas, ids=ids)

    print(f"  Embedded {col.count():,} authors  →  {CHROMA_OUT}")
    if skipped:
        print(f"  Skipped {skipped} authors with no topic link (not domain-attributable).")
    for d in ALL_DOMAINS:
        n = len(col.get(where={flag_key(d): True}, include=[])["ids"])
        print(f"    • {d:<20} {n:>4} authors")


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------
def fetch_domain(domain: str, papers_per_topic: int, max_authors: int) -> tuple[dict, dict, dict]:
    topics = topics_for(domain)
    all_authors: dict[str, dict] = {}    # author_id → full detail
    paper_store: dict[str, dict] = {}    # paper_id  → paper record
    topic_map:   dict[str, tuple] = {}   # topic → (tags, {author_id: relevance})

    print(f"\n{'═' * 62}")
    print(f"  DOMAIN: {domain}   ({len(topics)} topics)")
    print(f"{'═' * 62}")

    for i, (topic, tags) in enumerate(topics):
        print(f"\n[{i+1:02d}/{len(topics)}] {topic}")
        papers = search_papers(topic, limit=papers_per_topic)
        print(f"  Papers found: {len(papers)}")

        candidate_authors: dict[str, int] = {}  # author_id → total citations
        for paper in papers:
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

        top_candidates = sorted(
            candidate_authors.items(), key=lambda x: x[1], reverse=True
        )[:max_authors]

        topic_authors: dict[str, int] = {}

        for j, (author_id, relevance) in enumerate(top_candidates):
            if author_id in all_authors:
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
        print(f"  → {len(topic_authors)} authors for this topic  |  "
              f"Total unique in domain: {len(all_authors)}")

    return all_authors, topic_map, paper_store


def run(domains: list[str], papers_per_topic: int, max_authors: int,
        skip_chroma: bool, skip_sqlite: bool,
        max_papers_per_author: int = DEFAULT_MAX_PAPERS_PER_AUTHOR) -> None:
    if not API_KEY:
        print("[ERROR] SEMANTIC_SCHOLAR_API_KEY not set in .env")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Semantic Scholar Researcher Cache Builder            ║")
    print(f"║     Domains: {', '.join(domains):<43}║")
    print(f"║     Papers/topic: {papers_per_topic:<4} Max authors/topic: {max_authors:<15}║")
    print("╚══════════════════════════════════════════════════════════╝")

    for domain in domains:
        all_authors, topic_map, paper_store = fetch_domain(
            domain, papers_per_topic, max_authors
        )
        print(f"\n── Fetch complete [{domain}] ───────────────────────────────")
        print(f"  Unique authors : {len(all_authors):,}")
        print(f"  Papers cached  : {len(paper_store):,}")
        print(f"  API requests   : {_STATS['requests']:,} "
              f"({_STATS['throttled']} throttled, {_STATS['failed']} dropped)")
        if _STATS["failed"]:
            print(f"  [WARN] {_STATS['failed']} request(s) failed permanently — "
                  f"those researchers are missing from this domain.")

        if not skip_sqlite:
            build_sqlite(domain, all_authors, topic_map, paper_store)

    if not skip_sqlite and max_papers_per_author > 0:
        prune_papers(max_papers_per_author)

    if not skip_chroma:
        build_chroma_from_sqlite()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                        DONE                              ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build the per-domain Semantic Scholar researcher cache."
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN,
                        help=f"Domain to fetch, or 'all'. Options: {', '.join(ALL_DOMAINS)}")
    parser.add_argument("--papers-per-topic", type=int, default=75,
                        help="Papers to fetch per topic (default 75)")
    parser.add_argument("--max-authors", type=int, default=15,
                        help="Top N authors to detail-fetch per topic (default 15)")
    parser.add_argument("--rate-delay", type=float, default=_MIN_INTERVAL,
                        help=f"Seconds between API requests (default {_MIN_INTERVAL}). "
                             "Auto-increases on sustained 429s.")
    parser.add_argument("--skip-sqlite", action="store_true")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--rebuild-chroma", action="store_true",
                        help="Rebuild ChromaDB from the existing SQLite cache. "
                             "Makes no API calls.")
    parser.add_argument("--max-papers-per-author", type=int,
                        default=DEFAULT_MAX_PAPERS_PER_AUTHOR,
                        help=f"Cap cached papers per author "
                             f"(default {DEFAULT_MAX_PAPERS_PER_AUTHOR}; 0 disables).")
    parser.add_argument("--prune-only", action="store_true",
                        help="Prune the existing cache to --max-papers-per-author "
                             "and exit. Makes no API calls.")
    parser.add_argument("--list-domains", action="store_true")
    args = parser.parse_args()

    if args.list_domains:
        print("\nConfigured domains:\n")
        for d in ALL_DOMAINS:
            print(f"  {d:<20} {len(DOMAIN_TOPICS.get(d, [])):>3} seed topics")
        print()
        return

    if args.prune_only:
        prune_papers(args.max_papers_per_author)
        return

    if args.rebuild_chroma:
        build_chroma_from_sqlite()
        return

    set_rate_delay(args.rate_delay)

    if args.domain.lower() == "all":
        domains = list(ALL_DOMAINS)
    else:
        requested = normalize(args.domain)
        if requested != args.domain.strip().lower().replace(" ", "_").replace("-", "_"):
            print(f"[ERROR] Unknown domain '{args.domain}'. "
                  f"Options: {', '.join(ALL_DOMAINS)}, or 'all'.")
            sys.exit(1)
        domains = [requested]

    run(domains, args.papers_per_topic, args.max_authors,
        args.skip_chroma, args.skip_sqlite, args.max_papers_per_author)


if __name__ == "__main__":
    main()
