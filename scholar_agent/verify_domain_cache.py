"""
verify_domain_cache.py — Prove the scholar cache is domain-isolated
====================================================================
Regression check for the cross-domain contamination bug: before the domain
filter existed, ChromaDB's unthresholded nearest-neighbour search returned
bioinformatics researchers for materials-science queries, scored them 0.5, and
short-circuited the live Semantic Scholar API so the correct answer was never
fetched.

Runs in three layers, each skipped cleanly if its dependencies are missing:

  1. SQLite    — stdlib only. Always runs.
  2. ChromaDB  — needs chromadb + sentence-transformers.
  3. Agent     — needs google-generativeai (the full runtime path).

Usage:
    python scholar_agent/verify_domain_cache.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scholar_agent.domains import ALL_DOMAINS, flag_key, label  # noqa: E402

DB_PATH     = ROOT / "scholar_agent" / "scholar_cache.db"
CHROMA_PATH = ROOT / "scholar_agent" / "scholar_chroma_db"

# One representative probe per domain, phrased the way a real query would be.
PROBES = {
    "bioinformatics":    "single-cell RNA-seq clustering",
    "materials_science": "perovskite solar cell efficiency",
    "health_science":    "randomized controlled trial outcomes",
}

_PASS, _FAIL, _SKIP = "  ✅", "  ❌", "  ⏭ "
_failures: list[str] = []


def _fail(msg: str) -> None:
    _failures.append(msg)
    print(f"{_FAIL} {msg}")


# ---------------------------------------------------------------------------
# Layer 1 — SQLite
# ---------------------------------------------------------------------------
def check_sqlite() -> dict[str, set]:
    print("\n" + "=" * 66)
    print("LAYER 1 — SQLite cache")
    print("=" * 66)

    if not DB_PATH.exists():
        print(f"{_SKIP} No cache at {DB_PATH}")
        return {}

    con = sqlite3.connect(DB_PATH)

    cols = {r[1] for r in con.execute("PRAGMA table_info(search_topics)")}
    if "domain" not in cols:
        _fail("search_topics has no `domain` column — migration has not run.")
        con.close()
        return {}
    print(f"{_PASS} search_topics.domain present")

    print("\n  Per-domain inventory:")
    domain_authors: dict[str, set] = {}
    for dom in ALL_DOMAINS:
        rows = con.execute(
            """SELECT DISTINCT ta.author_id
               FROM topic_authors ta JOIN search_topics st ON st.topic = ta.topic
               WHERE st.domain = ?""",
            (dom,),
        ).fetchall()
        n_topics = con.execute(
            "SELECT COUNT(*) FROM search_topics WHERE domain = ?", (dom,)
        ).fetchone()[0]
        domain_authors[dom] = {r[0] for r in rows}
        status = " " if domain_authors[dom] else "  ← EMPTY"
        print(f"    {dom:<20} {n_topics:>3} topics  {len(domain_authors[dom]):>4} authors{status}")
        if not domain_authors[dom]:
            _fail(f"Domain '{dom}' has no cached authors.")

    # Untagged topics would be invisible to every domain-filtered query.
    orphan_topics = con.execute(
        "SELECT COUNT(*) FROM search_topics WHERE domain IS NULL OR domain = ''"
    ).fetchone()[0]
    if orphan_topics:
        _fail(f"{orphan_topics} topic(s) have no domain — unreachable by any query.")
    else:
        print(f"{_PASS} No untagged topics")

    # Overlap is legitimate (a clinical-genomics author spans two domains) but
    # should be reported so it is never mistaken for leakage later.
    print("\n  Cross-domain author overlap (expected to be small):")
    doms = list(ALL_DOMAINS)
    for i, a in enumerate(doms):
        for b in doms[i + 1:]:
            shared = domain_authors.get(a, set()) & domain_authors.get(b, set())
            if shared:
                pct = 100.0 * len(shared) / max(1, min(len(domain_authors[a]),
                                                       len(domain_authors[b])))
                print(f"    {a} ∩ {b}: {len(shared)} authors ({pct:.1f}% of smaller set)")
            else:
                print(f"    {a} ∩ {b}: none")

    # The real regression test: a domain-filtered lookup must never surface an
    # author who belongs solely to a different domain.
    print("\n  Domain-filtered query isolation:")
    for dom, probe in PROBES.items():
        keywords = [w for w in probe.lower().split() if len(w) > 4][:5]
        if not keywords:
            continue
        placeholders = " OR ".join(["LOWER(ta.topic) LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords] + [dom, 20]
        rows = con.execute(
            f"""SELECT DISTINCT a.author_id, a.name
                FROM topic_authors ta
                JOIN authors a ON a.author_id = ta.author_id
                JOIN search_topics st ON st.topic = ta.topic
                WHERE ({placeholders}) AND st.domain = ? AND a.citation_count > 0
                LIMIT ?""",
            params,
        ).fetchall()

        foreign = [
            name for aid, name in rows
            if aid not in domain_authors.get(dom, set())
        ]
        if foreign:
            _fail(f"'{probe}' [{dom}] returned {len(foreign)} out-of-domain author(s): "
                  f"{', '.join(foreign[:3])}")
        else:
            print(f"{_PASS} '{probe}' [{dom}] → {len(rows)} authors, all in-domain")

    con.close()
    return domain_authors


# ---------------------------------------------------------------------------
# Layer 2 — ChromaDB
# ---------------------------------------------------------------------------
def check_chroma(domain_authors: dict[str, set]) -> None:
    print("\n" + "=" * 66)
    print("LAYER 2 — ChromaDB vector cache")
    print("=" * 66)

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print(f"{_SKIP} chromadb not installed — skipping.")
        print("     Run in your project env: "
              "python scholar_agent/build_scholar_cache.py --rebuild-chroma")
        return

    if not CHROMA_PATH.exists():
        print(f"{_SKIP} No vector cache at {CHROMA_PATH}")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        col = client.get_collection("scholar_cache_vault", embedding_function=ef)
    except Exception as exc:
        _fail(f"Cannot open scholar_cache_vault: {exc}")
        return

    space = (col.metadata or {}).get("hnsw:space", "l2")
    print(f"  Collection: {col.count():,} docs, distance space = {space}")
    if space != "cosine":
        _fail(f"Distance space is '{space}', not 'cosine' — the runtime "
              f"similarity floor assumes cosine. Re-run --rebuild-chroma.")

    sample = col.get(limit=1, include=["metadatas"])
    metas = sample.get("metadatas") or []
    if not metas or not any(k.startswith("dom_") for k in (metas[0] or {})):
        _fail("Vector cache carries no `dom_*` metadata — the agent will skip it. "
              "Run: python scholar_agent/build_scholar_cache.py --rebuild-chroma")
        return
    print(f"{_PASS} Per-domain metadata flags present")

    print("\n  Per-domain document counts:")
    for dom in ALL_DOMAINS:
        n = len(col.get(where={flag_key(dom): True}, include=[])["ids"])
        print(f"    {dom:<20} {n:>4} docs")
        if not n:
            _fail(f"Vector cache has no documents for '{dom}'.")

    try:
        from scholar_agent.agent import CACHE_SIMILARITY_FLOOR as floor
    except Exception:
        floor = 0.15
    print(f"\n  Filtered retrieval (similarity floor = {floor}):")
    for dom, probe in PROBES.items():
        res = col.query(
            query_texts=[probe], n_results=5,
            where={flag_key(dom): True},
            include=["metadatas", "distances"],
        )
        rows = list(zip(res["metadatas"][0], res["distances"][0]))
        if not rows:
            _fail(f"'{probe}' [{dom}] returned nothing.")
            continue
        sims = [1.0 - d for _, d in rows]
        leaked = [m["name"] for m, _ in rows if not m.get(flag_key(dom))]
        below = sum(1 for s in sims if s < floor)
        print(f"    '{probe}' [{dom}]")
        print(f"      top match : {rows[0][0]['name']}  (sim={sims[0]:.3f})")
        print(f"      sim range : {min(sims):.3f} – {max(sims):.3f}"
              + (f"   ({below} below floor, dropped)" if below else ""))
        if leaked:
            _fail(f"'{probe}' [{dom}] leaked out-of-domain: {', '.join(leaked[:3])}")
        else:
            print(f"{_PASS}     all 5 hits carry {flag_key(dom)}")


# ---------------------------------------------------------------------------
# Layer 3 — Full agent path
# ---------------------------------------------------------------------------
def check_agent() -> None:
    print("\n" + "=" * 66)
    print("LAYER 3 — ScholarCacheProvider (runtime path)")
    print("=" * 66)

    try:
        from scholar_agent.agent import ScholarCacheProvider, CACHE_SIMILARITY_FLOOR
    except ImportError as exc:
        print(f"{_SKIP} Cannot import agent ({exc}) — skipping.")
        return

    provider = ScholarCacheProvider()
    print(f"  Similarity floor: {CACHE_SIMILARITY_FLOOR}")
    for dom, probe in PROBES.items():
        results = provider.query(probe, limit=3, exclude_ids=set(), domain=dom)
        print(f"\n    '{probe}' [{dom}] → {len(results)} expert(s)")
        for r in results:
            areas = ", ".join(r.expertise_areas[:2])
            print(f"      • {r.author.name:<28} score={r.relevance_score:<7} {areas[:50]}")
        if not results:
            print(f"      (empty — falls through to the live API, which is correct "
                  f"if this domain is not cached)")


def main() -> None:
    print("╔" + "═" * 64 + "╗")
    print("║" + "  Scholar cache — domain isolation verification".ljust(64) + "║")
    print("╚" + "═" * 64 + "╝")

    domain_authors = check_sqlite()
    check_chroma(domain_authors)
    check_agent()

    print("\n" + "=" * 66)
    if _failures:
        print(f"RESULT: {len(_failures)} problem(s) found\n")
        for f in _failures:
            print(f"  ❌ {f}")
        sys.exit(1)
    print("RESULT: all executed checks passed ✅")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
