import os
import json
import asyncio
import contextvars
import time
import math
import re
import sqlite3
import ssl
import urllib.request
import urllib.parse
import gemini_client as genai
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from scholar_agent import domains as _domains
from scholar_agent.domains import DEFAULT_DOMAIN
from scholar_agent.intent import is_paper_lookup
from orchestrator import llm_cache, trace

# Minimum cosine similarity for a cached author to be accepted as an expert.
# Without this floor, ChromaDB's nearest-neighbour search always returns its
# top-k regardless of how unrelated they are — a materials-science query would
# come back with confidently-scored bioinformatics researchers.
#
# Set below the lowest similarity observed across the bioinformatics benchmark
# probes (0.197, on "Kallisto pseudoalignment") so the existing bio results are
# preserved. The per-domain `where` filter is the primary defence against
# cross-domain contamination; this floor is a secondary guard against weak
# in-domain matches being presented as confident hits.
CACHE_SIMILARITY_FLOOR = 0.15

# Domains whose cached papers keep the original citation-only ordering.
#
# The bioinformatics cache was built by an earlier version of the builder that
# stored at most ~20 papers per author (most have exactly 5). Re-ranking those
# by topic overlap would change scholar output for already-published benchmark
# runs, so that domain stays on legacy behaviour.
#
# Domains collected against the current API hold up to S2's 500-paper limit,
# where ordering by citations surfaces an author's career-best work rather than
# anything related to the question — so they get topical ranking.
#
# Drop a domain from this set to opt it into topical ranking.
LEGACY_CITATION_RANKING_DOMAINS = frozenset({_domains.BIOINFORMATICS})

# Per-call token accumulator — set in call_scholar_agent, read by internal LLM helpers
_scholar_token_acc: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_scholar_token_acc", default=None
)

def _scholar_add_tokens(response) -> None:
    acc = _scholar_token_acc.get()
    if acc is None:
        return
    _usage = getattr(response, "usage_metadata", None)
    acc["input"]  += getattr(_usage, "prompt_token_count", 0) or 0
    acc["output"] += getattr(_usage, "candidates_token_count", 0) or 0

# macOS Python.org builds lack system CA bundle — disable cert check for S2 API
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
@dataclass
class AuthorBrief:
    id: Optional[str]
    name: str

@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    authors: List[AuthorBrief]
    publication_types: List[str]
    fields_of_study: List[str]
    doi: str = ""
    venue: str = ""

@dataclass
class Author:
    id: str
    name: str
    paper_count: int
    citation_count: int
    affiliations: Optional[List[str]] = None

@dataclass
class ExpertResult:
    author: Author
    relevance_score: float
    matched_papers: List[Paper]
    expertise_areas: List[str]

# ---------------------------------------------------------
# RATE LIMITER
# ---------------------------------------------------------
class TokenBucketRateLimiter:
    def __init__(self, rate_per_second: float, capacity: int):
        self.rate = rate_per_second
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    async def acquire(self):
        while True:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)
            self.last_update = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            await asyncio.sleep(0.05)

# ---------------------------------------------------------
# REAL SEMANTIC SCHOLAR API PROVIDER
# ---------------------------------------------------------
class RealSemanticScholarProvider:
    def __init__(self):
        self.limiter = TokenBucketRateLimiter(rate_per_second=1.0, capacity=2)

    async def _fetch(self, url: str, max_retries: int = 3) -> dict:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        for attempt in range(max_retries):
            await self.limiter.acquire()
            req = urllib.request.Request(url)
            if api_key:
                req.add_header("x-api-key", api_key)

            try:
                def do_request():
                    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=20) as response:
                        return json.loads(response.read().decode('utf-8'))
                return await asyncio.to_thread(do_request)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"[API Warning] HTTP {e.code} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    print(f"[API Warning] Failed to fetch data: {e}")
                    return {}
            except Exception as e:
                print(f"[API Warning] Failed to fetch data: {e}")
                return {}
        return {}

    async def search_papers(self, query: str, limit: int = 30) -> List[Paper]:
        safe_query = urllib.parse.quote(query)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={safe_query}&limit={limit}"
            f"&fields=title,abstract,year,citationCount,authors,publicationTypes,s2FieldsOfStudy,externalIds,venue"
        )
        data = await self._fetch(url)
        papers = []
        for item in data.get("data", []):
            fos_raw = item.get("s2FieldsOfStudy") or []
            fos = [f.get("category") for f in fos_raw if "category" in f]
            pub_types = item.get("publicationTypes") or []
            authors = [
                AuthorBrief(id=a.get("authorId"), name=a.get("name"))
                for a in item.get("authors", [])
            ]
            papers.append(Paper(
                id=item.get("paperId", ""),
                title=item.get("title", ""),
                abstract=item.get("abstract", "") or "",
                year=item.get("year") or 2024,
                citation_count=item.get("citationCount") or 0,
                authors=authors,
                publication_types=pub_types,
                fields_of_study=fos,
                doi=(item.get("externalIds") or {}).get("DOI") or "",
                venue=item.get("venue") or "",
            ))
        return papers

    async def search_authors(self, name: str) -> List[AuthorBrief]:
        safe_name = urllib.parse.quote(name)
        url = f"https://api.semanticscholar.org/graph/v1/author/search?query={safe_name}&limit=5"
        data = await self._fetch(url)
        return [
            AuthorBrief(id=a.get("authorId"), name=a.get("name"))
            for a in data.get("data", [])
        ]

# ---------------------------------------------------------
# LLM UTILITIES
# ---------------------------------------------------------
def get_llm_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        return None
    genai.configure(api_key=api_key)
    # return genai.GenerativeModel('gemini-flash-latest')
    return genai.GenerativeModel('gemini-3.1-flash-lite-preview')

async def extract_search_topic(query: str, model, domain: str = DEFAULT_DOMAIN) -> str:
    # Fallback regex when no LLM is available: strip only the known
    # meta-instruction prefixes, keep everything else.
    if not model:
        cleaned = re.sub(
            r'(?i)('
            r'find \d+ (?:human )?experts? (?:who have published (?:recent )?papers? on)?|'
            r'find me some research papers on|'
            r'i am looking for a collaborator to help with a|'
            r'retrieve(?: and list)?(?: the)?(?:\s+exact)?(?: parameter values from)?|'
            r'write a (?:shell|python|bash)? ?script(?: using)?|'
            r'tell me about|what is'
            r')',
            '',
            query,
        ).strip().strip("'\"")
        return cleaned
 
    hints = _domains.DOMAIN_SEARCH_HINTS.get(
        _domains.normalize(domain),
        _domains.DOMAIN_SEARCH_HINTS[DEFAULT_DOMAIN],
    )
    prompt = (
        "Convert this input into a concise academic search query (3–8 words).\n\n"
        f"The query is from the {_domains.label(domain)} research domain.\n\n"
        "KEEP — these are search signal:\n"
        f"{hints['keep']}\n"
        "  • Named authors or papers                 (e.g. 'Bray et al 2016', 'Pachter')\n\n"
        "REMOVE — these are meta-instructions, not search terms:\n"
        "  • Task verbs: 'find experts', 'retrieve papers', 'write a script',\n"
        "                'explain', 'list', 'design', 'identify', 'extract', 'retrieve'\n"
        "  • Count phrases: '2 human experts', 'one researcher'\n"
        "  • Explicitness markers: 'specifically', 'explicitly', 'exactly'\n"
        "  • Dependency references: '[result of goal 1]', '[output of task 2]'\n"
        "  • Retrieval-task object nouns: when a phrase says WHAT TO RETRIEVE (not the research field).\n"
        "    Pattern: 'retrieve/extract/get [NOUN] from [source]' → [NOUN] is a task instruction, not a domain.\n"
        "    Strip [NOUN], keep the named source/tool.\n"
        "    Examples:\n"
        f"{hints['examples']}\n\n"
        "Output ONLY the search query — no punctuation, no quotes, no explanation.\n\n"
        f"Input: '{query}'\n"
        "Search query:"
    )
    try:
        res = await asyncio.to_thread(model.generate_content, prompt)
        _scholar_add_tokens(res)
        return res.text.strip().lower()
    except Exception:
        return query

# ---------------------------------------------------------
# FIX 3 & 4: ACTIVE SELF-REFLECTION JUDGE
#
# Changes vs original:
#   - Judge prompt now explicitly receives requested_count vs found_count.
#   - Returns a parsed dict {score: int, reason: str, retry_query: str|None}
#     so the caller can act on the score — not just print it.
#   - retry_query: if relevance is low, the LLM suggests a better search term.
# ---------------------------------------------------------
async def scholar_llm_judge(
    query: str,
    agent_output: str,
    model,
    requested_count: int,
    found_count: int,
) -> Dict[str, Any]:
    """
    Active self-reflection judge.
    Returns a dict so callers can branch on the score:
        {
            "score": int (0-10),
            "reason": str,
            "retry_query": str | None,   # non-None when LLM thinks a requery would help
            "raw": str,                  # original LLM response
        }
    """
    if not model:
        return {
            "score": 8,
            "reason": "Mock judge — no LLM configured.",
            "retry_query": None,
            "raw": "Mock Judge: 8/10.",
        }
    try:
        prompt = f"""
You are an Academic Advisor evaluating an AI Scholar Search Agent.

User's Research Query: "{query}"
Experts requested: {requested_count}
Experts found: {found_count}
Agent's Output:
{agent_output}

Evaluate two things:
1. RELEVANCE — are the returned experts and their keywords genuinely related to the query?
2. COMPLETENESS — were all {requested_count} requested experts found?

Respond in this EXACT format (no extra text):
Score: X/10
Reason: [one sentence]
RetryQuery: [a better search string if score < 7 AND found < requested, otherwise write NONE]
"""
        response = await asyncio.to_thread(model.generate_content, prompt)
        _scholar_add_tokens(response)
        raw = response.text.strip()

        score = 5  # safe default
        reason = raw
        retry_query = None

        score_match = re.search(r'Score:\s*(\d+)/10', raw)
        if score_match:
            score = int(score_match.group(1))

        reason_match = re.search(r'Reason:\s*(.+)', raw)
        if reason_match:
            reason = reason_match.group(1).strip()

        retry_match = re.search(r'RetryQuery:\s*(.+)', raw)
        if retry_match:
            val = retry_match.group(1).strip()
            retry_query = None if val.upper() == "NONE" else val

        return {"score": score, "reason": reason, "retry_query": retry_query, "raw": raw}

    except Exception as e:
        return {
            "score": 5,
            "reason": f"Judge error: {e}",
            "retry_query": None,
            "raw": str(e),
        }

# ---------------------------------------------------------
# LOCAL CACHE PROVIDER (SQLite + ChromaDB fallback)
# ---------------------------------------------------------
_AGENT_DIR = Path(__file__).resolve().parent

class ScholarCacheProvider:
    """Queries the pre-built local SQLite + ChromaDB cache as a fallback."""

    def __init__(self):
        self.db_path    = _AGENT_DIR / "scholar_cache.db"
        self.chroma_path = _AGENT_DIR / "scholar_chroma_db"
        self._chroma_col = None
        self._chroma_space = "cosine"
        self._chroma_domain_ready: Optional[bool] = None

    def _get_chroma_col(self):
        if self._chroma_col is not None:
            return self._chroma_col
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._chroma_col = client.get_collection(
                "scholar_cache_vault", embedding_function=ef
            )
            meta = self._chroma_col.metadata or {}
            self._chroma_space = meta.get("hnsw:space", "l2")
        except Exception as exc:
            print(f"   [Cache] ChromaDB unavailable: {exc}")
            self._chroma_col = None
        return self._chroma_col

    def _has_domain_flags(self, col) -> bool:
        """
        True once the vector cache has been rebuilt with per-domain metadata.

        A pre-migration collection carries no `dom_*` keys, so a domain filter
        would silently match nothing. Rather than fall back to an unfiltered
        search — which is exactly the cross-domain contamination this filter
        exists to prevent — we disable the vector cache and let SQLite plus the
        live API answer instead.
        """
        if self._chroma_domain_ready is not None:
            return self._chroma_domain_ready
        try:
            sample = col.get(limit=1, include=["metadatas"])
            metas = sample.get("metadatas") or []
            ready = bool(metas) and any(
                k.startswith("dom_") for k in (metas[0] or {})
            )
        except Exception:
            ready = False
        if not ready:
            print("   [Cache] Vector cache has no domain metadata — skipping it. "
                  "Run: python scholar_agent/build_scholar_cache.py --rebuild-chroma")
        self._chroma_domain_ready = ready
        return ready

    def _to_similarity(self, distance: float) -> float:
        """Convert a ChromaDB distance to cosine similarity in [0, 1]."""
        if self._chroma_space == "cosine":
            sim = 1.0 - float(distance)
        elif self._chroma_space == "ip":
            sim = float(distance)
        else:  # (squared) L2 over normalised embeddings: d^2 = 2 - 2cos
            sim = 1.0 - (float(distance) / 2.0)
        return max(0.0, min(1.0, sim))

    def _papers_for_author(self, con: sqlite3.Connection, author_id: str,
                           topic: str = "", domain: str = DEFAULT_DOMAIN) -> List[Paper]:
        """
        Return this author's 5 most relevant papers for `topic`.

        Ranking by citations alone was adequate when every author had only ~5
        cached papers. Domains collected against the current API store up to
        500 per author, where "most cited" surfaces an author's career-best
        work rather than anything to do with the question asked. Rank by topic
        overlap first, then use citations to break ties.
        """
        rows = con.execute(
            """SELECT p.paper_id, p.title, p.abstract, p.year,
                      p.citation_count, p.fields
               FROM author_papers ap
               JOIN papers p ON p.paper_id = ap.paper_id
               WHERE ap.author_id = ?
               ORDER BY p.citation_count DESC
               LIMIT 200""",
            (author_id,),
        ).fetchall()

        keywords = _domains.keywords_of(topic)
        legacy = _domains.normalize(domain) in LEGACY_CITATION_RANKING_DOMAINS
        if keywords and len(rows) > 5 and not legacy:
            rows = sorted(
                rows,
                key=lambda r: _domains.topic_relevance(r[1], r[2], keywords, r[4]),
                reverse=True,
            )
        rows = rows[:5]
        papers = []
        for r in rows:
            fos = json.loads(r[5] or "[]") if r[5] else []
            papers.append(Paper(
                id=r[0], title=r[1] or "", abstract=r[2] or "",
                year=r[3] or 0, citation_count=r[4] or 0,
                authors=[], publication_types=[], fields_of_study=fos,
            ))
        return papers

    def query_sqlite(self, topic: str, limit: int, exclude_ids: set,
                     domain: str = DEFAULT_DOMAIN) -> List[ExpertResult]:
        if not self.db_path.exists():
            return []
        domain = _domains.normalize(domain)
        try:
            con = sqlite3.connect(self.db_path)
            keywords = [w for w in topic.lower().split() if len(w) > 4][:5]
            if not keywords:
                con.close()
                return []

            # Match authors whose cached topics contain any of the keywords,
            # restricted to topics belonging to the active domain.
            placeholders = " OR ".join(["LOWER(ta.topic) LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords] + [domain, limit * 4]
            rows = con.execute(
                f"""SELECT a.author_id, a.name, a.citation_count, a.paper_count,
                           SUM(ta.relevance) AS total_rel
                    FROM topic_authors ta
                    JOIN authors a ON a.author_id = ta.author_id
                    JOIN search_topics st ON st.topic = ta.topic
                    WHERE ({placeholders})
                      AND st.domain = ?
                      AND a.citation_count > 0
                    GROUP BY a.author_id
                    ORDER BY total_rel DESC, a.citation_count DESC
                    LIMIT ?""",
                params,
            ).fetchall()

            results: List[ExpertResult] = []
            for aid, name, cites, paper_count, relevance in rows:
                if aid in exclude_ids or len(results) >= limit:
                    continue
                papers = self._papers_for_author(con, aid, topic, domain)
                kws = list({f for p in papers for f in p.fields_of_study})[:4]
                if not kws:
                    # Fall back to associated topic strings for meaningful expertise labels
                    topic_rows = con.execute(
                        """SELECT ta.topic FROM topic_authors ta
                           JOIN search_topics st ON st.topic = ta.topic
                           WHERE ta.author_id = ? AND st.domain = ?
                           ORDER BY ta.relevance DESC LIMIT 4""",
                        (aid, domain),
                    ).fetchall()
                    kws = [r[0] for r in topic_rows] or [_domains.label(domain)]
                score = min(math.log1p(relevance or 0) / 20.0, 1.0)
                results.append(ExpertResult(
                    author=Author(
                        id=aid, name=name,
                        paper_count=paper_count or 0,
                        citation_count=cites or 0,
                    ),
                    relevance_score=round(score, 4),
                    matched_papers=papers,
                    expertise_areas=kws,
                ))
            con.close()
            return results
        except Exception as exc:
            print(f"   [Cache] SQLite query failed: {exc}")
            return []

    def query_chroma(self, topic: str, limit: int, exclude_ids: set,
                     domain: str = DEFAULT_DOMAIN) -> List[ExpertResult]:
        col = self._get_chroma_col()
        if col is None:
            return []
        if not self._has_domain_flags(col):
            return []
        domain = _domains.normalize(domain)
        try:
            # Extra headroom: the similarity floor below discards weak hits, so
            # a limit*3 window could come back short after filtering.
            n_ask = min(max(limit * 5, 10), col.count())
            if n_ask == 0:
                return []
            result = col.query(
                query_texts=[topic],
                n_results=n_ask,
                where={_domains.flag_key(domain): True},
                include=["metadatas", "distances"],
            )
            metas = result.get("metadatas", [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            results: List[ExpertResult] = []
            dropped = 0
            for i, meta in enumerate(metas):
                if len(results) >= limit:
                    break
                aid = meta.get("author_id", "")
                if aid in exclude_ids:
                    continue
                # Reject weak nearest-neighbour matches rather than presenting
                # them as confident hits.
                sim = self._to_similarity(dists[i]) if i < len(dists) else 0.0
                if sim < CACHE_SIMILARITY_FLOOR:
                    dropped += 1
                    continue
                name        = meta.get("name", "Unknown")
                cites       = meta.get("citation_count", 0)
                paper_count = meta.get("paper_count", 0)
                affil_str   = meta.get("affiliations", "") or ""
                topic_str   = meta.get("topics", "") or ""
                kws = [t.strip() for t in topic_str.split(";") if t.strip()][:4]
                if not kws:
                    kws = [_domains.label(domain)]
                results.append(ExpertResult(
                    author=Author(
                        id=aid, name=name,
                        paper_count=paper_count,
                        citation_count=cites,
                        affiliations=[affil_str] if affil_str else None,
                    ),
                    relevance_score=round(sim, 4),
                    matched_papers=[],
                    expertise_areas=kws,
                ))
            if dropped:
                print(f"   [Cache] Dropped {dropped} vector hit(s) below "
                      f"similarity floor {CACHE_SIMILARITY_FLOOR}.")

            # Enrich with paper data from SQLite cache
            if results and self.db_path.exists():
                try:
                    con = sqlite3.connect(self.db_path)
                    for r in results:
                        papers = self._papers_for_author(con, r.author.id)
                        if papers:
                            r.matched_papers = papers
                    con.close()
                except Exception:
                    pass
            return results
        except Exception as exc:
            print(f"   [Cache] ChromaDB query failed: {exc}")
            return []

    def query(self, topic: str, limit: int, exclude_ids: set,
              domain: str = DEFAULT_DOMAIN) -> List[ExpertResult]:
        if limit <= 0:
            return []
        # ChromaDB first — semantic similarity gives better topical relevance
        results = self.query_chroma(topic, limit, exclude_ids, domain)
        remaining = limit - len(results)
        if remaining > 0:
            filled_ids = exclude_ids | {r.author.id for r in results}
            sql_results = self.query_sqlite(topic, remaining, filled_ids, domain)
            results.extend(sql_results)
        return results


# ---------------------------------------------------------
# SCHOLAR AGENT CORE
# ---------------------------------------------------------
class ScholarAgent:
    def __init__(self):
        self.provider    = RealSemanticScholarProvider()
        self._cache      = ScholarCacheProvider()
        self.cache_used  = False  # set True when cache fills shortfall

    # FIX 4: smarter fallback query sequence instead of just word-dropping
    def _build_query_sequence(self, topic: str) -> List[str]:
        """
        Return a list of progressively broader queries to try in order.
        Strategy:
          1. Original topic (most specific)
          2. Drop the last word (standard broadening)
          3. Keep only the first 2–3 words (core concept)
        This prevents dropping clinically important qualifiers too early.
        For "CRISPR base editing off-target effects":
          → "CRISPR base editing off-target effects"
          → "CRISPR base editing off-target"   (drop "effects")
          → "CRISPR base editing"              (core concept)
        We stop at 2 words minimum to avoid queries so generic they're useless.
        """
        queries = [topic]
        words = topic.split()
        # Step-by-step: drop one word at a time but stop at 2 words
        for i in range(1, len(words) - 1):
            candidate = " ".join(words[:-i])
            if candidate not in queries:
                queries.append(candidate)
        return queries[:3]  # cap at 3 passes — each failed API call waits for HTTP timeout

    async def find_experts_by_topic(self, topic: str, limit: int = 5,
                                    domain: str = DEFAULT_DOMAIN) -> List[ExpertResult]:
        diverse_results: List[ExpertResult] = []
        seen_top_paper_ids: set = set()
        seen_author_ids: set = set()
        self.cache_used = False
        domain = _domains.normalize(domain)

        # ── STEP 1: Query local cache first (restricted to the active domain) ──
        cache_results = self._cache.query(topic, limit, set(), domain)
        if cache_results:
            print(f"   [Cache] Found {len(cache_results)}/{limit} from local DB "
                  f"[domain={domain}].")
            for r in cache_results:
                aid = r.author.id if r.author.id != "no-id" else r.author.name.lower().strip()
                if aid not in seen_author_ids:
                    diverse_results.append(r)
                    seen_author_ids.add(aid)
                    for p in r.matched_papers:
                        seen_top_paper_ids.add(p.id)
            self.cache_used = True

        if len(diverse_results) >= limit:
            return diverse_results

        # ── STEP 2: Cache insufficient — call API for the remaining slots ─────
        # Start from the SECOND (broadened) query: the cache already covers the
        # exact-topic match, so repeating pass 1 against the live API would just
        # duplicate what the cache already provided.
        query_sequence = self._build_query_sequence(topic)
        api_sequence = query_sequence[1:] if len(query_sequence) > 1 else query_sequence

        print(
            f"   [Cache] {len(diverse_results)}/{limit} found. "
            f"Falling back to API (starting at broadened query)..."
        )

        for attempt, current_topic in enumerate(api_sequence):
            if len(diverse_results) >= limit:
                break

            print(f"   [ScholarAgent] API Pass {attempt + 1}/{len(api_sequence)}: '{current_topic}'")
            papers = await llm_cache.memo_async("s2_papers", [current_topic, 150], self.provider.search_papers,
                                                current_topic, limit=150)

            # Drop News and Editorials
            valid_papers = [
                p for p in papers
                if not any(pt in ["News", "Editorial"] for pt in p.publication_types)
            ]

            author_papers: Dict[str, List[Paper]] = defaultdict(list)
            author_info: Dict[str, dict] = {}

            for p in valid_papers:
                for ab in p.authors:
                    if not ab.name:
                        continue
                    auth_key = ab.id if ab.id else ab.name.lower().strip()
                    author_papers[auth_key].append(p)
                    author_info[auth_key] = {"id": ab.id or "no-id", "name": ab.name}

            results: List[ExpertResult] = []
            for key, matched_papers in author_papers.items():
                if key in seen_author_ids:
                    continue

                info = author_info[key]
                count_score = len(matched_papers) / (len(matched_papers) + 2)
                total_cites = sum(p.citation_count for p in matched_papers)
                cite_score = math.log1p(total_cites) / (math.log1p(total_cites) + 5)
                score = min(round((0.6 * count_score) + (0.4 * cite_score), 4), 1.0)
                expertise = self._extract_smart_keywords(matched_papers)

                results.append(ExpertResult(
                    author=Author(
                        id=info["id"],
                        name=info["name"],
                        paper_count=len(matched_papers),
                        citation_count=total_cites,
                    ),
                    relevance_score=score,
                    matched_papers=matched_papers,
                    expertise_areas=expertise,
                ))

            # Must have at least 1 citation
            results = [r for r in results if r.author.citation_count > 0]
            results.sort(key=lambda r: (r.relevance_score, r.author.citation_count), reverse=True)

            # Diversity filter: one unique top paper per author
            for r in results:
                r.matched_papers.sort(key=lambda p: p.citation_count, reverse=True)

                unique_paper = None
                for p in r.matched_papers:
                    if p.id not in seen_top_paper_ids:
                        unique_paper = p
                        break

                if unique_paper:
                    r.matched_papers.remove(unique_paper)
                    r.matched_papers.insert(0, unique_paper)

                    diverse_results.append(r)
                    seen_top_paper_ids.add(unique_paper.id)
                    auth_key = (
                        r.author.id
                        if r.author.id != "no-id"
                        else r.author.name.lower().strip()
                    )
                    seen_author_ids.add(auth_key)

                if len(diverse_results) >= limit:
                    break

        return diverse_results

    async def find_collaborators(
        self, researcher_name: str, topic: str, limit: int = 5,
        domain: str = DEFAULT_DOMAIN
    ) -> Dict[str, Any]:
        candidates = await self.provider.search_authors(researcher_name)
        if not candidates:
            fallback_experts = await self.find_experts_by_topic(
                topic, limit=limit, domain=domain
            )
            return {
                "status": "fallback",
                "message": (
                    f"Researcher '{researcher_name}' not found. "
                    f"Returning top diverse experts in '{topic}' instead."
                ),
                "experts": fallback_experts,
            }
        experts = await self.find_experts_by_topic(topic, limit=limit, domain=domain)
        return {
            "status": "success",
            "message": f"Found diverse experts in '{topic}'.",
            "experts": experts,
        }

    def _extract_smart_keywords(self, author_papers: List[Paper]) -> List[str]:
        fos_counts: Dict[str, int] = defaultdict(int)
        for p in author_papers:
            for field in p.fields_of_study:
                fos_counts[field.lower()] += 1
        if fos_counts:
            return [
                f for f, _ in sorted(fos_counts.items(), key=lambda x: x[1], reverse=True)[:4]
            ]
        return ["specific research data unavailable"]

# ---------------------------------------------------------
# WRAPPER FOR MAIN.PY
# ---------------------------------------------------------
agent_instance = ScholarAgent()

# Sentinel embedded in output strings so the orchestrator can detect shortfalls
# without parsing natural language.  Format:  [SHORTFALL requested=N found=M]
# The orchestrator should check for this token and re-issue the call with limit=N.
SHORTFALL_MARKER = "[SHORTFALL requested={requested} found={found}]"
JUDGE_SCORE_THRESHOLD = 7  # re-query internally if judge scores below this

# ---------------------------------------------------------
# PAPER MODE: which paper is this request about, and what is its DOI?
# ---------------------------------------------------------
async def _llm_text(model, prompt: str) -> str:
    res = await asyncio.to_thread(model.generate_content, prompt)
    _scholar_add_tokens(res)
    return res.text or ""


def _europepmc_search(query: str, limit: int = 10) -> List[Paper]:
    """Europe PMC REST search (life sciences, includes preprints). Title matches first, then a full-text
    search sorted by citations, since papers that merely use a tool outnumber the one introducing it."""
    papers, seen = [], set()
    for q, sort in ((f"TITLE:({query})", ""), (query, "&sort=CITED%20desc")):
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core"
               f"&pageSize={limit}{sort}&query={urllib.parse.quote(q)}")
        try:
            with urllib.request.urlopen(url, timeout=20, context=_SSL_CTX) as r:
                hits = json.load(r).get("resultList", {}).get("result", [])
        except Exception:
            hits = []
        for h in hits:
            key = h.get("doi") or h.get("id")
            if key in seen:
                continue
            seen.add(key)
            papers.append(Paper(id="EPMC:" + str(key), title=re.sub(r"<[^>]+>", "", h.get("title") or ""),
                                abstract=h.get("abstractText") or "", year=int(h.get("pubYear") or 0),
                                citation_count=int(h.get("citedByCount") or 0), authors=[], publication_types=[],
                                fields_of_study=[], doi=h.get("doi") or "", venue=h.get("journalTitle") or ""))
        if len(papers) >= limit:
            break
    return papers[:limit]


def _quoted_phrases(text: str, words: int = 10) -> List[str]:
    """Spans of `words` consecutive words from passages quoted in the request (>= 8 words), for an
    exact-phrase full-text search: the most reliable way to find the paper a sentence comes from."""
    spans = []
    for q in re.findall(r"[\"“'‘]([^\"”’]{40,}?)[\"”’']", text or ""):
        toks = re.sub(r"[^\w\s-]", " ", q).split()
        if len(toks) >= 8:
            spans += [" ".join(toks[:words]), " ".join(toks[-words:])]
    return list(dict.fromkeys(spans))[:2]


async def find_paper(user_input: str, llm, depth: int = 1) -> str:
    """Search Semantic Scholar with LLM-written title-style queries, then let the LLM pick the matching
    paper from the candidates. DOIs in the answer come only from the candidate list."""
    try:
        queries = genai.parse_json(await _llm_text(llm, (
            "Write up to 3 short literature search queries that would find the paper this request is about. "
            "The first query is the tool, method or dataset name alone, exactly as written (if the request names "
            "one); the others are 2 to 5 distinctive keywords. Do not guess the paper's title. "
            'Return only JSON: {"queries": ["...", "..."]}\n\n'
            f"Request: {user_input}")))["queries"][: 2 + depth]
    except Exception:
        queries = [user_input[:200]]
    seen, candidates = set(), []
    for phrase in _quoted_phrases(user_input):  # exact-phrase hits first: they contain the quoted text
        for p in await asyncio.to_thread(llm_cache.memo, "europepmc", [f'"{phrase}"', 5], _europepmc_search,
                                         f'"{phrase}"', 5):
            key = (p.doi or p.id or p.title).lower()
            if key and key not in seen:
                seen.add(key)
                p.abstract = f"[contains the quoted text] {p.abstract}"
                candidates.append(p)
    for q in queries:  # Semantic Scholar and Europe PMC; the same paper from both is kept once (by DOI)
        try:
            found = await llm_cache.memo_async("s2_papers", [str(q), 10], agent_instance.provider.search_papers,
                                               str(q), limit=10)
        except Exception:
            found = []
        found = list(found) + await asyncio.to_thread(llm_cache.memo, "europepmc", [str(q), 10], _europepmc_search,
                                                      str(q), 10)
        for p in found:
            key = (p.doi or p.id or p.title).lower()
            if key and key not in seen:
                seen.add(key)
                candidates.append(p)
    print(f"   [Paper mode] queries={queries} | {len(candidates)} candidates")
    trace.record_retrieval("paper_search", "; ".join(queries), [
        {"id": p.doi or p.id, "title": p.title, "year": p.year} for p in candidates[:25]])
    if not candidates:
        return "[STRUCTURAL_FAIL: 0 papers retrieved]\n[Scholar Agent - Paper Search] No candidate papers found."
    listing = "\n".join(
        f"[{i + 1}] {p.title} ({p.year}; {p.venue or 'venue n/a'}) DOI: {p.doi or 'none'} | "
        f"{(p.abstract or '')[:1500]}" for i, p in enumerate(candidates[:25]))  # tools are often named late in the abstract
    answer = await _llm_text(llm, (
        "Pick the paper from the candidates below that the request is about. Use only titles and DOIs that "
        "appear in the list; never invent a DOI. If several candidates fit (e.g. a tool with several versions), "
        "give the original paper first. A candidate marked [contains the quoted text] was found by an exact "
        "full-text search for the passage quoted in the request, so it contains that passage even when its "
        "abstract does not; prefer it. A tool's paper often does not name the tool in its title; check the "
        "abstract for the tool's name or method. If none fits, say that no candidate matches.\n"
        "Reply with: the title, year, venue and DOI of the chosen paper, then one sentence on why it matches.\n\n"
        f"Request: {user_input}\n\nCandidates:\n{listing}"))
    return (f"[Scholar Agent - Paper Search]\n{answer.strip()}\n\n   Candidates considered:\n"
            + "\n".join(f"   - {p.title} ({p.year}) DOI: {p.doi or 'none'}" for p in candidates[:8]))


async def call_scholar_agent(
    user_input: str,
    is_collaboration: bool = False,
    user_name: str = "Unknown Researcher",
    limit: int = 1,
    strategy: str = "cot",
    depth: int = 1,
    agent_registry: str = "",
    domain: str = DEFAULT_DOMAIN,
) -> tuple[str, dict]:
    _acc = {"input": 0, "output": 0}
    _tok = _scholar_token_acc.set(_acc)
    domain = _domains.normalize(domain)

    # depth > 1: raise internal quality bar (harder judge threshold → more self-correction)
    # depth >= 3: also broaden expert search by +1 to increase coverage
    effective_threshold = JUDGE_SCORE_THRESHOLD + max(0, depth - 1)  # 7 → 8 → 9
    if depth >= 3:
        limit = limit + 1

    llm = get_llm_model()
    if llm and is_paper_lookup(user_input):
        result = await find_paper(user_input, llm, depth)
        _scholar_token_acc.reset(_tok)
        return result, _acc

    clean_topic = await extract_search_topic(user_input, llm, domain)
    print(f"   [System] Cleaned query for API: '{clean_topic}' | "
          f"Requesting {limit} expert(s) | domain={domain}")

    # -------------------------------------------------------
    # STEP 1: First search pass
    # -------------------------------------------------------
    if is_collaboration:
        res = await agent_instance.find_collaborators(
            user_name, clean_topic, limit=limit, domain=domain
        )
        experts_list: List[ExpertResult] = res["experts"]
        mode_label = "Collaborator Mode"
        status_msg = res["message"]
    else:
        experts_list = await agent_instance.find_experts_by_topic(
            clean_topic, limit=limit, domain=domain
        )
        mode_label = "Expert Search Mode"
        status_msg = ""

    # -------------------------------------------------------
    # STEP 2: Self-reflection — judge the first-pass result
    # -------------------------------------------------------
    def _format_experts(experts: List[ExpertResult], label: str = "Expert") -> str:
        if not experts:
            return "No experts found. The API might not have data on this."
        out = f"Found {len(experts)} top distinct expert(s) in this field:"
        for i, exp in enumerate(experts):
            top_papers = exp.matched_papers[:3]
            papers_str = ""
            for p in top_papers:
                papers_str += (
                    f"\n       • [{p.year}] \"{p.title}\"  ({p.citation_count:,} citations)"
                )
            affil = ""
            if exp.author.affiliations:
                affil = f"\n     Institution     : {exp.author.affiliations[0]}"
            out += (
                f"\n\n   {label} {i + 1}: {exp.author.name}"
                f"{affil}"
                f"\n     Relevance Score : {exp.relevance_score}"
                f"\n     Total Citations : {exp.author.citation_count:,}"
                f"\n     Matched Papers  : {exp.author.paper_count} on this topic"
                f"\n     Research Areas  : {', '.join(exp.expertise_areas[:3])}"
                f"\n     Top Papers      : {papers_str if papers_str else ' (not available)'}"
            )
        return out

    base_output = (
        f"{status_msg} " if status_msg else ""
    ) + _format_experts(
        experts_list, label="Candidate" if is_collaboration else "Expert"
    )

    # Judge evaluates against the cleaned search topic, not the raw orchestrator
    # sub-task which may contain algorithm/extraction instructions that scholar
    # was never meant to fulfil.
    judge_result = await scholar_llm_judge(
        query=clean_topic,
        agent_output=base_output,
        model=llm,
        requested_count=limit,
        found_count=len(experts_list),
    )
    print(
        f"   [Judge] Score: {judge_result['score']}/10 | Reason: {judge_result['reason']}"
    )

    # -------------------------------------------------------
    # STEP 3: Active self-correction
    # If the judge score is low AND it suggests a better query,
    # do one more internal search with that query before giving up.
    # Retry whenever quality is low — not only on shortfall.
    # -------------------------------------------------------
    if (
        judge_result["score"] < effective_threshold
        and judge_result["retry_query"]
    ):
        retry_topic = judge_result["retry_query"]
        print(
            f"   [Self-correction] Judge score {judge_result['score']}/10 < {effective_threshold} "
            f"(depth={depth}). Retrying with suggested query: '{retry_topic}'"
        )
        retry_experts = await agent_instance.find_experts_by_topic(
            retry_topic, limit=limit, domain=domain
        )
        if len(retry_experts) > len(experts_list):
            experts_list = retry_experts
            base_output = _format_experts(
                experts_list, label="Candidate" if is_collaboration else "Expert"
            )
            # Re-evaluate after correction
            judge_result = await scholar_llm_judge(
                query=clean_topic,
                agent_output=base_output,
                model=llm,
                requested_count=limit,
                found_count=len(experts_list),
            )
            print(
                f"   [Judge after correction] Score: {judge_result['score']}/10"
            )

    # -------------------------------------------------------
    # FIX 1: Emit a machine-readable shortfall marker so the
    # orchestrator knows to retry with the same limit value.
    # -------------------------------------------------------
    shortfall_line = ""
    if len(experts_list) < limit:
        shortfall_line = (
            "\n   " + SHORTFALL_MARKER.format(requested=limit, found=len(experts_list))
        )
        print(
            f"   [Shortfall] Requested {limit}, found {len(experts_list)}. "
            f"Orchestrator should retry with limit={limit}."
        )

    cache_line = "\n   [CACHE-SUPPLEMENTED: results from local researcher cache]" \
        if agent_instance.cache_used else ""

    judge_summary = (
        f"Score: {judge_result['score']}/10 — {judge_result['reason']}"
    )

    result = (
        f"[Scholar Agent - {mode_label}]\n"
        f"   {base_output}"
        f"{shortfall_line}"
        f"{cache_line}\n"
        f"   -----------------------------------\n"
        f"   ⚖️  LLM Judge: {judge_summary}"
    )
    if not experts_list:
        result = f"[STRUCTURAL_FAIL: 0 experts retrieved — API returned no results for this topic]\n{result}"
    _scholar_token_acc.reset(_tok)
    return result, _acc