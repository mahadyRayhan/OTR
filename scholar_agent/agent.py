import os
import json
import asyncio
import time
import math
import re
import urllib.request
import urllib.parse
import google.generativeai as genai
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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

@dataclass
class Author:
    id: str
    name: str
    paper_count: int
    citation_count: int

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

    async def _fetch(self, url: str) -> dict:
        await self.limiter.acquire()
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("x-api-key", api_key)

        try:
            def do_request():
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode('utf-8'))
            return await asyncio.to_thread(do_request)
        except Exception as e:
            print(f"[API Warning] Failed to fetch data: {e}")
            return {}

    async def search_papers(self, query: str, limit: int = 30) -> List[Paper]:
        safe_query = urllib.parse.quote(query)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={safe_query}&limit={limit}"
            f"&fields=title,abstract,year,citationCount,authors,publicationTypes,s2FieldsOfStudy"
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
    # return genai.GenerativeModel('gemini-2.5-flash')
    return genai.GenerativeModel('gemini-3.1-flash-lite-preview')

async def extract_search_topic(query: str, model) -> str:
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
 
    prompt = (
        "Convert this input into a concise academic search query (3–8 words).\n\n"
        "KEEP — these are search signal:\n"
        "  • Specific tool or software names        (e.g. 'Kallisto', 'STAR aligner')\n"
        "  • Specific biological subtypes            (e.g. 'base editing', not just 'CRISPR';\n"
        "                                              'adenine base editor', not just 'gene editor')\n"
        "  • Named authors or papers                 (e.g. 'Bray et al 2016', 'Pachter')\n"
        "  • Domain-narrowing modifiers that narrow the RESEARCH FIELD\n"
        "    (e.g. 'off-target effects', 'pseudoalignment', 'RNA-seq quantification')\n\n"
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
        "      'retrieve parameter values from the Kallisto paper'  → 'Kallisto RNA-seq pseudoalignment'\n"
        "      'extract exact values from BWA documentation'        → 'BWA short read alignment'\n"
        "      'get detection methods from the CRISPR article'      → 'CRISPR base editing'\n\n"
        "Output ONLY the search query — no punctuation, no quotes, no explanation.\n\n"
        f"Input: '{query}'\n"
        "Search query:"
    )
    try:
        res = await asyncio.to_thread(model.generate_content, prompt)
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
# SCHOLAR AGENT CORE
# ---------------------------------------------------------
class ScholarAgent:
    def __init__(self):
        self.provider = RealSemanticScholarProvider()

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

    async def find_experts_by_topic(self, topic: str, limit: int = 5) -> List[ExpertResult]:
        diverse_results: List[ExpertResult] = []
        seen_top_paper_ids: set = set()
        seen_author_ids: set = set()

        query_sequence = self._build_query_sequence(topic)

        # FIX 2: attempts increments only when we actually need to retry (inside the
        # shortfall branch), so we don't waste a retry slot on a successful pass.
        for attempt, current_topic in enumerate(query_sequence):
            if len(diverse_results) >= limit:
                break

            print(f"   [ScholarAgent] Pass {attempt + 1}/{len(query_sequence)}: '{current_topic}'")
            papers = await self.provider.search_papers(current_topic, limit=100)

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
        self, researcher_name: str, topic: str, limit: int = 5
    ) -> Dict[str, Any]:
        candidates = await self.provider.search_authors(researcher_name)
        if not candidates:
            fallback_experts = await self.find_experts_by_topic(topic, limit=limit)
            return {
                "status": "fallback",
                "message": (
                    f"Researcher '{researcher_name}' not found. "
                    f"Returning top diverse experts in '{topic}' instead."
                ),
                "experts": fallback_experts,
            }
        experts = await self.find_experts_by_topic(topic, limit=limit)
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

async def call_scholar_agent(
    user_input: str,
    is_collaboration: bool = False,
    user_name: str = "Unknown Researcher",
    limit: int = 1,
    strategy: str = "cot",
    depth: int = 1,
    agent_registry: str = "",
) -> str:
    llm = get_llm_model()
    clean_topic = await extract_search_topic(user_input, llm)
    print(f"   [System] Cleaned query for API: '{clean_topic}' | Requesting {limit} expert(s)")

    # -------------------------------------------------------
    # STEP 1: First search pass
    # -------------------------------------------------------
    if is_collaboration:
        res = await agent_instance.find_collaborators(user_name, clean_topic, limit=limit)
        experts_list: List[ExpertResult] = res["experts"]
        mode_label = "Collaborator Mode"
        status_msg = res["message"]
    else:
        experts_list = await agent_instance.find_experts_by_topic(clean_topic, limit=limit)
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
            out += (
                f"\n\n   {label} {i + 1}: {exp.author.name} "
                f"(Score: {exp.relevance_score}, Citations: {exp.author.citation_count})"
                f"\n     Keywords: {exp.expertise_areas}"
                f"\n     Top Paper: '{exp.matched_papers[0].title}'"
            )
        return out

    base_output = (
        f"{status_msg} " if status_msg else ""
    ) + _format_experts(
        experts_list, label="Candidate" if is_collaboration else "Expert"
    )

    # FIX 3 & 4: call the active judge, pass count context
    judge_result = await scholar_llm_judge(
        query=user_input,
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
    # -------------------------------------------------------
    if (
        judge_result["score"] < JUDGE_SCORE_THRESHOLD
        and judge_result["retry_query"]
        and len(experts_list) < limit
    ):
        retry_topic = judge_result["retry_query"]
        print(
            f"   [Self-correction] Judge score {judge_result['score']}/10 < {JUDGE_SCORE_THRESHOLD}. "
            f"Retrying with suggested query: '{retry_topic}'"
        )
        retry_experts = await agent_instance.find_experts_by_topic(
            retry_topic, limit=limit
        )
        if len(retry_experts) > len(experts_list):
            experts_list = retry_experts
            base_output = _format_experts(
                experts_list, label="Candidate" if is_collaboration else "Expert"
            )
            # Re-evaluate after correction
            judge_result = await scholar_llm_judge(
                query=user_input,
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

    judge_summary = (
        f"Score: {judge_result['score']}/10 — {judge_result['reason']}"
    )

    return (
        f"[Scholar Agent - {mode_label}]\n"
        f"   {base_output}"
        f"{shortfall_line}\n"
        f"   -----------------------------------\n"
        f"   ⚖️  LLM Judge: {judge_summary}"
    )