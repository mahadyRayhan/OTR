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
    publication_types: List[str]  # Added to filter News
    fields_of_study: List[str]    # Added for smart keywords

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
        # 🚀 Added publicationTypes and s2FieldsOfStudy
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={safe_query}&limit={limit}&fields=title,abstract,year,citationCount,authors,publicationTypes,s2FieldsOfStudy"
        
        data = await self._fetch(url)
        papers = []
        
        for item in data.get("data", []):
            # Extract fields of study tags
            fos_raw = item.get("s2FieldsOfStudy") or []
            fos = [f.get("category") for f in fos_raw if "category" in f]
            
            pub_types = item.get("publicationTypes") or []
            
            authors = [AuthorBrief(id=a.get("authorId"), name=a.get("name")) for a in item.get("authors", [])]
            papers.append(Paper(
                id=item.get("paperId", ""),
                title=item.get("title", ""),
                abstract=item.get("abstract", "") or "",
                year=item.get("year") or 2024,
                citation_count=item.get("citationCount") or 0,
                authors=authors,
                publication_types=pub_types,
                fields_of_study=fos
            ))
        return papers

    async def search_authors(self, name: str) -> List[AuthorBrief]:
        safe_name = urllib.parse.quote(name)
        url = f"https://api.semanticscholar.org/graph/v1/author/search?query={safe_name}&limit=5"
        data = await self._fetch(url)
        return [AuthorBrief(id=a.get("authorId"), name=a.get("name")) for a in data.get("data", [])]

# ---------------------------------------------------------
# LLM UTILITIES
# ---------------------------------------------------------
def get_llm_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here": return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

async def extract_search_topic(query: str, model) -> str:
    if not model:
        return re.sub(r'(?i)(find me some research papers on|i am looking for a collaborator to help with a|tell me about|what is)', '', query).strip()
    prompt = f"Extract ONLY the core scientific/academic topic from this user input. No extra words. Input: '{query}'"
    try:
        res = await asyncio.to_thread(model.generate_content, prompt)
        return res.text.strip().lower()
    except:
        return query

async def scholar_llm_judge(query: str, agent_output: str, model) -> str:
    if not model: return "Mock Judge: 8/10."
    try:
        prompt = f"""
        You are an Academic Advisor evaluating the output of an AI Scholar Search Agent.
        User's Research Query: "{query}"
        Agent's Output: "{agent_output}"
        
        Evaluate how relevant these experts and their extracted keywords are to the user's query. 
        Provide a score out of 10 and a strictly 1-sentence justification. Format: 'Score: X/10 - [Reason]'
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        return f"Judge Error: {str(e)}"

# ---------------------------------------------------------
# SCHOLAR AGENT CORE
# ---------------------------------------------------------
class ScholarAgent:
    def __init__(self):
        self.provider = RealSemanticScholarProvider()

    async def find_experts_by_topic(self, topic: str, limit: int = 5) -> List[ExpertResult]:
        papers = await self.provider.search_papers(topic, limit=40)
        
        # 🚀 Fix 1: Drop News and Editorials (The Journalist Problem)
        valid_papers = [p for p in papers if not any(pt in ["News", "Editorial"] for pt in p.publication_types)]
        
        author_papers = defaultdict(list)
        author_info = {}

        for p in valid_papers:
            for ab in p.authors:
                if not ab.name: continue
                auth_key = ab.id if ab.id else ab.name.lower().strip()
                author_papers[auth_key].append(p)
                author_info[auth_key] = {"id": ab.id or "no-id", "name": ab.name}

        results = []
        for key, matched_papers in author_papers.items():
            info = author_info[key]
            
            count_score = len(matched_papers) / (len(matched_papers) + 2)
            total_cites = sum(p.citation_count for p in matched_papers)
            cite_score = math.log1p(total_cites) / (math.log1p(total_cites) + 5)
            score = min(round((0.6 * count_score) + (0.4 * cite_score), 4), 1.0)
            
            # 🚀 Fix 3: Use real AI tags from Semantic Scholar instead of TF-IDF
            expertise = self._extract_smart_keywords(matched_papers)

            results.append(ExpertResult(
                author=Author(id=info["id"], name=info["name"], paper_count=len(matched_papers), citation_count=total_cites),
                relevance_score=score,
                matched_papers=matched_papers,
                expertise_areas=expertise
            ))

        # Must have at least 1 citation
        results = [r for r in results if r.author.citation_count > 0]
        # Sort by score initially
        results.sort(key=lambda r: (r.relevance_score, r.author.citation_count), reverse=True)
        
        # 🚀 Fix 2: The Diversity Filter (The Clone Problem)
        diverse_results = []
        seen_top_paper_ids = set()
        
        for r in results:
            top_paper_id = r.matched_papers[0].id
            # Only add this expert if their top paper wasn't already used by a higher-scoring expert
            if top_paper_id not in seen_top_paper_ids:
                diverse_results.append(r)
                seen_top_paper_ids.add(top_paper_id)
            
            if len(diverse_results) == limit:
                break
                
        return diverse_results

    async def find_collaborators(self, researcher_name: str, topic: str, limit: int = 5) -> Dict[str, Any]:
        candidates = await self.provider.search_authors(researcher_name)
        if not candidates:
            fallback_experts = await self.find_experts_by_topic(topic, limit=limit)
            return {
                "status": "fallback",
                "message": f"Researcher '{researcher_name}' not found. Returning top diverse experts in '{topic}' instead.",
                "experts": fallback_experts
            }
        experts = await self.find_experts_by_topic(topic, limit=limit)
        return {"status": "success", "message": f"Found diverse experts in '{topic}'.", "experts": experts}

    def _extract_smart_keywords(self, author_papers: List[Paper]) -> List[str]:
        # Aggregate the actual fields of study tagged by Semantic Scholar AI
        fos_counts = defaultdict(int)
        for p in author_papers:
            for field in p.fields_of_study:
                fos_counts[field.lower()] += 1
                
        if fos_counts:
            return [f for f, c in sorted(fos_counts.items(), key=lambda x: x[1], reverse=True)[:4]]
        return ["specific research data unavailable"]

# ---------------------------------------------------------
# WRAPPER FOR MAIN.PY
# ---------------------------------------------------------
agent_instance = ScholarAgent()

async def call_scholar_agent(user_input: str, is_collaboration=False, user_name="Unknown Researcher", limit=1, strategy: str = "cot", depth: int = 1, agent_registry: str = "") -> str:
    llm = get_llm_model()
    clean_topic = await extract_search_topic(user_input, llm)
    print(f"   [System] Cleaned query for API: '{clean_topic}' | Requesting {limit} expert(s)")
    
    if is_collaboration:
        res = await agent_instance.find_collaborators(user_name, clean_topic, limit=limit)
        experts_list = res['experts']
        
        base_output = f"{res['message']} Found {len(experts_list)} candidate(s)."
        for i, exp in enumerate(experts_list):
            base_output += f"\n\n   Candidate {i+1}: {exp.author.name} (Score: {exp.relevance_score}, Citations: {exp.author.citation_count})"
            base_output += f"\n     Keywords: {exp.expertise_areas}"
            base_output += f"\n     Top Paper: '{exp.matched_papers[0].title}'"
            
        judge_eval = await scholar_llm_judge(user_input, base_output, llm)
        return f"[Scholar Agent - Collaborator Mode]\n   {base_output}\n   -----------------------------------\n   ⚖️  LLM Judge: {judge_eval}"
        
    else:
        res = await agent_instance.find_experts_by_topic(clean_topic, limit=limit)
        
        if not res:
            base_output = "No experts found. The API might not have data on this."
        else:
            base_output = f"Found {len(res)} top distinct expert(s) in this field:"
            for i, exp in enumerate(res):
                base_output += f"\n\n   Expert {i+1}: {exp.author.name} (Score: {exp.relevance_score}, Citations: {exp.author.citation_count})"
                base_output += f"\n     Keywords: {exp.expertise_areas}"
                base_output += f"\n     Top Paper: '{exp.matched_papers[0].title}'"

        judge_eval = await scholar_llm_judge(user_input, base_output, llm)
        return f"[Scholar Agent - Expert Search Mode]\n   {base_output}\n   -----------------------------------\n   ⚖️  LLM Judge: {judge_eval}"