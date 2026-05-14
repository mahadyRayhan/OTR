"""
build_index.py — Build the FAISS knowledge base index.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import aiohttp
import numpy as np

from knowledge_base import KnowledgeBase, Paper
from config import settings

QUERIES = [
    # Single-cell / Scanpy
    "Scanpy single-cell RNA sequencing clustering analysis",
    "Leiden community detection single-cell transcriptomics",
    "PCA UMAP dimensionality reduction single-cell",
    "differential expression analysis scRNA-seq marker genes",
    "Harmony batch effect correction single-cell integration",
    "MAGIC imputation single-cell RNA sequencing",
    "tumor microenvironment single-cell RNA sequencing breast cancer",
    "PCA UMAP dimensionality reduction single-cell transcriptomics",
    "Scanpy Wilcoxon rank sum test marker genes cell clusters",      
    "Leiden algorithm graph modularity community detection",          
    "single-cell RNA-seq dimensionality reduction t-SNE diffusion",  

    # Sequence analysis
    "Kallisto pseudoalignment RNA-seq quantification",
    "FASTQ file format sequencing reads quality scores",
    "FastQC Trimmomatic RNA-seq preprocessing pipeline",
    "Needleman-Wunsch Smith-Waterman sequence alignment algorithms",
    "Kallisto pseudoalignment versus STAR aligner RNA-seq",          
    "FASTQ quality score Phred encoding Illumina sequencing",        

    # Protein / structural biology
    "AlphaFold protein structure prediction deep learning",
    "Rosetta PyRosetta protein folding de novo",
    "AlphaFold2 versus Rosetta protein structure comparison",         
    "protein structure prediction methods deep learning benchmark",   

    # CRISPR
    "CRISPR base editing off-target effects detection",
    "CRISPR base editing GUIDE-seq CIRCLE-seq off-target detection", 
    "adenine cytosine base editor off-target genomic effects",       

    # Proteomics
    "mass spectrometry proteomics MaxQuant data processing",

    # Neuroscience
    "synaptic plasticity STDP hippocampus LTP",
    "AMPA receptor trafficking long-term potentiation",
    "calcium signaling synaptic plasticity STDP hippocampus",        

    # Graph neural networks
    "graph neural networks protein structure biology",
    "graph neural networks protein interaction network GNN biology",  

    # RNA-seq alignment (covers coreference_explanation gap)
    "splice-aware RNA-seq alignment STAR HISAT2 algorithm",         
    "RNA-seq splice junction alignment intron splicing detection",   

    # Cloud / HPC for bioinformatics
    "memory requirements single-cell RNA-seq clustering million cells",
    "cloud computing bioinformatics HPC AWS single-cell analysis",
    "distributed computing Scanpy Seurat large scale single-cell",
    "memory-optimized instances bioinformatics genomics cloud",
    "scalable single-cell analysis computational resources benchmark", 
]

S2_FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf"


async def fetch_semantic_scholar(session, query, limit=20, retries=5):
    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
    for attempt in range(retries):
        try:
            async with session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status == 429:
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if r.status != 200:
                    print(f"  S2 status {r.status}")
                    return []
                data = await r.json(content_type=None)

            papers = []
            for item in data.get("data", []):
                abstract = item.get("abstract") or ""
                if not abstract:
                    continue
                authors = item.get("authors", [])
                author_str = ", ".join(a.get("name", "") for a in authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                ext_ids = item.get("externalIds", {})
                pmid = f"PMID{ext_ids['PubMed']}" if "PubMed" in ext_ids else f"SS:{item.get('paperId','')[:8]}"
                papers.append(Paper(
                    title=item.get("title", "Untitled"),
                    authors=author_str,
                    journal=item.get("venue", ""),
                    year=item.get("year") or 2000,
                    pmid=pmid,
                    abstract=abstract,
                    source="semantic_scholar",
                    url=(item.get("openAccessPdf") or {}).get("url", ""),
                ))
            return papers
        except Exception as e:
            print(f"  S2 attempt {attempt+1} failed: {e}")
            await asyncio.sleep(5)
    return []


async def fetch_pubmed(session, query, max_results=20):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {"db": "pubmed", "term": query, "retmax": max_results,
              "retmode": "json", "email": settings.ncbi_email, "sort": "relevance"}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    try:
        async with session.get(f"{base}/esearch.fcgi", params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json(content_type=None)
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        async with session.get(f"{base}/esummary.fcgi",
                               params={"db": "pubmed", "id": ",".join(ids),
                                       "retmode": "json", "email": settings.ncbi_email},
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json(content_type=None)
        papers = []
        for uid, summary in data.get("result", {}).items():
            if uid == "uids":
                continue
            authors = summary.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            papers.append(Paper(
                title=summary.get("title", "Untitled"),
                authors=author_str,
                journal=summary.get("source", ""),
                year=int(summary.get("pubdate", "2000")[:4] or 2000),
                pmid=f"PMID{uid}",
                abstract="",
                source="pubmed",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            ))
        return papers
    except Exception as e:
        print(f"  PubMed failed: {e}")
        return []


def embed_in_batches(model, texts, batch_size=4):
    """Embed texts in small batches to avoid segfault on macOS Python 3.9."""
    all_vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size} ({len(batch)} papers)...")
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_vectors.append(vecs)
    return np.vstack(all_vectors)


async def main():
    kb = KnowledgeBase()
    kb.load()

    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
        print(f"Using S2 API key: ...{settings.semantic_scholar_api_key[-6:]}")
    else:
        print("No S2 key — unauthenticated (heavy rate limiting expected)")

    all_papers = []

    async with aiohttp.ClientSession(headers=headers) as session:
        for i, query in enumerate(QUERIES):
            print(f"\n[{i+1}/{len(QUERIES)}] Fetching: {query}")
            if i > 0:
                await asyncio.sleep(1.1)

            papers = await fetch_semantic_scholar(session, query, limit=20)
            source = "Semantic Scholar"

            if not papers:
                print("  S2 returned 0 — trying PubMed...")
                papers = await fetch_pubmed(session, query)
                source = "PubMed"

            print(f"  → {len(papers)} papers from {source}")
            all_papers.extend(papers)

    all_papers = kb._deduplicate(all_papers)
    all_papers = [p for p in all_papers if p.abstract.strip()]
    print(f"\nTotal papers to index: {len(all_papers)}")

    if not all_papers:
        print("No papers fetched.")
        return

    # Embed in small batches to avoid macOS/Python 3.9 segfault
    print(f"\nEmbedding {len(all_papers)} papers in small batches...")
    texts = [f"{p.title} {p.abstract}" for p in all_papers]
    vectors = embed_in_batches(kb.embedder._model, texts, batch_size=4)

    metadata = [
        {"title": p.title, "authors": p.authors, "journal": p.journal,
         "year": p.year, "pmid": p.pmid, "abstract": p.abstract, "url": p.url}
        for p in all_papers
    ]

    kb.faiss.add(vectors, metadata)
    kb.faiss.save()
    print(f"\nDone — indexed {len(all_papers)} papers into ./data/")
    print("You can now run: python main.py")


asyncio.run(main())