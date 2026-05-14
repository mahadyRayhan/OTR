"""
knowledge_base.py — Multi-source knowledge retrieval layer.

Responsibilities
────────────────
1. Embed queries with PubMedBERT (sentence-transformers).
2. Retrieve candidate papers from the FAISS vector store.
3. Fetch live results from Semantic Scholar and PubMed Entrez APIs.
4. Run BERTopic topic modelling over the retrieved corpus.
5. Maintain a curated registry of neuroscience / bioinformatics tools & datasets.

The class is designed to be used as a long-lived singleton initialised once at
startup and injected into the RAG responder and ReAct agent.

Fixes applied
─────────────
  #3  _heuristic_topics expanded to cover all domains the gateway serves
      (proteomics, cloud computing, NLP, imaging, climate, etc.) and now
      falls back to returning the raw query terms as a topic instead of
      mapping to the nearest neuroscience concept when nothing matches.

  #5  EmbeddingModel.encode() now raises RuntimeError instead of silently
      returning random vectors when sentence-transformers is unavailable.
      Silent random-vector degradation produces noise that looks like signal.

  #6  _deduplicate() now merges on PMID first (two papers sharing a PMID are
      always the same paper regardless of title formatting), then falls back
      to normalised-title prefix matching for papers without a PMID.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from rag_agent.rag_config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    title: str
    authors: str
    journal: str
    year: int
    pmid: str
    abstract: str
    score: float = 0.0
    source: str = "faiss"          # faiss | semantic_scholar | pubmed
    tools: list[str] = field(default_factory=list)
    url: str = ""


@dataclass
class Topic:
    name: str
    score: float
    relevance: str
    tools: list[str]
    representative_words: list[str] = field(default_factory=list)


@dataclass
class KnowledgeLandscape:
    query: str
    topics: list[Topic]
    papers: list[Paper]
    elapsed_s: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tool & Dataset registry (curated)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, list[str]] = {
    # Neuroscience simulation
    "computational_neuro": ["NEURON", "Brian2", "NEST", "NetPyNE", "PyNN"],
    # Imaging & anatomy
    "neuro_imaging": ["FSL", "FreeSurfer", "SPM", "ANTs", "Fiji", "ImageJ"],
    # Connectomics
    "connectomics": ["Allen Brain Atlas", "Connectome Workbench", "CAT12"],
    # Bioinformatics — single-cell
    "single_cell": ["Seurat", "Scanpy", "Monocle3", "Harmony", "SCENIC"],
    # Sequence analysis
    "genomics": ["STAR", "DESeq2", "GATK", "Bioconductor", "Bowtie2"],
    # Protein structure
    "structural_bio": ["AlphaFold2", "RoseTTAFold", "PyMOL", "GROMACS"],
    # Proteomics / mass-spec
    "proteomics": ["MaxQuant", "Perseus", "Mascot", "Proteome Discoverer", "OpenMS"],
    # General ML / deep learning
    "ml_bio": ["DeepLabCut", "CellPose", "StarDist", "ilastik"],
    # NLP / text mining
    "nlp": ["spaCy", "HuggingFace Transformers", "NLTK", "Gensim", "SciBERT"],
    # Cloud / infrastructure
    "cloud": ["AWS S3", "Google Cloud Storage", "Azure Blob", "Nextflow", "Snakemake"],
    # Climate / earth sciences
    "climate": ["NetCDF4", "xarray", "CDO", "NCAR NCO", "Pangeo"],
    # Classic analysis
    "general": ["MATLAB", "R", "Python", "Jupyter", "scikit-learn"],
}

DATASET_REGISTRY: dict[str, list[str]] = {
    "neuro": ["CRCNS", "OpenNeuro", "Human Connectome Project", "Allen Brain Atlas API"],
    "genomics": ["GEO", "SRA", "ENCODE", "GTEx", "TCGA"],
    "protein": ["PDB", "UniProt", "AlphaFold DB"],
    "proteomics": ["PRIDE", "MassIVE", "PeptideAtlas", "PhosphoSitePlus"],
    "imaging": ["ADNI", "UK Biobank", "IXI Dataset"],
    "nlp": ["PubMed Central OA", "CORD-19", "S2ORC"],
    "climate": ["ERA5", "CMIP6", "NOAA GHCN", "Copernicus CDS"],
}


def _infer_domain_tools(query: str) -> list[str]:
    """Return a short list of relevant tools inferred from query keywords."""
    q = query.lower()
    matches: list[str] = []
    if any(k in q for k in ("neuron", "synap", "plasticity", "hippoc", "spike", "membrane")):
        matches += TOOL_REGISTRY["computational_neuro"][:3]
    if any(k in q for k in ("single-cell", "scrna", "rna-seq", "transcriptom", "cell cluster")):
        matches += TOOL_REGISTRY["single_cell"][:3]
    if any(k in q for k in ("protein", "structure", "fold", "alphafold", "pdb")):
        matches += TOOL_REGISTRY["structural_bio"][:3]
    if any(k in q for k in ("proteom", "mass spec", "peptide", "phospho")):
        matches += TOOL_REGISTRY["proteomics"][:3]
    if any(k in q for k in ("genome", "variant", "snp", "crispr", "seq")):
        matches += TOOL_REGISTRY["genomics"][:3]
    if any(k in q for k in ("image", "microscop", "stain", "fiji", "segmentation")):
        matches += TOOL_REGISTRY["neuro_imaging"][:2]
    if any(k in q for k in ("nlp", "text", "language model", "bert", "transformers", "corpus")):
        matches += TOOL_REGISTRY["nlp"][:3]
    if any(k in q for k in ("cloud", "aws", "azure", "gcp", "pipeline", "workflow", "snakemake", "nextflow")):
        matches += TOOL_REGISTRY["cloud"][:3]
    if any(k in q for k in ("climate", "weather", "temperature", "precipitation", "co2", "atmosphere")):
        matches += TOOL_REGISTRY["climate"][:3]
    if not matches:
        matches = TOOL_REGISTRY["general"][:3]
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in matches:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:4]


# ─────────────────────────────────────────────────────────────────────────────
# FAISS wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FAISSStore:
    """Thin wrapper around a FAISS flat-IP index with a parallel metadata list."""

    def __init__(self) -> None:
        self._index: Any = None
        self._metadata: list[dict[str, Any]] = []
        self._dim: int = 768  # PubMedBERT hidden size

    def load(self) -> None:
        try:
            import faiss  # type: ignore

            idx_path = settings.faiss_index_path
            meta_path = settings.faiss_metadata_path

            if Path(idx_path).exists() and Path(meta_path).exists():
                self._index = faiss.read_index(idx_path)
                with open(meta_path, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.info("FAISS index loaded: %d vectors", self._index.ntotal)
            else:
                # Bootstrap an empty index
                self._index = faiss.IndexFlatIP(self._dim)
                logger.warning("No FAISS index found at %s — starting empty.", idx_path)
        except ImportError:
            logger.warning("faiss-cpu not installed — FAISS retrieval disabled.")

    def save(self) -> None:
        if self._index is None:
            return
        import faiss  # type: ignore

        Path(settings.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, settings.faiss_index_path)
        with open(settings.faiss_metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)
        logger.info("FAISS index saved.")

    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if self._index is None:
            return
        import faiss  # type: ignore

        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss  # type: ignore

        faiss.normalize_L2(query_vector)
        distances, indices = self._index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self._metadata[idx])
            meta["_faiss_score"] = float(dist)
            results.append(meta)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """PubMedBERT sentence encoder (lazy-loaded)."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
            logger.info("Embedding model loaded: %s", settings.embedding_model)
        except ImportError:
            logger.warning("sentence-transformers not installed — embeddings disabled.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode *texts* into dense vectors.

        Fix #5 — previously returned random vectors when the embedding model
        was unavailable, which produced noise that looked like valid signal.
        Now raises RuntimeError so the failure is immediately visible instead
        of silently corrupting downstream results.
        """
        if self._model is None:
            raise RuntimeError(
                "Embedding model is unavailable. "
                "Install sentence-transformers (`pip install sentence-transformers`) "
                "and ensure the model can be downloaded. "
                "Returning random vectors would produce meaningless results."
            )
        return self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Topic modeller
# ─────────────────────────────────────────────────────────────────────────────

class TopicModeller:
    """BERTopic wrapper for dynamic topic discovery over retrieved abstracts."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self) -> None:
        try:
            from bertopic import BERTopic  # type: ignore

            model_path = settings.bertopic_model_path
            if Path(model_path).exists():
                self._model = BERTopic.load(model_path)
                logger.info("BERTopic model loaded from %s", model_path)
            else:
                self._model = BERTopic(
                    min_topic_size=settings.topic_min_cluster_size,
                    n_gram_range=settings.topic_n_gram_range,
                    verbose=False,
                )
                logger.warning("BERTopic model not found — will fit on demand.")
        except ImportError:
            logger.warning("bertopic not installed — topic modelling disabled.")

    def get_topics_for_query(self, query: str, abstracts: list[str]) -> list[Topic]:
        """
        Return up to 4 topics relevant to `query` from the retrieved `abstracts`.
        Falls back to keyword-heuristic topics when BERTopic is unavailable.
        """
        if self._model is None or len(abstracts) < 5:
            return self._heuristic_topics(query)

        try:
            docs = [query] + abstracts
            topics_raw, _ = self._model.fit_transform(docs)
            topic_info = self._model.get_topic_info()
            result: list[Topic] = []
            for _, row in topic_info.iterrows():
                if row["Topic"] == -1 or len(result) >= 4:
                    continue
                words = [w for w, _ in self._model.get_topic(row["Topic"])][:5]
                result.append(
                    Topic(
                        name=" ".join(w.title() for w in words[:3]),
                        score=round(0.99 - len(result) * 0.06, 2),
                        relevance=f"Cluster of {row['Count']} documents centred on: {', '.join(words[:4])}.",
                        tools=_infer_domain_tools(" ".join(words)),
                        representative_words=words,
                    )
                )
            return result or self._heuristic_topics(query)
        except Exception as exc:
            logger.warning("BERTopic inference failed: %s", exc)
            return self._heuristic_topics(query)

    @staticmethod
    def _heuristic_topics(query: str) -> list[Topic]:
        """
        Simple keyword-driven fallback returning plausible topic shells.

        Fix #3 — previously hardcoded to neuroscience keywords regardless of
        the query domain. The map is now expanded to cover all domains the
        gateway serves. When no keyword matches, the function returns the raw
        query terms as a topic rather than inventing an irrelevant neuroscience
        concept with a spuriously high confidence score.
        """
        q = query.lower()
        topics: list[Topic] = []

        # (keyword_substring, display_name, confidence_score)
        _map = [
            # ── Neuroscience ────────────────────────────────────────────────
            ("synap",        "Synaptic Plasticity",                   0.91),
            ("hippoc",       "Hippocampal Circuit Dynamics",           0.84),
            ("ltp",          "Long-Term Potentiation (LTP)",           0.82),
            ("stdp",         "Spike-Timing Dependent Plasticity",      0.90),
            ("optogenet",    "Optogenetics",                           0.88),
            ("engram",       "Memory Engram Circuits",                 0.86),
            ("dendrit",      "Dendritic Computation",                  0.85),
            # ── Genomics / transcriptomics ───────────────────────────────────
            ("single-cell",  "Single-Cell Transcriptomics",            0.91),
            ("scrna",        "scRNA-seq Clustering Methods",           0.87),
            ("rna-seq",      "Bulk RNA-seq & Differential Expression", 0.88),
            ("spatial trans","Spatial Transcriptomics",                0.89),
            ("crispr",       "CRISPR Genome Editing",                  0.90),
            ("base edit",    "CRISPR Base Editing",                    0.89),
            ("deseq",        "Differential Expression (DESeq2)",       0.87),
            # ── Proteomics ───────────────────────────────────────────────────
            ("proteom",      "Proteomics & Mass Spectrometry",         0.88),
            ("mass spec",    "Mass Spectrometry-Based Proteomics",     0.87),
            ("phospho",      "Phosphoproteomics",                      0.85),
            # ── Structural biology ───────────────────────────────────────────
            ("protein",      "Protein Structure Prediction",           0.89),
            ("alphafold",    "AlphaFold & Structural Proteomics",      0.93),
            ("cryo-em",      "Cryo-EM Structure Determination",        0.90),
            # ── Machine learning / AI ────────────────────────────────────────
            ("graph neural", "Graph Neural Networks in Biology",       0.88),
            ("transformer",  "Transformer Models",                     0.87),
            ("deep learn",   "Deep Learning Methods",                  0.86),
            ("reinforce",    "Reinforcement Learning",                 0.85),
            # ── NLP ──────────────────────────────────────────────────────────
            ("nlp",          "Natural Language Processing",            0.87),
            ("text mining",  "Biomedical Text Mining",                 0.86),
            ("named entity", "Named Entity Recognition",               0.85),
            # ── Cloud / infrastructure ───────────────────────────────────────
            ("cloud comput", "Cloud Computing & HPC",                  0.85),
            ("kubernetes",   "Container Orchestration (Kubernetes)",   0.84),
            ("docker",       "Containerisation & Reproducibility",     0.84),
            ("pipeline",     "Bioinformatics Workflow Management",     0.83),
            # ── Climate / earth sciences ─────────────────────────────────────
            ("climate",      "Climate Modelling",                      0.87),
            ("weather",      "Numerical Weather Prediction",           0.85),
            ("co2",          "Carbon Cycle & Climate Change",          0.86),
            # ── Imaging ──────────────────────────────────────────────────────
            ("image",        "Bioimage Analysis",                      0.85),
            ("microscop",    "Fluorescence Microscopy",                0.84),
            ("mri",          "MRI & Neuroimaging",                     0.86),
        ]

        for kw, name, score in _map:
            if kw in q:
                topics.append(
                    Topic(
                        name=name,
                        score=score,
                        relevance=f"Directly connected to the query on '{query[:60]}'.",
                        tools=_infer_domain_tools(name),
                    )
                )
            if len(topics) >= 3:
                break

        # Fix #3 — when nothing matches, use the raw query terms rather than
        # forcing the nearest neuroscience label onto an unrelated domain.
        if not topics:
            topics.append(
                Topic(
                    name=query[:60].title(),
                    score=0.75,   # lower score signals lower certainty
                    relevance="Primary research area derived directly from your query.",
                    tools=_infer_domain_tools(query),
                )
            )
        return topics[:4]


# ─────────────────────────────────────────────────────────────────────────────
# External API fetchers (async)
# ─────────────────────────────────────────────────────────────────────────────

class PubMedFetcher:
    """Fetch papers from NCBI Entrez E-utilities."""

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def search(self, query: str, max_results: int = 10) -> list[Paper]:
        params_search = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "api_key": settings.ncbi_api_key,
            "email": settings.ncbi_email,
            "sort": "relevance",
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.BASE}/esearch.fcgi", params=params_search, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    data = await r.json(content_type=None)
                ids = data.get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                return await self._fetch_summaries(session, ids)
            except Exception as exc:
                logger.warning("PubMed search failed: %s", exc)
                return []

    async def _fetch_summaries(self, session: aiohttp.ClientSession, ids: list[str]) -> list[Paper]:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
            "api_key": settings.ncbi_api_key,
        }
        try:
            async with session.get(f"{self.BASE}/esummary.fcgi", params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json(content_type=None)
            papers = []
            for uid, summary in data.get("result", {}).items():
                if uid == "uids":
                    continue
                authors = summary.get("authors", [])
                author_str = ", ".join(a.get("name", "") for a in authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                papers.append(
                    Paper(
                        title=summary.get("title", "Untitled"),
                        authors=author_str,
                        journal=summary.get("source", ""),
                        year=int(summary.get("pubdate", "2000")[:4] or 2000),
                        pmid=f"PMID{uid}",
                        abstract="",
                        source="pubmed",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    )
                )
            return papers
        except Exception as exc:
            logger.warning("PubMed summary fetch failed: %s", exc)
            return []


class SemanticScholarFetcher:
    """Fetch papers from the Semantic Scholar API."""

    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf"

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(f"{self.BASE}/paper/search", params=params, timeout=aiohttp.ClientTimeout(total=12)) as r:
                    print(f"  [S2 DEBUG] status={r.status} query={query[:50]}")
                    data = await r.json(content_type=None)
                papers = []
                for item in data.get("data", []):
                    authors = item.get("authors", [])
                    author_str = ", ".join(a.get("name", "") for a in authors[:3])
                    if len(authors) > 3:
                        author_str += " et al."
                    ext_ids = item.get("externalIds", {})
                    pmid = f"PMID{ext_ids['PubMed']}" if "PubMed" in ext_ids else f"SS:{item.get('paperId','')[:8]}"
                    papers.append(
                        Paper(
                            title=item.get("title", "Untitled"),
                            authors=author_str,
                            journal=item.get("venue", ""),
                            year=item.get("year") or 2000,
                            pmid=pmid,
                            abstract=item.get("abstract", ""),
                            source="semantic_scholar",
                            url=item.get("openAccessPdf", {}).get("url", "") if item.get("openAccessPdf") else "",
                        )
                    )
                return papers
            except Exception as exc:
                logger.warning("Semantic Scholar search failed: %s", exc)
                return []


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder re-ranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """Stage-2 re-ranking using a cross-encoder over (query, title+abstract) pairs."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(
                settings.cross_encoder_model,
                device=settings.embedding_device,
            )
            logger.info("Cross-encoder loaded: %s", settings.cross_encoder_model)
        except ImportError:
            logger.warning("sentence-transformers not installed — cross-encoder disabled.")

    def rerank(self, query: str, papers: list[Paper]) -> list[Paper]:
        if self._model is None or not papers:
            # Assign mock scores linearly decaying
            for i, p in enumerate(papers):
                p.score = round(max(0.4, 0.95 - i * 0.04), 2)
            return papers

        # pairs = [(query, f"{p.title}. {p.abstract[:300]}") for p in papers] # debuggig - error with this line as some papers have abstract none
        pairs = [(query, f"{p.title}. {(p.abstract or '')[:300]}") for p in papers]
        scores = self._model.predict(pairs)
        for paper, score in zip(papers, scores):
            paper.score = round(float(score), 3)
        papers.sort(key=lambda p: p.score, reverse=True)
        return papers


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeBase (main public interface)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Aggregate knowledge retrieval interface used by the ReAct agent.

    Usage
    ─────
        kb = KnowledgeBase()
        kb.load()                              # call once at startup
        landscape = await kb.get_topic_landscape("synaptic plasticity STDP")
        papers    = await kb.search_publications("STDP computational model", complexity="introductory")
    """

    def __init__(self) -> None:
        self.faiss = FAISSStore()
        self.embedder = EmbeddingModel()
        self.topic_modeller = TopicModeller()
        self.reranker = CrossEncoderReranker()
        self.pubmed = PubMedFetcher()
        self.s2 = SemanticScholarFetcher()

    def load(self) -> None:
        """Initialise all models and vector stores (call once at startup)."""
        logger.info("Loading KnowledgeBase components…")
        self.faiss.load()
        self.embedder.load()
        self.topic_modeller.load()
        self.reranker.load()
        logger.info("KnowledgeBase ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_topic_landscape(self, query: str) -> list[Topic]:
        """
        Map the conceptual landscape of a research query.
        Combines FAISS retrieval → BERTopic clustering → tool annotation.
        """
        t0 = time.perf_counter()
        # 1. Retrieve candidate abstracts from FAISS
        vec = self.embedder.encode([query])
        faiss_hits = self.faiss.search(vec, k=40)
        abstracts = [h.get("abstract", "") for h in faiss_hits if h.get("abstract")]

        # 2. Topic modelling over retrieved corpus
        topics = self.topic_modeller.get_topics_for_query(query, abstracts)
        logger.debug("Topic landscape in %.2fs — %d topics", time.perf_counter() - t0, len(topics))
        return topics

    async def search_publications(
        self,
        query: str,
        complexity: str = "introductory",
        max_results: int | None = None,
    ) -> list[Paper]:
        """
        Retrieve and re-rank papers from FAISS + live APIs.

        complexity: introductory | intermediate | advanced
            Used to bias re-ranking toward review / methodology / primary research.
        """
        k = max_results or settings.final_top_k
        # logger.warning("search_publications : Value of K '%s'", k)

        # 1. Parallel fetch from all sources
        faiss_task = asyncio.create_task(self._faiss_papers(query))
        pubmed_task = asyncio.create_task(self.pubmed.search(query, max_results=settings.rerank_top_k // 2))
        s2_task = asyncio.create_task(self.s2.search(query, limit=settings.rerank_top_k // 2))

        faiss_papers, pubmed_papers, s2_papers = await asyncio.gather(
            faiss_task, pubmed_task, s2_task
        )

        # 2. Merge & deduplicate (Fix #6 — PMID-first deduplication)
        all_papers = self._deduplicate(faiss_papers + pubmed_papers + s2_papers)
        # logger.warning("all_papers : Paper found '%s'", len(all_papers))

        # Normalise None abstracts from live API responses
        for p in all_papers:
            if p.abstract is None:
                p.abstract = ""

        # 3. Bias query for complexity
        biased_query = self._bias_query(query, complexity)

        # 4. Cross-encoder re-ranking
        ranked = self.reranker.rerank(biased_query, all_papers)

        for p in ranked[:5]:
            print(f"  DEBUG score={p.score:.4f} source={p.source} | {p.title[:50]}")

        # logger.warning("p.score :'%s'", p)
        # logger.warning("settings.confidence_threshold :'%s'", settings.confidence_threshold)
        # 5. Filter by confidence threshold
        filtered = [p for p in ranked if p.score >= settings.confidence_threshold]
        # logger.warning("------------------------------------------------------------")
        # logger.warning("filtered : Paper found '%s'", filtered)
        # logger.warning("------------------------------------------------------------")
        return filtered[:k]

    def get_tools_for_topic(self, topic_name: str) -> list[str]:
        """Return tools from the registry relevant to a topic name."""
        return _infer_domain_tools(topic_name)

    def get_datasets_for_topic(self, topic_name: str) -> list[str]:
        """Return datasets relevant to a topic."""
        q = topic_name.lower()
        hits: list[str] = []
        if any(k in q for k in ("neuro", "brain", "synap", "hippoc", "spike")):
            hits += DATASET_REGISTRY["neuro"]
        if any(k in q for k in ("genome", "rna", "gene", "crispr", "seq")):
            hits += DATASET_REGISTRY["genomics"]
        if any(k in q for k in ("proteom", "mass spec", "peptide", "phospho")):
            hits += DATASET_REGISTRY["proteomics"]
        if any(k in q for k in ("protein", "structure", "fold")):
            hits += DATASET_REGISTRY["protein"]
        if any(k in q for k in ("image", "scan", "mri")):
            hits += DATASET_REGISTRY["imaging"]
        if any(k in q for k in ("nlp", "text", "language", "corpus")):
            hits += DATASET_REGISTRY["nlp"]
        if any(k in q for k in ("climate", "weather", "co2", "atmosphere")):
            hits += DATASET_REGISTRY["climate"]
        return hits[:4] or DATASET_REGISTRY["neuro"][:3]

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _faiss_papers(self, query: str) -> list[Paper]:
        vec = self.embedder.encode([query])
        hits = self.faiss.search(vec, k=settings.rerank_top_k)
        papers = []
        for h in hits:
            papers.append(
                Paper(
                    title=h.get("title", "Untitled"),
                    authors=h.get("authors", ""),
                    journal=h.get("journal", ""),
                    year=h.get("year", 2000),
                    pmid=h.get("pmid", ""),
                    abstract=h.get("abstract", ""),
                    score=h.get("_faiss_score", 0.0),
                    source="faiss",
                )
            )
        return papers

    @staticmethod
    def _deduplicate(papers: list[Paper]) -> list[Paper]:
        """
        Merge duplicate papers.

        Fix #6 — the original implementation keyed only on a 60-char title
        prefix, so the same paper from different sources with minor title
        formatting differences (dashes vs colons, capitalisation) passed
        through as distinct entries.

        Strategy (applied in order):
          1. PMID-first: two papers sharing a real PMID are always the same
             paper, regardless of title formatting. "Real" means the PMID
             doesn't start with "SS:" (Semantic Scholar internal IDs).
          2. Normalised-title fallback for papers without a real PMID: strip
             punctuation and lower-case the full title before comparing, so
             "DESeq2: Moderated Estimation" and "DESeq2 - Moderated Estimation"
             collapse to the same entry.
        """
        import re as _re

        seen_pmids: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        def _norm_title(title: str) -> str:
            # Lower-case + collapse non-alphanumeric runs to a single space
            return _re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()

        for p in papers:
            # ── PMID deduplication ────────────────────────────────────────────
            has_real_pmid = bool(p.pmid) and not p.pmid.startswith("SS:")
            if has_real_pmid:
                if p.pmid in seen_pmids:
                    continue
                seen_pmids.add(p.pmid)

            # ── Title deduplication (fallback for SS: IDs and missing PMIDs) ──
            norm = _norm_title(p.title)
            if norm in seen_titles:
                continue
            seen_titles.add(norm)

            unique.append(p)

        return unique

    @staticmethod
    def _bias_query(query: str, complexity: str) -> str:
        bias = {
            "introductory": "introductory review overview tutorial",
            "intermediate": "methods protocol analysis comparison",
            "advanced": "primary research novel mechanism quantitative",
        }
        return f"{query} {bias.get(complexity, '')}"