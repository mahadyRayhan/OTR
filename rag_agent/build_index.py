"""
build_index.py — Build the FAISS index from offline PDF papers.

Run from the project root:
    python rag_agent/build_index.py

Scans rag_agent/knowledge_base/*.pdf, extracts and chunks the text,
embeds every chunk, and saves the result to the FAISS store.
No network calls — fully offline.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import sys
import numpy as np
from pathlib import Path

# ── path setup — works whether invoked as `python rag_agent/build_index.py`
#    from the project root or as `python build_index.py` from rag_agent/ ───────
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from rag_agent.knowledge_base import KnowledgeBase, Paper

# ── tunables ─────────────────────────────────────────────────────────────────
KB_DIR        = THIS_DIR / "knowledge_base"   # folder containing the PDFs
CHUNK_SIZE    = 350   # words per chunk
CHUNK_OVERLAP = 60    # words of overlap between adjacent chunks
MIN_CHUNK_WORDS = 80  # discard very short chunks (headers, page numbers, etc.)
EMBED_BATCH   = 4     # keep small to avoid macOS / Python 3.9 segfault


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(path: Path) -> str:
    """
    Return the full text of a PDF.

    Library preference (best quality first):
      1. pymupdf4llm  — LLM-optimised extraction, handles columns/tables well
      2. fitz (PyMuPDF) — fast, accurate
      3. pypdf         — pure-Python fallback
      4. pdfplumber    — last resort
    """
    # 1. pymupdf4llm (best for academic papers)
    try:
        import pymupdf4llm  # type: ignore
        return pymupdf4llm.to_markdown(str(path))
    except ImportError:
        pass

    # 2. PyMuPDF (fitz)
    try:
        import fitz  # type: ignore  (PyMuPDF)
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass

    # 3. pypdf
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass

    # 4. pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        raise RuntimeError(
            "No PDF library available. Install one:\n"
            "  pip install pymupdf4llm   (recommended)\n"
            "  pip install pypdf\n"
            "  pip install pdfplumber"
        )


def clean_text(text: str) -> str:
    """Remove hyphenation artifacts and collapse excessive whitespace."""
    text = re.sub(r"-\n\s*", "", text)       # rejoin hyphenated line-breaks
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal space
    text = re.sub(r"\n{3,}", "\n\n", text)    # at most one blank line
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """Sliding-window word chunker; returns non-empty chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def parse_metadata(text: str, filename_stem: str) -> dict:
    """
    Heuristically extract title, authors, journal, year from the first ~1200
    characters of the PDF text (the title page).  Falls back to the filename
    stem when a field cannot be detected.
    """
    header = text[:1200]
    lines = [l.strip() for l in header.splitlines() if l.strip()]

    # Title: first line longer than 20 chars that is not all-uppercase
    title = filename_stem.replace("-", " ").replace("_", " ")
    for line in lines[:8]:
        if len(line) > 20 and not line.isupper() and not re.match(r"^\d", line):
            title = line
            break

    # Year: four-digit year in range 2000–2026
    year_m = re.search(r"\b(200\d|201\d|202[0-6])\b", header)
    year = int(year_m.group(1)) if year_m else 0

    # Authors: first line that looks like a list of author names
    authors = ""
    for line in lines[1:10]:
        if re.search(
            r"[A-Z][a-z]+,\s+[A-Z]"     # "Smith, J" style
            r"|\bet\s+al\b"              # "et al"
            r"|[A-Z]\.\s+[A-Z][a-z]",   # "J. Smith" style
            line,
        ):
            authors = line[:150]
            break

    # Journal: substring match against known journal names
    journal = ""
    for kw in [
        "Nature", "Science", "Cell", "Bioinformatics", "Genome Biology",
        "Nucleic Acids", "PLOS", "eLife", "Scientific Reports", "PNAS",
        "BMC", "NAR", "Briefings",
    ]:
        if kw.lower() in header.lower():
            journal = kw
            break

    return {"title": title, "authors": authors, "journal": journal, "year": year}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper  (small batches to avoid macOS segfault)
# ─────────────────────────────────────────────────────────────────────────────

def embed_in_batches(model, texts: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    total = len(texts)
    n_batches = (total + EMBED_BATCH - 1) // EMBED_BATCH
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        print(
            f"  Embedding batch {i // EMBED_BATCH + 1}/{n_batches}"
            f" ({len(batch)} chunks)..."
        )
        vecs = model.encode(
            batch,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        parts.append(vecs)
    return np.vstack(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    pdfs = sorted(KB_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {KB_DIR}")
        print("Place your paper PDFs in that folder and re-run.")
        return

    print(f"Found {len(pdfs)} PDF(s) in {KB_DIR}\n")

    print("Loading KnowledgeBase (embedding model + FAISS)...")
    kb = KnowledgeBase()
    kb.load()
    print()

    all_papers: list[Paper] = []
    all_texts:  list[str]   = []

    for pdf_path in pdfs:
        print(f"[PDF] {pdf_path.name}")
        try:
            raw = extract_pdf_text(pdf_path)
        except Exception as exc:
            print(f"  ✗ Extraction failed: {exc}\n")
            continue

        text = clean_text(raw)
        word_count = len(text.split())
        if word_count < 200:
            print(f"  ✗ Too short after extraction ({word_count} words) — skipping\n")
            continue

        meta = parse_metadata(text, pdf_path.stem)
        print(f"  Title  : {meta['title'][:80]}")
        print(f"  Authors: {meta['authors'][:60] or '(not detected)'}")
        print(f"  Journal: {meta['journal'] or '(not detected)'}")
        print(f"  Year   : {meta['year'] or '(not detected)'}")
        print(f"  Words  : {word_count:,}")

        chunks = chunk_text(text)
        print(f"  Chunks : {len(chunks)}  (≈{CHUNK_SIZE} words, {CHUNK_OVERLAP}-word overlap)")

        for i, chunk in enumerate(chunks):
            pmid = f"LOCAL:{pdf_path.stem[:28]}:c{i}"
            all_papers.append(Paper(
                title=meta["title"],
                authors=meta["authors"],
                journal=meta["journal"],
                year=meta["year"],
                pmid=pmid,
                abstract=chunk,
                source="offline_pdf",
                url="",
            ))
            # embed "title + chunk" so retrieval picks up title-level keywords too
            all_texts.append(f"{meta['title']} {chunk}")

        print()

    if not all_papers:
        print("No content could be extracted from any PDF. Check PDF readability.")
        return

    print(f"Total chunks to embed and index: {len(all_papers)}\n")

    print("Embedding all chunks...")
    vectors = embed_in_batches(kb.embedder._model, all_texts)

    metadata = [
        {
            "title":    p.title,
            "authors":  p.authors,
            "journal":  p.journal,
            "year":     p.year,
            "pmid":     p.pmid,
            "abstract": p.abstract,
            "url":      p.url,
        }
        for p in all_papers
    ]

    print("\nAdding vectors to FAISS index...")
    kb.faiss.add(vectors, metadata)
    kb.faiss.save()

    print(f"\n✓ Done — {len(all_papers)} chunks from {len(pdfs)} PDF(s) indexed.")
    print(f"  FAISS total vectors: {kb.faiss._index.ntotal}")
    print("  Run the main application to query the knowledge base.")


if __name__ == "__main__":
    main()
