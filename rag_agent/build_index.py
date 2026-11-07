"""
build_index.py — Build the ChromaDB index from offline PDF papers.

Run from the project root:
    python rag_agent/build_index.py

Scans rag_agent/knowledge_base/*.pdf, extracts and chunks the text,
and adds every chunk to the ChromaDB collection (which handles embedding
internally via PubMedBERT through SentenceTransformerEmbeddingFunction).
No manual embedding step — ChromaDB owns that.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import re
import sys
from pathlib import Path

# ── path setup — works whether invoked as `python rag_agent/build_index.py`
#    from the project root or as `python build_index.py` from rag_agent/ ───────
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from rag_agent.knowledge_base import KnowledgeBase

# ── tunables ─────────────────────────────────────────────────────────────────
KB_DIR          = THIS_DIR / "knowledge_base"   # folder containing the PDFs
OA_DIR          = KB_DIR / "oa_corpus"          # open-access full texts + manifest.json
CHUNK_SIZE      = 350   # words per chunk
CHUNK_OVERLAP   = 60    # words of overlap between adjacent chunks
MIN_CHUNK_WORDS = 80    # discard very short chunks (headers, page numbers, etc.)


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
    # RAG_PDF_EXTRACTOR=fitz skips pymupdf4llm, whose Tesseract OCR pass can
    # crash the process on some PDFs.
    if os.environ.get("RAG_PDF_EXTRACTOR", "").lower() != "fitz":
        try:
            import pymupdf4llm  # type: ignore
            return pymupdf4llm.to_markdown(str(path))
        except ImportError:
            pass

    try:
        import fitz  # type: ignore  (PyMuPDF)
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass

    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass

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
    characters of the PDF text (the title page). Falls back to the filename
    stem when a field cannot be detected.
    """
    # Use a wider window for year detection (year often not on first page)
    header      = text[:1200]
    year_window = text[:8000]
    lines = [l.strip() for l in header.splitlines() if l.strip()]

    # Strip markdown bold/italic/header markers from a line before matching
    def _clean_line(s: str) -> str:
        s = re.sub(r"^#+\s*", "", s)       # leading ## headers
        s = re.sub(r"\*+|_+", "", s)       # bold / italic
        return s.strip()

    # Patterns that mark non-title lines (journal headers, DOI lines, etc.)
    _skip_title = re.compile(
        r"(?:"
        r"^\s*(?:doi|http|www|vol\.?|volume|pages?|doi\.org|10\.\d{4})"  # starts with
        r"|advance\s+access"                   # "Advance Access publication …"
        r"|received\s+on"                      # "Received on May 29, …"
        r"|doi:\s*10\."                         # inline "doi: 10."
        r"| doi\s+10\."                         # inline " doi 10."
        r"|\b10\.\d{4}/"                        # bare DOI anywhere in line
        r")",
        re.IGNORECASE,
    )
    title = filename_stem.replace("-", " ").replace("_", " ")
    for line in lines[:16]:
        cleaned = _clean_line(line)
        if (
            len(cleaned) > 18
            and not cleaned.isupper()
            and not re.match(r"^\d", cleaned)
            and not _skip_title.search(cleaned)
        ):
            title = cleaned
            break

    # Year: four-digit year in range 2000–2026; broaden to handle "©2017", "(2016)"
    year_m = re.search(r"(?<!\d)(200\d|201\d|202[0-6])(?!\d)", year_window)
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    pdfs = sorted(KB_DIR.glob("*.pdf"))
    oa_manifest_path = OA_DIR / "manifest.json"
    oa_manifest = json.loads(oa_manifest_path.read_text()) if oa_manifest_path.exists() else {}
    if not pdfs and not oa_manifest:
        print(f"No PDF files found in {KB_DIR}")
        print("Place your paper PDFs in that folder and re-run.")
        return

    print(f"Found {len(pdfs)} PDF(s) in {KB_DIR}\n")

    print("Loading KnowledgeBase (ChromaDB + PubMedBERT)...")
    kb = KnowledgeBase()
    kb.load()

    # Wipe existing collection so rebuilds are clean and idempotent
    print("Resetting ChromaDB collection...")
    kb.chroma.reset()
    kb.chroma.load()
    print()

    all_texts:     list[str]       = []
    all_metadatas: list[dict]      = []
    all_ids:       list[str]       = []

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

        stem = pdf_path.stem[:28]
        for i, chunk in enumerate(chunks):
            pmid = f"LOCAL:{stem}:c{i}"
            # Store "title + chunk" as the document so ChromaDB embeds both
            # context layers; keep the plain chunk text in metadata as abstract.
            all_texts.append(f"{meta['title']} {chunk}")
            all_metadatas.append({
                "title":    meta["title"],
                "authors":  meta["authors"],
                "journal":  meta["journal"],
                "year":     meta["year"],
                "pmid":     pmid,
                "abstract": chunk,
                "url":      "",
                "source":   "offline_pdf",
            })
            all_ids.append(pmid)

        print()

    # ── Open-access full texts (journal_fixes/scripts/collect_oa_papers.py) ──
    # Metadata comes from Europe PMC, not from heuristics, so every chunk
    # carries a real title, year, journal, PMID and citable URL.
    for pmcid, rec in sorted(oa_manifest.items()):
        txt_path = OA_DIR / f"{pmcid}.txt"
        if not txt_path.exists():
            continue
        text = clean_text(txt_path.read_text(encoding="utf-8"))
        chunks = chunk_text(text)
        print(f"[OA] {pmcid}  {len(chunks):3d} chunks  {str(rec.get('title'))[:70]}")
        for i, chunk in enumerate(chunks):
            cid = f"PMC:{pmcid}:c{i}"
            all_texts.append(f"{rec.get('title') or ''} {chunk}")
            all_metadatas.append({
                "title":    rec.get("title") or "",
                "authors":  "",
                "journal":  rec.get("journal") or "",
                "year":     int(rec.get("year") or 0),
                "pmid":     str(rec.get("pmid") or cid),
                "abstract": chunk,
                "url":      rec.get("url") or "",
                "source":   "europepmc_oa",
            })
            all_ids.append(cid)

    if not all_texts:
        print("No content could be extracted from any PDF. Check PDF readability.")
        return

    total = len(all_texts)
    print(f"Adding {total} chunks to ChromaDB (embedding handled internally)...")
    kb.chroma.add(all_texts, all_metadatas, all_ids)

    print(f"\n✓ Done — {total} chunks from {len(pdfs)} PDF(s) indexed.")
    print(f"  ChromaDB collection size: {kb.chroma.count()}")
    print("  Run the main application to query the knowledge base.")


if __name__ == "__main__":
    main()
