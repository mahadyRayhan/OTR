import os
import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup local ChromaDB folder
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=DB_DIR)

# 2. Use the local embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Create or reset the collection
collection_name = "scanpy_code_vault"
try:
    client.delete_collection(name=collection_name)
except:
    pass
collection = client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)

# 4. Read .ipynb and .pdf files from knowledge_base and chunk them
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
docs, metadatas, ids = [], [], []

for filename in sorted(os.listdir(KB_DIR)):
    filepath = os.path.join(KB_DIR, filename)

    # ── Jupyter notebooks ─────────────────────────────────────────────────────
    if filename.endswith(".ipynb"):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                nb_data = json.load(f)
            except json.JSONDecodeError:
                print(f"  ✘ Skipping {filename}: Invalid JSON")
                continue

        # Group each markdown cell with the code cell that follows it,
        # giving the DB both the explanation and the implementation as one chunk.
        current_markdown = ""
        chunk_index = 0
        for cell in nb_data.get("cells", []):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            if cell_type == "markdown":
                current_markdown += "\n" + source
            elif cell_type == "code":
                combined_chunk = f"### Explanation:\n{current_markdown.strip()}\n\n### Python Code:\n```python\n{source}\n```"
                docs.append(combined_chunk)
                metadatas.append({"source": filename, "chunk_index": chunk_index})
                ids.append(f"{filename}_chunk_{chunk_index}")
                chunk_index += 1
                current_markdown = ""
        print(f"  ✔ {filename}: {chunk_index} chunks")

    # ── PDFs ──────────────────────────────────────────────────────────────────
    elif filename.endswith(".pdf"):
        full_text = ""
        try:
            try:
                import pymupdf4llm
                full_text = pymupdf4llm.to_markdown(filepath)
            except ImportError:
                pass
            if not full_text:
                try:
                    import fitz
                    pdf_doc = fitz.open(filepath)
                    full_text = "\n".join(page.get_text() for page in pdf_doc)
                    pdf_doc.close()
                except ImportError:
                    pass
            if not full_text:
                try:
                    import pypdf
                    reader = pypdf.PdfReader(filepath)
                    full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    pass
            if not full_text:
                try:
                    import pdfplumber
                    with pdfplumber.open(filepath) as pdf:
                        full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                except ImportError:
                    pass
            if not full_text:
                print(f"  ✘ {filename}: No PDF library available. Install: pip install pypdf")
            else:
                chunk_size, overlap, chunk_index = 800, 100, 0
                i = 0
                while i < len(full_text):
                    chunk = full_text[i:i + chunk_size].strip()
                    if len(chunk) > 50:
                        docs.append(chunk)
                        metadatas.append({"source": filename, "chunk_index": chunk_index})
                        ids.append(f"{filename}_chunk_{chunk_index}")
                        chunk_index += 1
                    i += chunk_size - overlap
                print(f"  ✔ {filename}: {chunk_index} chunks")
        except Exception as e:
            print(f"  ✘ {filename}: PDF error — {e}")

# 5. Insert into Database
if docs:
    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"\n✅ Successfully ingested {len(docs)} total chunks into Vector DB!")
else:
    print("❌ No .ipynb or .pdf files found in the knowledge_base folder.")
