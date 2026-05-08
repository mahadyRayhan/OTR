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

# 4. Read the .ipynb files and chunk them intelligently
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
docs, metadatas, ids = [], [], []

for filename in os.listdir(KB_DIR):
    if filename.endswith(".ipynb"):
        filepath = os.path.join(KB_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                nb_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {filename}: Invalid JSON")
                continue

        # We will group a Markdown cell with the Code cell that immediately follows it.
        # This gives the Vector DB the "Theory" + the "Implementation" as one chunk.
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
                # Combine the accumulated markdown explanation with the python code
                combined_chunk = f"### Explanation:\n{current_markdown.strip()}\n\n### Python Code:\n```python\n{source}\n```"
                
                doc_id = f"{filename}_chunk_{chunk_index}"
                docs.append(combined_chunk)
                metadatas.append({"source": filename, "chunk_index": chunk_index})
                ids.append(doc_id)
                
                chunk_index += 1
                current_markdown = "" # Reset markdown for the next code block

# 5. Insert into Database
if docs:
    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"✅ Successfully ingested {len(docs)} Notebook code chunks into Vector DB!")
else:
    print("❌ No .ipynb files found in the knowledge_base folder.")