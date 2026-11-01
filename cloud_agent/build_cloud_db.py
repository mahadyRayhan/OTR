import os
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=DB_DIR)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection_name = "cloud_architecture_vault"
try:
    client.delete_collection(name=collection_name)
except: pass

collection = client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
docs, metadatas, ids = [], [], []

for filename in os.listdir(KB_DIR):
    if filename.endswith(".md"):
        filepath = os.path.join(KB_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = content.split("\n## ")
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            text = ("## " + chunk) if i > 0 else chunk
            docs.append(text)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_chunk_{i}")

if docs:
    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"✅ Ingested {len(docs)} Cloud Architecture chunks into Vector DB!")