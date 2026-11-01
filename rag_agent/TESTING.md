# OnTimeRecommender — Quick Start & Testing Guide

## Prerequisites

* Python 3.9+
* Gemini API key (or Claude / OpenAI / Ollama)
* Semantic Scholar API key (optional but recommended)

---

## 1. Environment Setup

```bash
# Create and activate virtualenv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings \
    python-dotenv aiohttp google-genai structlog rich \
    numpy sentence-transformers faiss-cpu
```

---

## 2. Configure `.env`

Minimum required to run with Gemini:

```bash
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your-key-here
NCBI_EMAIL=your@email.com
SEMANTIC_SCHOLAR_API_KEY=your-s2-key-here   # optional
```

To switch providers later — no code changes needed:

```bash
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...

# or
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Verify your key is loaded:

```bash
python -c "from config import settings; print('Gemini key ends:', settings.gemini_api_key[-6:])"
python -c "from config import settings; print('Model:', settings.gemini_model)"
```

---

## 3. Build the Knowledge Base

Run once before starting the server:

```bash
python build_index.py
```

Expected output:

```
Using S2 API key: ...abc123
[1/8] Fetching: synaptic plasticity STDP hippocampus
  → 12 papers from Semantic Scholar
...
Total unique papers after dedup: 104
Embedding 104 papers in small batches...
Done — indexed 104 papers into ./data/
```

This creates `./data/faiss_index` and `./data/faiss_metadata.pkl`.
The PubMedBERT model (~440MB) downloads automatically on first run.

---

## 4. Start the Server

```bash
python main.py --log-level DEBUG
```

Expected startup:

```
INFO  Loading Knowledge Discovery Agent...  provider=gemini
INFO  FAISS index loaded: 104 vectors
INFO  Embedding model loaded: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
INFO  Cross-encoder loaded: cross-encoder/ms-marco-MiniLM-L-6-v2
INFO  Ready.  provider=gemini  model=(default)
```

---

## 5. Test All Endpoints

### Health & Readiness

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/readiness
```

Expected:

```json
{"status": "ok", "timestamp": 1234567890.0}
{"status": "ready", "provider": "gemini"}
```

---

### Full Search

```bash
# Advanced complexity
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR base editing off-target effects", "complexity": "advanced"}' \
  | python -m json.tool

# Introductory complexity
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is synaptic plasticity", "complexity": "introductory"}' \
  | python -m json.tool

# Intermediate complexity
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "single cell RNA sequencing tumor microenvironment", "complexity": "intermediate"}' \
  | python -m json.tool
```

What to check in the response:

* `topics` — meaningful names, not stopwords like "The And Of"
* `publications` — ranked papers with non-empty `relevance` fields
* `react_trace` — real Gemini thoughts, not fallback ("Starting by mapping...")
* `handoff` — contains `topics[]`, `tools[]`, `authors[]` as full arrays

---

### Topics Only (fast — no paper retrieval)

```bash
curl -X POST http://localhost:8000/api/topics \
  -H "Content-Type: application/json" \
  -d '{"query": "protein structure prediction AlphaFold"}' \
  | python -m json.tool
```

---

### Tools & Datasets Registry

```bash
curl http://localhost:8000/api/tools/CRISPR | python -m json.tool
curl http://localhost:8000/api/tools/single-cell | python -m json.tool
curl http://localhost:8000/api/tools/protein-structure | python -m json.tool
```

---

### Multi-Turn Chat

```bash
# Step 1 — create a session
SESSION=$(curl -s http://localhost:8000/api/session/new \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session ID: $SESSION"

# Step 2 — first message (triggers a full agent search)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"tell me about CRISPR base editing\", \"complexity\": \"advanced\"}" \
  | python -m json.tool

# Step 3 — follow-up (answered from session context, no new search)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"what are the main off-target risks?\", \"complexity\": \"advanced\"}" \
  | python -m json.tool

# Step 4 — new topic (triggers a fresh agent search)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"now tell me about AlphaFold protein folding\", \"complexity\": \"intermediate\"}" \
  | python -m json.tool
```

What to check:

* Step 2 `agent_payload` is populated (search was triggered)
* Step 3 `agent_payload` is null (follow-up used session context)
* Step 3 `reply` references content from Step 2
* Step 4 `agent_payload` is populated again (new topic triggered new search)

---

### Provider Management

```bash
# List all providers and active one
curl http://localhost:8000/api/providers | python -m json.tool

# Hot-swap to a different provider (no restart needed)
curl -X POST http://localhost:8000/api/providers/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini", "model": "gemini-2.5-flash"}' \
  | python -m json.tool
```

---

## 6. Common Issues

| Symptom                                       | Cause                                                     | Fix                                                                                                                                                                                    |
| --------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Planner call failed: 404`                  | Wrong model name for your API key                         | Run `python -c "from google import genai; from config import settings; [print(m.name) for m in genai.Client(api_key=settings.gemini_api_key).models.list()]"`to see available models |
| `Planner call failed: Unterminated string`  | `max_tokens`too low for Gemini 2.5                      | Increase `max_tokens`in `react_agent.py` `_plan()`to 1024                                                                                                                        |
| Topic names are stopwords ("The And Of")      | BERTopic fitting on small corpus without stopword removal | Expected with small index — rebuild with more queries in `build_index.py`                                                                                                           |
| `RuntimeError: Embedding model unavailable` | `sentence-transformers`not installed                    | `pip install sentence-transformers`                                                                                                                                                  |
| `segmentation fault`on macOS Python 3.9     | PyTorch threading bug                                     | Add `os.environ["OMP_NUM_THREADS"] = "1"`at top of the file that crashes                                                                                                             |
| S2 returns 0 papers                           | Rate limiting without API key                             | Add `SEMANTIC_SCHOLAR_API_KEY`to `.env`                                                                                                                                            |
| `topic_n_gram_range`parse error             | Pydantic can't parse tuple from `.env`                  | Comment out `TOPIC_N_GRAM_RANGE`in `.env`— default is fine                                                                                                                        |

---

## 7. Add More Papers to the Index

Edit the `QUERIES` list in `build_index.py` and re-run:

```python
QUERIES = [
    "your new topic here",
    "another research area",
    # ... existing queries
]
```

```bash
python build_index.py
# restart server after — it reloads the index on startup
python main.py --log-level DEBUG
```

---

## 8. Switch Provider After NCBI Key Arrives

Just add to `.env` and restart — no code changes:

```bash
NCBI_API_KEY=your-ncbi-key
NCBI_EMAIL=your@university.edu
```

PubMed results will be layered in automatically alongside FAISS and Semantic Scholar.
