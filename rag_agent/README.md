
# OnTimeRecommender-Knowledge-Agent

A RAG-powered scientific literature discovery system with multi-step ReAct reasoning, PubMedBERT embeddings, BERTopic topic modelling, a cross-encoder re-ranking pipeline, and pluggable CoT/ToT/GoT reasoning strategies.

## Architecture

```
User Query
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                      ▼
KnowledgeDiscoveryAgent  (react_agent.py)         RAGAgent  (rag_agent.py)
    │   ┌─── THOUGHT: plan next retrieval step         │   ┌─── retrieve papers from KB
    │   ├─── ACTION:  get_topic_landscape /            │   ├─── run CoT / ToT / GoT reasoning
    │   │             search_publications /            │   ├─── LLM-as-judge evaluation
    │   │             get_tools                        │   └─── return synthesised answer
    │   └─── OBS:     results → loop or STOP          │
    │                                                  │
    ▼                                                  ▼
KnowledgeBase  (knowledge_base.py)          ReasoningEngine  (reasoning.py)
    ├── PubMedBERT embedder  →  FAISS            ├── ChainOfThought   (cot)
    ├── PubMed Entrez API    →  live fetch        ├── TreeOfThought    (tot)
    ├── Semantic Scholar API →  live fetch        └── GraphOfThought   (got)
    ├── BERTopic             →  topic clustering
    └── CrossEncoder         →  stage-2 re-ranking
    │
    ▼
RAGResponder  (rag_responder.py)
    └── LLM explanation layer → "Why relevant / Why for you"
    │
    ▼
FastAPI Response  (main.py)
```

All LLM calls are routed through `llm_provider.py`, making it trivial to swap
Claude → GPT → Gemini → Ollama without touching any agent code.

## File Structure

```
knowledge_discovery_agent/
├── logs/                       # Structured log output (auto-created)
├── data/                       # FAISS index + metadata (auto-created)
├── models/                     # BERTopic saved model (auto-created)
├── .env                        # Environment variables
├── .gitignore
├── README.md
├── requirements.txt
├── build_index.py              # One-time script to build FAISS index from S2 + PubMed
├── config.py                   # Pydantic Settings — single source of truth
├── llm_provider.py             # Provider abstraction: Claude · OpenAI · Gemini · Ollama
├── knowledge_base.py           # FAISS · PubMedBERT · BERTopic · APIs · CrossEncoder
├── react_agent.py              # ReAct orchestration loop (Thought→Action→Obs)
├── rag_responder.py            # LLM explanation layer + API payload serialiser
├── rag_agent.py                # RAG agent with CoT / ToT / GoT reasoning strategies
├── reasoning.py                # Pluggable reasoning plugins (CoT, ToT, GoT)
├── chatbot.py                  # Multi-turn session manager
├── main.py                     # FastAPI app · all endpoints · logging · Uvicorn launcher
├── provider_benchmark.py       # Head-to-head latency/cost comparison across providers
├── evaluations.py              # Eval harness: precision@k · topic recall · latency
└── evaluation_data.json        # Ground-truth eval cases
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Edit `.env`:

```bash
# Active provider
LLM_PROVIDER=gemini               # claude | openai | gemini | ollama
LLM_MODEL=gemini-2.5-flash        # leave blank to use provider default

# API keys — only fill in the provider you're using
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434

# Literature APIs
NCBI_API_KEY=...                  # free at ncbi.nlm.nih.gov/account
NCBI_EMAIL=your@email.com         # required by PubMed TOS
SEMANTIC_SCHOLAR_API_KEY=...      # optional, higher rate limits
```

Verify your key loaded correctly:

```bash
python -c "from config import settings; print(settings.gemini_api_key[-6:])"
python -c "from config import settings; print(settings.gemini_model)"
```

### 3. Build the knowledge base

Run once before starting the server. Fetches papers from Semantic Scholar
(and PubMed as fallback) and indexes them into FAISS:

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

The PubMedBERT model (~440MB) downloads automatically on first run.
Add more topics by editing the `QUERIES` list in `build_index.py` and re-running.

### 4. Start the server

```bash
python main.py                    # production
python main.py --reload           # development with auto-reload
python main.py --port 8080 --log-level DEBUG
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

## API Usage

### Search (ReAct pipeline)

Full single-turn search using the ReAct agent → cross-encoder re-ranking → LLM explanations:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "synaptic plasticity in hippocampal neurons", "complexity": "introductory"}'
```

Response includes `react_trace`, `topics`, `publications`, and `handoff`.

### RAG Agent (CoT / ToT / GoT)

Retrieves papers from the same KB but synthesises them into a single answer
using a configurable reasoning strategy:

```bash
# Chain of Thought — linear step-by-step reasoning
curl -X POST http://localhost:8080/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main off-target risks of CRISPR base editing?", "strategy": "cot", "depth": 2}' \
  | python -m json.tool

# Tree of Thought — parallel branches, picks the best
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR off-target detection methods", "strategy": "tot", "depth": 2}' \
  | python -m json.tool

# Graph of Thought — decomposes into sub-questions, merges answers
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "protein structure prediction methods", "strategy": "got", "depth": 3}' \
  | python -m json.tool
```

Response shape:

```json
{
  "answer":      "Synthesised answer text...",
  "quality":     "Score: 9/10 - Well grounded in retrieved literature.",
  "strategy":    "COT",
  "source_used": "Paper title 1 | Paper title 2 | ..."
}
```

**Strategy comparison:**

| Strategy | How it works                                  | Best for                                  | LLM calls (depth=3) |
| -------- | --------------------------------------------- | ----------------------------------------- | ------------------- |
| `cot`  | Answer → self-critique → synthesis          | Focused factual questions                 | 3 sequential        |
| `tot`  | 4 parallel branches → evaluate → synthesise | Open-ended questions with multiple angles | 6 (4 parallel + 2)  |
| `got`  | Decompose → answer sub-questions → merge    | Complex multi-part questions              | N+3                 |

**Depth guide:**

* `depth=1` — single pass, fastest
* `depth=2` — adds critique/evaluation pass
* `depth=3` — full reasoning with synthesis, most thorough

### Multi-turn Chat

```bash
# Create a session
SESSION=$(curl -s http://localhost:8000/api/session/new \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# First message — triggers a full agent search
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"synaptic plasticity STDP\", \"complexity\": \"advanced\"}"

# Follow-up — resolved from session context, no new search
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"tell me more about the Caporale paper\"}"
```

### Topic Landscape (fast)

```bash
curl -X POST http://localhost:8000/api/topics \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR base editing"}'
```

### Tools & Datasets Registry

```bash
curl http://localhost:8000/api/tools/hippocampal%20plasticity
curl http://localhost:8000/api/tools/single-cell
curl http://localhost:8000/api/tools/protein-structure
```

### Provider Management

```bash
# List all providers and active one
curl http://localhost:8000/api/providers

# Hot-swap provider at runtime (no restart needed)
curl -X POST http://localhost:8000/api/providers/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini", "model": "gemini-2.5-flash"}'

# Benchmark providers head-to-head
curl -X POST http://localhost:8000/api/providers/benchmark \
  -H "Content-Type: application/json" \
  -d '{"providers": ["claude", "openai", "gemini"], "component": "all"}'
```

---

## Reasoning Strategies

### Chain of Thought (CoT)

Linear reasoning — the model thinks step by step before giving a final answer.
At higher depths it adds a self-critique pass and a synthesis pass.

```
depth=1: Answer with explicit reasoning steps
depth=2: Answer → critique → improved answer
depth=3: Answer → critique → refinement → final synthesis
```

### Tree of Thought (ToT)

Explores multiple reasoning branches concurrently then picks the best.
Good for questions that benefit from considering multiple perspectives.

```
depth=1: 2 branches → evaluate → best answer
depth=2: 3 branches → evaluate → best answer
depth=3: 4 branches → evaluate → synthesise top-2 → final answer
```

### Graph of Thought (GoT)

Decomposes the query into sub-questions, answers each independently,
then merges at a final aggregation node. Best for complex multi-part queries.

```
depth=1: Decompose → 2 sub-answers → merge
depth=2: Decompose → 3 sub-answers → cross-link → merge
depth=3: Decompose → 4 sub-answers → cross-link → merge → polish
```

---

## Difference Between `/api/search` and `/api/rag`

|          | `/api/search`                         | `/api/rag`                        |
| -------- | --------------------------------------- | ----------------------------------- |
| Agent    | ReAct loop (Thought→Action→Obs)       | Direct retrieval + reasoning plugin |
| Output   | List of ranked papers + topics          | Single synthesised answer           |
| LLM role | Planner + paper explainer               | Reasoner + judge                    |
| Best for | Finding papers to read                  | Getting a direct answer             |
| Files    | `react_agent.py`+`rag_responder.py` | `rag_agent.py`+`reasoning.py`   |

Both use the same `KnowledgeBase` (FAISS + PubMed + Semantic Scholar + cross-encoder).

---

## Evaluations

```bash
python evaluations.py                    # all suites, precision@5
python evaluations.py --suite retrieval  # retrieval precision only
python evaluations.py --suite latency    # timing benchmarks
python evaluations.py --k 3             # precision@3
```

Results are saved to `logs/eval_<suite>_<timestamp>.json`.

---

## Extending the Knowledge Base

### Adding papers to FAISS

```python
from knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.load()

papers = [
    {
        "title": "Spike-timing dependent plasticity: a comprehensive overview",
        "authors": "Caporale & Dan",
        "journal": "Neuron",
        "year": 2008,
        "pmid": "PMID18702468",
        "abstract": "Full abstract text here...",
    }
]

texts = [f"{p['title']} {p['abstract']}" for p in papers]
vectors = kb.embedder.encode(texts)
kb.faiss.add(vectors, papers)
kb.faiss.save()
```

Or just add more queries to `build_index.py` and re-run it — easier for bulk indexing.

### Training BERTopic on your corpus

```python
from bertopic import BERTopic

docs = [...]  # list of abstracts
model = BERTopic(min_topic_size=10, n_gram_range=(1, 3))
model.fit(docs)
model.save("./models/bertopic_model")
```

---

## Common Issues

| Symptom                                       | Cause                                    | Fix                                                                                                                                                             |
| --------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Planner call failed: 404`                  | Model not available on your API key      | Run `python -c "from google import genai; from config import settings; [print(m.name) for m in genai.Client(api_key=settings.gemini_api_key).models.list()]"` |
| `Planner call failed: Unterminated string`  | `max_tokens`too low for Gemini 2.5     | Increase `max_tokens`in `react_agent.py` `_plan()`to 1024                                                                                                 |
| Topic names are stopwords ("The And Of")      | BERTopic fitting on small corpus         | Rebuild index with more queries in `build_index.py`                                                                                                           |
| `RuntimeError: Embedding model unavailable` | `sentence-transformers`not installed   | `pip install sentence-transformers`                                                                                                                           |
| `segmentation fault`on macOS Python 3.9     | PyTorch threading bug                    | Ensure `os.environ["OMP_NUM_THREADS"] = "1"`is at top of `main.py`and `build_index.py`                                                                    |
| S2 returns 0 papers                           | Rate limiting                            | Add `SEMANTIC_SCHOLAR_API_KEY`to `.env`                                                                                                                     |
| `topic_n_gram_range`parse error             | Pydantic can't parse tuple from `.env` | Comment out `TOPIC_N_GRAM_RANGE`in `.env`                                                                                                                   |
| `/api/rag`returns empty answer              | No papers above confidence threshold     | Lower `CONFIDENCE_THRESHOLD`in `.env`or broaden the query                                                                                                   |

---

## API Reference

| Method   | Path                         | Description                                          |
| -------- | ---------------------------- | ---------------------------------------------------- |
| `GET`  | `/api/health`              | Liveness probe                                       |
| `GET`  | `/api/readiness`           | Readiness probe (models loaded)                      |
| `POST` | `/api/search`              | Full ReAct search → ranked papers + topics          |
| `POST` | `/api/rag`                 | RAG agent with CoT / ToT / GoT → synthesised answer |
| `POST` | `/api/chat`                | Multi-turn chat message                              |
| `GET`  | `/api/session/new`         | Create chat session                                  |
| `POST` | `/api/topics`              | Topic landscape only (fast)                          |
| `GET`  | `/api/tools/{topic}`       | Tool & dataset registry                              |
| `GET`  | `/api/providers`           | List providers and active one                        |
| `POST` | `/api/providers/switch`    | Hot-swap LLM provider at runtime                     |
| `POST` | `/api/providers/benchmark` | Head-to-head provider comparison                     |

---

## Technology Stack

| Component       | Technology                                                                 |
| --------------- | -------------------------------------------------------------------------- |
| LLM / Planner   | Claude · GPT · Gemini · Ollama (switchable via `llm_provider.py`)     |
| Reasoning       | Chain of Thought · Tree of Thought · Graph of Thought (`reasoning.py`) |
| Embeddings      | PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)          |
| Vector Store    | FAISS (faiss-cpu)                                                          |
| Topic Modelling | BERTopic + UMAP + HDBSCAN                                                  |
| Re-Ranking      | Cross-encoder (ms-marco-MiniLM-L-6-v2)                                     |
| Literature APIs | PubMed Entrez, Semantic Scholar                                            |
| Web Framework   | FastAPI + Uvicorn                                                          |
| Logging         | structlog                                                                  |
