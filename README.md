# AI Review Search

A semantic search tool that lets internal stakeholders (support, product managers, analysts) query ~81,000 AI-related app reviews using natural language. Built on LangGraph, Qdrant Cloud, and Flask, deployed on Railway.

---

## End-to-End Architecture

```
reviews_dataset.csv.gz  (3.3M rows)
          │
          ▼
┌─────────────────────────────┐
│  1. Batch Ingestion         │  pipeline.py
│     ThreadPoolExecutor      │
│     Mock sentiment agents   │
└────────────┬────────────────┘
             │ sentiment_results.jsonl.gz
             ▼
┌─────────────────────────────┐
│  2. AI Review Pipeline      │  trials.ipynb
│     LangGraph (3-node DAG)  │
│     Keyword filter          │
│     Embed → Chroma (local)  │
└────────────┬────────────────┘
             │ chroma_reviews_db/ (297 MB, 80,992 docs)
             ▼
┌─────────────────────────────┐
│  3. Migration               │  migrate_to_qdrant.py
│     Chroma → Qdrant Cloud   │
└────────────┬────────────────┘
             │ Qdrant Cloud collection: ai_reviews
             ▼
┌─────────────────────────────┐
│  4. Search App              │  search_app.py
│     Flask + FastEmbed       │
│     Similarity threshold    │
│     Deduplication           │
└────────────┬────────────────┘
             │
             ▼
      Railway (public URL)
      templates/index.html
```

---

## Phase 1 — Batch Ingestion (`pipeline.py`)

Processes `reviews_dataset.csv.gz` (3.3M rows) without loading the full file into memory.

- **Reader**: `batch_reader()` — streams gzipped CSV using `gzip.open` + `csv.DictReader`, yields batches of 5,000 rows
- **Orchestrator**: `ThreadPoolExecutor` (MapReduce pattern) — submits each batch as a concurrent task. Thread-based because real LLM calls are I/O-bound
- **Worker**: `mock_ai_sentiment_agent()` — placeholder for an LLM sentiment classifier; returns sentiment + confidence per review
- **Sink**: results written incrementally to `sentiment_results.jsonl.gz` as each batch completes (`as_completed`), so no progress is lost on interruption

---

## Phase 2 — AI Review Classification + Embeddings (`trials.ipynb`)

A **LangGraph pipeline** identifies reviews mentioning AI features and stores them as vectors for semantic search.

### Graph

```
START → stream_and_filter → embed_and_store → summarize_run → END
```

### Node 1: `stream_and_filter`

- Streams CSV in chunks of 5,000 rows via `pandas.read_csv(..., chunksize=5000, compression='gzip')`
- Applies a compiled regex (`AI_KEYWORDS`) against the `message` field — 25 terms including: `ai`, `llm`, `chatbot`, `gpt`, `gemini`, `copilot`, `neural network`, `voice assistant`, `natural language`, `deep learning`
- Zero API calls — pure Python, processes 3.3M rows in ~3 minutes
- **Yield**: ~81,000 matching reviews (~2.5% of dataset)

### Node 2: `embed_and_store`

- Initialises `HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")` — 384-dimensional vectors, runs on CPU
- Converts each review to a `Document` with full metadata: `key`, `score` (star rating), `timestamp`, `language`, `app_version`, `thumbs_up_count`
- Stores in **Chroma** (local SQLite-backed vector DB) in sub-batches of 5,000 using `Chroma.from_documents()` then `.add_documents()`
- Output: `chroma_reviews_db/` (~297 MB on disk)

### Node 3: `summarize_run`

- Prints total rows scanned, AI reviews found, percentage yield, DB document count

### LLMs configured (for future LLM-classification upgrade)

| Role | Model | Via |
|------|-------|-----|
| Planner | `llama-3.3-70b-versatile` | Groq |
| Worker | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq |

Worker has `.with_retry(stop_after_attempt=8, wait_exponential_jitter=True)` to handle Groq rate limits automatically.

---

## Phase 3 — Migration to Qdrant Cloud (`migrate_to_qdrant.py`)

One-time script that copies all documents from local Chroma to Qdrant Cloud.

- Reads local Chroma in integer-offset pages of 500 documents
- Creates the `ai_reviews` collection in Qdrant (384-dim, cosine distance)
- Uploads via `QdrantVectorStore.add_documents()` in batches
- **Result**: 80,992 documents in Qdrant Cloud — the production vector store

---

## Phase 4 — Search App (`search_app.py`)

Flask app that serves the stakeholder UI and handles search queries.

### Startup (lazy)

The embedding model and Qdrant connection are initialised on the **first search request**, not at module load. This lets Railway's healthcheck pass instantly on `GET /`.

### Search logic (`GET /search?q=...`)

1. Fetch `MAX_RESULTS × 4` (200) candidates from Qdrant via `similarity_search_with_score`
2. **Similarity filter**: drop results with cosine distance > `SIMILARITY_THRESHOLD` (default `0.40` — roughly 60%+ semantic match). Controlled via `SIMILARITY_THRESHOLD` env var
3. **Deduplication**: normalise each message (lowercase + collapsed whitespace), skip exact duplicates
4. Return up to `MAX_RESULTS` (50) unique, relevant results as JSON

### Embeddings at query time

Uses **FastEmbed** (`fastembed` + `langchain_community.embeddings.FastEmbedEmbeddings`) with the same `all-MiniLM-L6-v2` model used during ingestion. FastEmbed uses ONNX runtime instead of PyTorch — reduces the Docker image from ~2 GB to ~300 MB.

---

## Phase 5 — Frontend (`templates/index.html`)

Single-page vanilla JS app.

- **Search bar** centered on page, Enter key supported
- **Sample query chips**: 8 curated queries based on real review patterns — click to run instantly
- **Client-side pagination**: server returns up to 50 results; frontend shows 15 at a time with a **Load more** button (no extra API calls)
- **Result cards**: show match %, star rating, date, thumbs-up count; long reviews are clamped to 3 lines with a Show more toggle
- Fully responsive / mobile-friendly

---

## Running Locally

```bash
# Install deps
pip install flask gunicorn fastembed langchain-core langchain-community \
            langchain-qdrant qdrant-client python-dotenv

# Run ingestion pipeline (Jupyter)
jupyter notebook trials.ipynb    # run cells 1–6

# Start search app against local Chroma (no Qdrant needed)
# Add langchain-chroma and chromadb to the install above, then:
python search_app.py
# → http://localhost:5000
```

## Deploying to Production

```bash
# 1. Migrate local Chroma → Qdrant Cloud (run once)
QDRANT_URL=https://your-cluster.qdrant.io \
QDRANT_API_KEY=your-key \
python migrate_to_qdrant.py

# 2. Push to GitHub
git push origin main

# 3. Connect repo to Railway, set env vars (see below)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | Yes (prod) | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes (prod) | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | No | `ai_reviews` | Collection name |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed model |
| `SIMILARITY_THRESHOLD` | No | `0.40` | Max cosine distance (lower = stricter) |
| `PORT` | No | `5000` | Web server port (set automatically by Railway) |

## File Reference

| File | Purpose |
|------|---------|
| `pipeline.py` | Batch ingestion — streams CSV, mock sentiment agents, JSONL sink |
| `trials.ipynb` | LangGraph pipeline — keyword filter, embed, store in Chroma |
| `migrate_to_qdrant.py` | One-time migration from local Chroma to Qdrant Cloud |
| `search_app.py` | Flask search backend — similarity filter, dedup, pagination |
| `templates/index.html` | Stakeholder search UI |
| `build.sh` | Pre-downloads embedding model at Railway build time |
| `Procfile` | Gunicorn start command for Railway/Render |
| `railway.json` | Railway deployment config |
