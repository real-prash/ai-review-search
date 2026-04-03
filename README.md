# AI Review Search

A semantic search tool for querying ~81,000 AI-related app reviews. Built for internal stakeholders (support, product managers, analysts) to explore user sentiment around AI features.

## How It Works

### 1. Data Ingestion (`pipeline.py`)

A memory-safe batch pipeline processes `reviews_dataset.csv.gz` (~3.3 million rows) without loading the entire file into memory.

- Reads the gzipped CSV in streaming batches using `gzip` + `csv.DictReader`
- Dispatches batches to a `ThreadPoolExecutor` (MapReduce pattern)
- Each worker runs a sentiment analysis function (mock → replace with real LLM)
- Results are written incrementally to `sentiment_results.jsonl.gz` as batches complete

### 2. AI Review Classification + Vector DB (`trials.ipynb`)

A LangGraph pipeline identifies reviews that mention AI features and stores them for semantic search.

**Graph architecture:**
```
START → stream_and_filter → embed_and_store → summarize_run → END
```

- **`stream_and_filter`** — Streams the CSV in chunks of 5,000 rows using `pandas`. Applies a regex keyword filter (25 AI-related terms: `ai`, `llm`, `chatbot`, `gpt`, `autocomplete`, etc.) with zero API calls. Yields ~81,000 matching reviews from 3.3M (~2.5%).

- **`embed_and_store`** — Embeds matching reviews with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors, runs locally on CPU). Stores them in a local **Chroma** vector database (`chroma_reviews_db/`) with full metadata: rating, timestamp, language, app version, thumbs-up count.

- **`summarize_run`** — Prints a run summary (rows scanned, AI reviews found, DB size).

**Models used:**
- Planner: `llama-3.3-70b-versatile` via Groq (structured outputs, report synthesis)
- Worker: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq (with `.with_retry()` for rate limit handling)

### 3. Search App (`search_app.py`)

A Flask web app that lets stakeholders query the reviews via natural language.

- Query is embedded using **FastEmbed** (`all-MiniLM-L6-v2`, ONNX-based — no PyTorch)
- Similarity search runs against **Qdrant Cloud** (81,000 vectors)
- Optional filters: minimum star rating, language
- Returns top-K results with similarity score, rating, language, and date

### 4. Deployment

- **Vector DB**: Qdrant Cloud (free tier) — migrated from local Chroma via `migrate_to_qdrant.py`
- **Web app**: Deployed on Railway (auto-deploys from GitHub on push)
- **Embeddings at query time**: FastEmbed (ONNX runtime, ~50MB vs ~1.5GB for PyTorch)

---

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| LLMs | Groq (Llama 3.3 70B + Llama 4 Scout 17B) |
| Batch ingestion | Python `gzip`, `csv`, `ThreadPoolExecutor` |
| Keyword filter | Python `re` (compiled regex) |
| Embeddings (ingestion) | `sentence-transformers/all-MiniLM-L6-v2` |
| Embeddings (query) | FastEmbed (ONNX) |
| Local vector DB | Chroma (SQLite-backed) |
| Cloud vector DB | Qdrant Cloud |
| Web framework | Flask + Gunicorn |
| Deployment | Railway |

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ingestion pipeline (notebook)
jupyter notebook trials.ipynb
# Execute cells 1–6 in order

# 3. Start the search app (uses local Chroma DB)
python search_app.py
# Open http://localhost:5000
```

## Deploying to Production

```bash
# 1. Migrate local Chroma → Qdrant Cloud (run once after ingestion)
QDRANT_URL=https://your-cluster.qdrant.io \
QDRANT_API_KEY=your-key \
python migrate_to_qdrant.py

# 2. Push to GitHub
git push origin main

# 3. On Railway: add environment variables
#    QDRANT_URL  = https://your-cluster.qdrant.io
#    QDRANT_API_KEY = your-key
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | Production | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Production | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | No | `ai_reviews` | Collection name |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed model |
| `PORT` | No | `5000` | Web server port |
