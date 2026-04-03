"""
AI Review Search — Flask app.

Vector DB backend is selected by environment variables:
  - Default (local dev): uses Chroma at CHROMA_PERSIST_DIR (./chroma_reviews_db)
  - Production:          set QDRANT_URL + QDRANT_API_KEY to use Qdrant Cloud

Embeddings use fastembed (ONNX, no PyTorch) for a small Docker image.
The vector store is initialised lazily on the first search request so the
healthcheck endpoint (GET /) responds immediately on startup.
"""
import os
from typing import Optional
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_reviews_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "langchain")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_reviews")
DEFAULT_TOP_K = 10
# Maximum unique results returned per query (frontend paginates within this pool).
MAX_RESULTS = 50

app = Flask(__name__)

# Lazily initialised on first search request
_vectorstore = None


def get_vectorstore():
    """Return the vector store, initialising it on first call."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.embeddings import FastEmbedEmbeddings

    print("Loading embedding model...")
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL environment variable is not set. "
            "Add it in your Railway dashboard under Variables."
        )

    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    print(f"Connecting to Qdrant Cloud at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    _vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )
    print("Vector store ready (qdrant).")

    return _vectorstore


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"results": [], "count": 0, "query": query})

    # Fetch a large pool so dedup + threshold filtering still leaves plenty
    fetch_k = MAX_RESULTS * 4

    try:
        vs = get_vectorstore()
        raw = vs.similarity_search_with_score(query, k=fetch_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    results = []
    seen = set()
    for doc, score in raw:
        # Deduplicate by normalised message text
        normalised = " ".join(doc.page_content.lower().split())
        if normalised in seen:
            continue
        seen.add(normalised)

        m = doc.metadata
        results.append({
            "message": doc.page_content,
            "score": m.get("score", ""),
            "language": m.get("language", ""),
            "timestamp": str(m.get("timestamp", ""))[:10],
            "app_version": m.get("app_version", ""),
            "thumbs_up": m.get("thumbs_up_count", 0),
            "similarity": round(float(score), 4),
        })
        if len(results) >= MAX_RESULTS:
            break

    return jsonify({"results": results, "count": len(results), "query": query})


def _build_filter(min_rating, language):
    # type: (str, str) -> Optional[dict]
    """Build a Chroma/Qdrant compatible metadata filter."""
    clauses = []
    if min_rating.isdigit():
        clauses.append({"score": {"$gte": int(min_rating)}})
    if language:
        clauses.append({"language": {"$eq": language}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
