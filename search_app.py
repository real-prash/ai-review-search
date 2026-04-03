"""
AI Review Search — Flask app.

Vector DB backend is selected by environment variables:
  - Default (local dev): uses Chroma at CHROMA_PERSIST_DIR (./chroma_reviews_db)
  - Production:          set QDRANT_URL + QDRANT_API_KEY to use Qdrant Cloud

Embedding: fastembed (ONNX-based, no PyTorch — keeps the Docker image small).
"""
import os
from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import FastEmbedEmbeddings

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

# ---------------------------------------------------------------------------
# App + vector store initialisation (once at startup)
# ---------------------------------------------------------------------------
app = Flask(__name__)

print("Loading embedding model...")
embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

if QDRANT_URL:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    print(f"Connecting to Qdrant Cloud at {QDRANT_URL} ...")
    _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(
        client=_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )
    BACKEND = "qdrant"
else:
    from langchain_chroma import Chroma

    print(f"Loading local Chroma DB from {CHROMA_PERSIST_DIR!r} ...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    BACKEND = "chroma"

print(f"Vector store ready ({BACKEND}).")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", DEFAULT_TOP_K))
    min_rating = request.args.get("min_rating", "")
    language = request.args.get("language", "").strip().lower()

    if not query:
        return jsonify({"results": [], "count": 0, "query": query})

    # Build optional metadata filter
    filters = _build_filter(min_rating, language)

    try:
        raw = vectorstore.similarity_search_with_score(query, k=top_k, filter=filters)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    results = []
    for doc, score in raw:
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

    return jsonify({"results": results, "count": len(results), "query": query})


def _build_filter(min_rating: str, language: str) -> dict | None:
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
