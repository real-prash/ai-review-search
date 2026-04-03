"""
One-time migration: local Chroma DB → Qdrant Cloud.

Run this ONCE after the ingestion pipeline has finished:
    python migrate_to_qdrant.py

Requirements:
    pip install qdrant-client langchain-qdrant langchain-chroma langchain-huggingface sentence-transformers

Environment variables needed (set in .env or shell):
    QDRANT_URL      e.g. https://xxxx.us-east4-0.gcp.cloud.qdrant.io
    QDRANT_API_KEY  your Qdrant Cloud API key

Where to get these:
    1. Sign up free at https://cloud.qdrant.io
    2. Create a cluster (free tier: 1 GB, no expiry)
    3. Copy the cluster URL and API key from the dashboard
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_reviews_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "langchain")
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_reviews")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension
BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Load embeddings + source Chroma
# ---------------------------------------------------------------------------
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)

print(f"Opening local Chroma at {CHROMA_DIR!r}...")
chroma = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=CHROMA_COLLECTION,
    embedding_function=embeddings,
)

total_docs = chroma._collection.count()
print(f"Found {total_docs:,} documents in local Chroma.")

# ---------------------------------------------------------------------------
# Create Qdrant collection (idempotent — skips if already exists)
# ---------------------------------------------------------------------------
print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

existing = [c.name for c in client.get_collections().collections]
if QDRANT_COLLECTION not in existing:
    print(f"Creating collection '{QDRANT_COLLECTION}'...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
else:
    print(f"Collection '{QDRANT_COLLECTION}' already exists — appending.")

qdrant_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

# ---------------------------------------------------------------------------
# Stream documents from Chroma → Qdrant in batches
# ---------------------------------------------------------------------------
print(f"Migrating {total_docs:,} documents in batches of {BATCH_SIZE}...")

from langchain_core.documents import Document

migrated = 0
offset = 0  # Chroma uses integer offsets

while offset < total_docs:
    result = chroma._collection.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["documents", "metadatas"],
    )
    ids = result["ids"]
    if not ids:
        break

    docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]

    qdrant_store.add_documents(docs)
    migrated += len(docs)
    offset += len(docs)
    print(f"  Migrated {migrated:,} / {total_docs:,}...")

print(f"\nDone. {migrated:,} documents in Qdrant Cloud collection '{QDRANT_COLLECTION}'.")
print("You can now deploy search_app.py with QDRANT_URL and QDRANT_API_KEY set.")
