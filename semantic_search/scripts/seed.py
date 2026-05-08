"""
scripts/seed.py
---------------
Populates the vector index with sample data for demo / manual testing.
Run: python scripts/seed.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.db.database import init_db, bulk_insert
from app.services.embedding import embed_batch

SAMPLE_TEXTS = [
    "Machine learning is transforming the way we personalise user experiences.",
    "Semantic search enables finding relevant content based on meaning, not keywords.",
    "Vector databases store high-dimensional embeddings for fast similarity retrieval.",
    "FAISS is a library developed by Meta for efficient similarity search.",
    "Sentence transformers produce dense representations of text for downstream tasks.",
    "Recommendation systems analyse user behaviour to surface personalised content.",
    "Lead intelligence identifies high-intent prospects based on their digital signals.",
    "Natural language processing allows machines to understand human language.",
    "Cosine similarity measures the angle between two vectors in high-dimensional space.",
    "Embeddings compress semantic meaning into compact numerical representations.",
    "Real-time retrieval requires low-latency index structures like HNSW or IVF.",
    "Data pipelines orchestrate the flow from raw input to stored vector representations.",
    "FastAPI is a modern Python web framework ideal for building ML-powered APIs.",
    "SQLite provides a lightweight relational store for document metadata.",
    "Production ML systems require robust error handling and fallback strategies.",
]

if __name__ == "__main__":
    print("Initialising DB...")
    init_db()

    print(f"Encoding {len(SAMPLE_TEXTS)} texts...")
    vecs = embed_batch(SAMPLE_TEXTS)

    print("Inserting into FAISS + SQLite...")
    ids = bulk_insert(SAMPLE_TEXTS, vecs)

    print(f"Done! Inserted {len(ids)} documents. IDs: {ids}")
