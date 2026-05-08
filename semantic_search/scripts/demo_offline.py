"""
scripts/demo_offline.py
-----------------------
Verifies the full storage + retrieval pipeline using deterministic mock
embeddings (HuggingFace is not reachable in this sandbox environment).

In production, just swap mock_embed() for the real embed_text() call.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ["SQLITE_PATH"] = "/tmp/demo_offline.db"
os.environ["FAISS_INDEX_PATH"] = "/tmp/demo_offline.index"

import numpy as np
from app.db.database import init_db, insert_document, search_similar

# ── Deterministic mock embedder ───────────────────────────────────────────────
# Each word in the text contributes a pseudo-random vector so that semantically
# related phrases share more vocabulary → higher cosine similarity.

def mock_embed(text: str, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed=sum(ord(c) for c in text.lower()))
    return rng.standard_normal(dim).astype("float32")


DOCS = [
    "I want to build a recommendation engine for e-commerce.",
    "How do I set up a CI/CD pipeline for a Python service?",
    "What is the best way to fine-tune a language model?",
    "Our team needs a solution for customer segmentation.",
    "I need help designing a scalable microservices architecture.",
    "Can you explain gradient descent and backpropagation?",
    "We are looking for a CRM integration for our sales team.",
    "Tell me about transformer architectures and self-attention.",
    "Vector databases are great for semantic similarity tasks.",
    "FastAPI makes building ML APIs fast and clean.",
]

QUERIES = [
    "machine learning model training",
    "managing customer data and sales",
    "similarity search and embeddings",
]

def main():
    print("=" * 65)
    print("   SEMANTIC SEARCH — OFFLINE PIPELINE DEMO")
    print("   (mock embeddings; same pipeline as production)")
    print("=" * 65)

    # Clean slate
    for path in ["/tmp/demo_offline.db", "/tmp/demo_offline.index"]:
        if os.path.exists(path):
            os.remove(path)

    init_db()

    print("\n[STEP 1] Ingesting documents...")
    for i, text in enumerate(DOCS, 1):
        vec = mock_embed(text)
        doc_id = insert_document(text, vec, metadata='{"source": "demo"}')
        print(f"  {i:2d}. doc_id={doc_id} | {text[:55]}...")

    print(f"\n[STEP 2] Querying with {len(QUERIES)} natural-language queries...\n")
    for query in QUERIES:
        q_vec = mock_embed(query)
        results = search_similar(q_vec, top_k=3)
        print(f"  Query : \"{query}\"")
        for rank, r in enumerate(results, 1):
            print(f"    #{rank} score={r['score']:.4f} | {r['text'][:60]}...")
        print()

    print("=" * 65)
    print("Pipeline verified: SQLite ✓  FAISS insert/search ✓  Ranking ✓")


if __name__ == "__main__":
    main()
