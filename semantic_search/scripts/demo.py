"""
scripts/demo.py
---------------
Demonstrates the full system without an HTTP server:
  1. Store several texts with embeddings
  2. Query with a natural language question
  3. Print ranked results

Run: python scripts/demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json

# Override paths so demo doesn't pollute production index
os.environ["SQLITE_PATH"] = "/tmp/demo_semantic.db"
os.environ["FAISS_INDEX_PATH"] = "/tmp/demo_semantic.index"

from app.db.database import init_db, insert_document, search_similar
from app.services.embedding import embed_text

DOCS = [
    "I want to build a recommendation engine for e-commerce.",
    "How do I set up a CI/CD pipeline for a Python service?",
    "What is the best way to fine-tune a language model?",
    "Our team needs a solution for customer segmentation.",
    "I need help designing a scalable microservices architecture.",
    "Can you explain gradient descent and backpropagation?",
    "We are looking for a CRM integration for our sales team.",
    "Tell me about transformer architectures and self-attention.",
]

QUERIES = [
    "suggestions for similar products",
    "training neural networks",
    "managing customer data",
]

def main():
    print("=" * 60)
    print("   SEMANTIC SEARCH DEMO")
    print("=" * 60)

    init_db()

    print("\n[1] Storing documents...")
    for text in DOCS:
        vec = embed_text(text)
        doc_id = insert_document(text, vec)
        print(f"   Stored doc_id={doc_id}: {text[:55]}...")

    for query in QUERIES:
        print(f"\n[2] Query: \"{query}\"")
        q_vec = embed_text(query)
        results = search_similar(q_vec, top_k=3)
        for rank, r in enumerate(results, 1):
            print(f"   #{rank} (score={r['score']:.4f}): {r['text']}")

    print("\n" + "=" * 60)
    print("Demo complete.")

if __name__ == "__main__":
    main()
