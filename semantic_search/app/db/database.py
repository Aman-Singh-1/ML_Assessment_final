"""
Database layer
- SQLite  : stores id, text, metadata, timestamp
- FAISS   : stores L2-normalised vectors for cosine similarity search

Design decision: keep SQL and vector stores in sync via the same integer row-id.
FAISS internal ids == SQLite rowids, so joins are O(1).
"""

import sqlite3
import faiss
import numpy as np
import threading
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)

# Thread-safe lock for FAISS writes (reads are concurrent-safe)
_faiss_write_lock = threading.Lock()


# ─────────────────────────── SQLite helpers ──────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Return a connection with row_factory for dict-like rows."""
    conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT    NOT NULL,
                metadata  TEXT,                    -- JSON blob for custom fields
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding_status TEXT DEFAULT 'ok' -- ok | missing | failed
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON documents(created_at)")
        conn.commit()
    logger.info("SQLite initialised at %s", settings.SQLITE_PATH)


# ─────────────────────────── FAISS helpers ───────────────────────────────────

def load_or_create_index() -> faiss.Index:
    """
    Load existing FAISS index from disk, or create a fresh one.
    Index type: IndexFlatIP (inner product) on L2-normalised vectors → cosine similarity.
    For large scale (100k+) swap to IndexIVFFlat or HNSW (see comments).
    """
    path = settings.FAISS_INDEX_PATH
    if Path(path).exists():
        index = faiss.read_index(path)
        logger.info("Loaded FAISS index from %s (%d vectors)", path, index.ntotal)
    else:
        # Flat exact search — fine up to ~50k vectors
        index = faiss.IndexIDMap(faiss.IndexFlatIP(settings.EMBEDDING_DIM))

        # ── Scale-up option A: IVF (inverted file, ANN) ──────────────────
        # nlist = 100  # number of Voronoi cells
        # quantiser = faiss.IndexFlatIP(settings.EMBEDDING_DIM)
        # index = faiss.IndexIVFFlat(quantiser, settings.EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        # index.train(training_vectors)   # needs ≥39×nlist vectors first
        # index.nprobe = 10              # trade recall vs speed

        # ── Scale-up option B: HNSW (graph-based, no training needed) ────
        # index = faiss.IndexHNSWFlat(settings.EMBEDDING_DIM, 32)  # M=32
        # index = faiss.IndexIDMap(index)

        logger.info("Created new FAISS IndexFlatIP (dim=%d)", settings.EMBEDDING_DIM)
    return index


def save_index(index: faiss.Index):
    faiss.write_index(index, settings.FAISS_INDEX_PATH)


# ─────────────────────────── Core operations ─────────────────────────────────

def insert_document(text: str, embedding: np.ndarray, metadata: Optional[str] = None) -> int:
    """
    Insert one document atomically:
    1. Write metadata to SQLite → get rowid
    2. Add vector to FAISS under that same rowid
    3. Persist index to disk
    """
    # Step 1: SQLite insert
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO documents (text, metadata) VALUES (?, ?)",
            (text, metadata),
        )
        doc_id = cursor.lastrowid
        conn.commit()

    # Step 2: FAISS insert (thread-safe)
    vec = _normalise(embedding).reshape(1, -1).astype("float32")
    with _faiss_write_lock:
        index = load_or_create_index()
        index.add_with_ids(vec, np.array([doc_id], dtype="int64"))
        save_index(index)

    logger.debug("Inserted doc_id=%d", doc_id)
    return doc_id


def bulk_insert(texts: list[str], embeddings: np.ndarray, metadatas: list[Optional[str]] = None) -> list[int]:
    """
    Batch insert — significantly faster than N individual inserts.
    Writes all SQLite rows first, then a single FAISS add_with_ids call.
    """
    metadatas = metadatas or [None] * len(texts)
    doc_ids = []

    with get_db() as conn:
        for text, meta in zip(texts, metadatas):
            cur = conn.execute(
                "INSERT INTO documents (text, metadata) VALUES (?, ?)", (text, meta)
            )
            doc_ids.append(cur.lastrowid)
        conn.commit()

    vecs = _normalise(embeddings).astype("float32")
    ids = np.array(doc_ids, dtype="int64")
    with _faiss_write_lock:
        index = load_or_create_index()
        index.add_with_ids(vecs, ids)
        save_index(index)

    logger.info("Bulk inserted %d documents", len(doc_ids))
    return doc_ids


def search_similar(query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k most similar documents.
    Returns list of {id, text, metadata, score, created_at}.
    """
    index = load_or_create_index()
    if index.ntotal == 0:
        return []

    top_k = min(top_k, index.ntotal)
    q = _normalise(query_vec).reshape(1, -1).astype("float32")
    scores, ids = index.search(q, top_k)

    # Fetch metadata from SQLite
    valid = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    if not valid:
        return []

    id_list = [v[0] for v in valid]
    placeholders = ",".join("?" * len(id_list))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, text, metadata, created_at FROM documents WHERE id IN ({placeholders})",
            id_list,
        ).fetchall()

    row_map = {row["id"]: dict(row) for row in rows}
    results = []
    for doc_id, score in valid:
        if doc_id in row_map:
            entry = row_map[doc_id].copy()
            entry["score"] = round(score, 4)
            results.append(entry)

    return results


def get_document(doc_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, text, metadata, created_at, embedding_status FROM documents WHERE id=?",
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def mark_embedding_failed(doc_id: int):
    with get_db() as conn:
        conn.execute(
            "UPDATE documents SET embedding_status='failed' WHERE id=?", (doc_id,)
        )
        conn.commit()


# ─────────────────────────── Utils ───────────────────────────────────────────

def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise so inner product == cosine similarity."""
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)   # guard zero-vector
    return vecs / norms
