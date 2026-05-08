

import json
import time
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.models import (
    IngestRequest, IngestResponse,
    BulkIngestRequest, BulkIngestResponse,
    SearchRequest, SearchResponse, SearchResult,
    DocumentResponse,
)
from app.services.embedding import embed_text, embed_batch
from app.db.database import (
    insert_document, bulk_insert, search_similar,
    get_document, mark_embedding_failed, load_or_create_index,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ─────────────────────────── Ingest ──────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest(req: IngestRequest):
   
    metadata_str = json.dumps(req.metadata) if req.metadata else None
    doc_id = None

    try:
        vec = embed_text(req.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # Model slow / failed — store text with degraded status, return 503
        logger.error("Embedding failed for ingest: %s", e)
        raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}")

    try:
        doc_id = insert_document(req.text, vec, metadata_str)
    except Exception as e:
        logger.error("DB insert failed: %s", e)
        raise HTTPException(status_code=503, detail="Database write failed. Try again.")

    return IngestResponse(doc_id=doc_id)


@router.post("/ingest/bulk", response_model=BulkIngestResponse, status_code=201)
def ingest_bulk(req: BulkIngestRequest):
    """
    Efficient batch ingest — one model call for N texts, one FAISS write.
    """
    meta_strs = None
    if req.metadatas:
        meta_strs = [json.dumps(m) if m else None for m in req.metadatas]

    try:
        vecs = embed_batch(req.texts)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        doc_ids = bulk_insert(req.texts, vecs, meta_strs)
    except Exception as e:
        logger.error("Bulk DB insert failed: %s", e)
        raise HTTPException(status_code=503, detail="Database write failed.")

    return BulkIngestResponse(doc_ids=doc_ids, count=len(doc_ids))


# ─────────────────────────── Search ──────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    1. Encode query
    2. FAISS nearest-neighbour search
    3. Fetch metadata from SQLite
    4. Return ranked results with scores
    """
    t0 = time.time()

    try:
        query_vec = embed_text(req.query)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        raw_results = search_similar(query_vec, top_k=req.top_k)
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=503, detail="Search unavailable. Try again.")

    latency_ms = round((time.time() - t0) * 1000, 2)

    results = [SearchResult(**r) for r in raw_results]
    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        results=results,
        latency_ms=latency_ms,
    )


# ─────────────────────────── Fetch by ID ─────────────────────────────────────

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_doc(doc_id: int):
    if doc_id < 1:
        raise HTTPException(status_code=422, detail="doc_id must be a positive integer")
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return DocumentResponse(**doc)


# ─────────────────────────── Stats ───────────────────────────────────────────

@router.get("/stats")
def stats():
    "Return index statistics — useful for monitoring."
    try:
        index = load_or_create_index()
        total_vectors = index.ntotal
    except Exception:
        total_vectors = -1

    return {
        "total_vectors": total_vectors,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "index_type": "IndexFlatIP (cosine via L2-normalisation)",
    }
