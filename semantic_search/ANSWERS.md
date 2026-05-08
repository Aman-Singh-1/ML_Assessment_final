# ML Engineer Assessment — Dream Reflection Media
## Written Answers (Q2–Q6)

---

## Q2. Architecture

### Embedding Model — `all-MiniLM-L6-v2`

Sentence Transformers' `all-MiniLM-L6-v2` was chosen for four reasons:

1. **Dimensionality vs quality sweet spot** — 384 dimensions. Dense enough to capture semantic nuance; compact enough to keep FAISS memory and compute low.
2. **Speed** — ~14,000 sentences/second on CPU. No GPU needed at this scale.
3. **Size** — ~80 MB. Loads in under 2 seconds; fits in any container.
4. **Proven benchmark** — top performer on MTEB (Massive Text Embedding Benchmark) for its size class.

*Scale-up path:* `text-embedding-3-small` (OpenAI API, 1536-dim) or `bge-large-en-v1.5` (1024-dim) if richer semantics are needed later.

---

### Database Choice

Two stores are used together — each doing what it does best:

| Store | Role | Why |
|-------|------|-----|
| **SQLite** | Metadata: text, JSON metadata, timestamps, embedding status | Zero-config, ACID-compliant, fast reads, no extra service to run |
| **FAISS** (`IndexFlatIP` + `IndexIDMap`) | Vector similarity search | Cosine search over 384-dim vectors, microsecond latency at <50k vectors |

**Key design insight:** FAISS internal IDs are set to the same integer rowids as SQLite. After a vector search returns IDs, the metadata lookup is a single `WHERE id IN (...)` query — no join complexity, no foreign key overhead.

**Why not pgvector?** pgvector is excellent but adds a Postgres dependency. For a self-contained mini system, SQLite + FAISS has zero ops overhead and the same search semantics. The migration path to pgvector (for multi-service deployments) is straightforward because the API contract doesn't change.

---

### API Flow

```
Client
  │
  ▼ POST /api/v1/ingest { "text": "...", "metadata": {...} }
  │
  ├─► [1] Pydantic validation (length, non-empty, type checks)
  ├─► [2] embed_text(text) → 384-dim float32 vector
  │         └─ checks LRU cache first (MD5 hash key)
  │         └─ ThreadPoolExecutor with 10s timeout guard
  ├─► [3] L2-normalise vector → inner product == cosine similarity
  ├─► [4] SQLite INSERT → get auto-increment rowid
  ├─► [5] FAISS add_with_ids(vector, rowid) + persist to disk
  └─► [6] Return { "doc_id": 42 }

  ▼ POST /api/v1/search { "query": "...", "top_k": 5 }
  │
  ├─► [1] Validate query
  ├─► [2] embed_text(query) → normalise
  ├─► [3] FAISS index.search(query_vec, top_k) → [(score, id), ...]
  ├─► [4] SQLite SELECT WHERE id IN (...) → fetch text + metadata
  └─► [5] Return ranked list with scores + latency_ms
```

---

### Data Pipeline

```
Raw Input
   │
   ▼ Validation (Pydantic) — reject empty/oversized/malformed
   │
   ▼ Normalisation — strip whitespace, length check
   │
   ▼ Embedding (sentence-transformers) — float32 numpy array
   │
   ▼ L2 Normalisation — convert to unit vector
   │
   ├── SQLite write (text + metadata) → rowid
   └── FAISS write (vector + rowid) → persist to disk
```

For bulk ingest, the pipeline batches all N texts through the model in a **single forward pass** (the model's `encode()` handles internal batching), then does a single `add_with_ids()` to FAISS — orders of magnitude faster than N serial calls.

---

### Scaling Approach

| Scale | Strategy |
|-------|----------|
| < 50k records | Current: SQLite + `IndexFlatIP` (exact search, ~1ms) |
| 50k – 500k | Replace with `IndexIVFFlat` (nlist=256, nprobe=10). ~4× faster at 90%+ recall |
| 500k – 5M | `IndexHNSWFlat` (M=32). Graph-based ANN, no training needed, sub-10ms |
| 5M+ | Move to **pgvector** on Postgres with IVFFlat index + connection pooling (PgBouncer). Or managed Pinecone/Weaviate. Horizontal read replicas for search load. |
| Embedding latency | Switch to async embedding with a dedicated model server (TorchServe / Triton). Cache embeddings for repeat inputs (already implemented via LRU). |

---

## Q3. Retrieval Speed Optimisation & Indexing Strategy

### Current: `IndexFlatIP` — Exact Search
- O(N × D) per query. At N=50k, D=384: ~19M multiply-adds ≈ **< 2ms on modern CPU**.
- No false negatives. Best for datasets where precision matters more than speed.

### 10k Records
`IndexFlatIP` is the right choice. Exact search, no training phase, results in ~0.5ms.

### 100k Records — Switch to `IndexIVFFlat`
```python
nlist = 256  # Voronoi cells
quantiser = faiss.IndexFlatIP(384)
index = faiss.IndexIVFFlat(quantiser, 384, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(training_vectors)  # needs ≥ 39×nlist vectors
index.nprobe = 10              # search 10/256 cells — tune for recall/speed tradeoff
```
- Reduces search space from N to N/nlist per probe ≈ **10× speedup** at ~97% recall.

### 1M+ Records — `IndexHNSWFlat`
```python
index = faiss.IndexHNSWFlat(384, 32)  # M=32 neighbours per node
```
- Graph-based ANN. Builds during insert; no offline training needed.
- Sub-10ms at 1M vectors, ~95% recall at M=32.

### Additional Speed Optimisations Implemented

1. **Embedding LRU cache** — identical queries skip the model entirely. Hit rate is often 20–40% in production (repeat searches, autocomplete prefetches).
2. **L2 normalisation** — done once at insert time, not at query time.
3. **Batch encoding** — `embed_batch()` runs one model forward pass for N texts (5–10× faster than N individual calls).
4. **Index persisted to disk** — loaded once into RAM on the first query, then reused in-process.
5. **Thread pool** — embedding runs in a `ThreadPoolExecutor` so the main FastAPI thread stays non-blocking.

---

## Q4. Error Handling Strategy

### Invalid Input
- Pydantic validators reject at the schema layer before any code runs.
- Empty strings, strings over 5000 chars, and wrong types return **HTTP 422 Unprocessable Entity** with a structured error body.
- Input is stripped of leading/trailing whitespace before validation — "  " returns the same error as "".

### Missing Embeddings
- If the embedding step succeeds but the FAISS write fails (e.g., disk full), the `embedding_status` column in SQLite is set to `'failed'`. The document exists but is unsearchable.
- A background job (e.g., APScheduler or Celery task) can periodically query `WHERE embedding_status = 'failed'` and retry.
- If the entire FAISS index is missing at search time, `load_or_create_index()` creates a fresh empty one — search returns an empty list rather than crashing.

### Slow Model Responses
- All embedding calls run inside a `ThreadPoolExecutor` with a configurable timeout (`MODEL_TIMEOUT_SECONDS`, default 10s).
- On timeout, the API returns **HTTP 503 Service Unavailable** with a clear message.
- The LRU cache absorbs repeat queries — a slow model doesn't degrade cached results.
- Scale-up: move the model to a dedicated async server (Triton Inference Server, Ray Serve) with its own health checks.

### Database Failures
- All DB calls are wrapped in try/except. On failure, **HTTP 503** is returned with a safe error message (no stack traces exposed to callers).
- SQLite uses WAL (Write-Ahead Logging) mode for better concurrent-read performance and crash recovery.
- FAISS index is persisted after every write. On crash+restart, the index is reloaded from disk — no data loss.
- For production, add a circuit breaker pattern (e.g., `pybreaker`) so repeated DB failures trigger fast-fail rather than pile-up.

---

## Q5. Trade-offs

| Decision | What I chose | What I gave up | Why |
|----------|-------------|----------------|-----|
| **FAISS vs pgvector** | FAISS + SQLite | Full SQL joins, ACID across stores | Zero ops overhead; same query semantics; easy to swap later |
| **Flat vs ANN index** | `IndexFlatIP` (exact) | Speed at >50k | 100% precision; the code has IVF/HNSW commented in ready to swap |
| **In-process model vs API** | In-process | Horizontal scaling of embedding | Simpler deployment; avoids network hop; adequate for <100 RPS |
| **LRU dict vs Redis cache** | In-memory dict | Cross-process cache sharing | No infra dependency; Redis is the obvious next step |
| **SQLite vs Postgres** | SQLite | Multi-writer concurrency | Right-sizes the system; Postgres migration is a config change |
| **Sync FastAPI vs async** | Sync (ThreadPoolExecutor) | True async non-blocking | sentence-transformers is a synchronous library; the executor preserves FastAPI's responsiveness |
| **Disk-persisted FAISS vs in-memory only** | Disk-persisted | Slightly slower first load | Survives restarts without data loss |

---

## Q6. Evaluation Strategy

### Offline Evaluation

**Ground truth dataset construction:**
Create 100–500 labelled (query, relevant_doc) pairs using one of:
- Human annotators (gold standard)
- GPT-4 to generate plausible query/document pairs
- Historical search logs with click-through data

**Metrics:**

| Metric | What it measures |
|--------|-----------------|
| **Precision@K** | Of the top-K results, what fraction are relevant? |
| **Recall@K** | Of all relevant docs, how many appear in top-K? |
| **MRR (Mean Reciprocal Rank)** | How high is the first relevant result? |
| **NDCG@K** | Graded relevance — rewards ranking very relevant docs higher |

```python
# Example: Precision@5
def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    return len(set(retrieved_ids[:k]) & relevant_ids) / k
```

### Online / Production Evaluation

1. **A/B testing** — Compare new model/index configuration against baseline on real traffic. Measure click-through rate, session depth, conversion.
2. **Query latency p50/p95/p99** — Track via logs or Prometheus. Alert if p95 exceeds 200ms.
3. **Zero-result rate** — High rate means index is too sparse or query distribution has shifted.
4. **Score distribution monitoring** — If average similarity scores drop over time, the index may be stale (new document types diverging from old ones).

### Regression Tests (already in `tests/test_api.py`)
- Smoke tests for each endpoint
- Validation boundary tests (empty input, oversized input, bad top_k)
- Mock-patched model tests to verify the retrieval pipeline independent of the model

---

*Full source code in the accompanying zip. Run `python scripts/demo_offline.py` to see the pipeline in action, then `pytest tests/ -v` for the test suite (16/16 pass).*
