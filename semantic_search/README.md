# Semantic Search API — 

 Mini ML system: 
 
 
 store inputs   → generate embeddings →   retrieve similar results in real-time.



## Architecture


┌────────────────────────────────────────────────────────┐
│                    FastAPI Application                 │
│                                                        │
│  POST /ingest        POST /search      GET /stats      │
│        │                   │                           │
│        ▼                   ▼                           │
│  ┌──────────────────────────────────────────────┐      │
│  │          Embedding Service                   │      │
│  │  • all-MiniLM-L6-v2 (384-dim)               │      │
│  │  • LRU in-memory cache                       │      │
│  │  • Timeout guard (ThreadPoolExecutor)        │      │
│  └──────────────────────────────────────────────┘      │
│        │                   │                           │
│        ▼                   ▼                           │
│  ┌──────────────┐   ┌──────────────────────────┐       │
│  │   SQLite     │   │    FAISS Index           │       │
│  │  (metadata)  │   │  IndexFlatIP + IDMap     │       │
│  │  text, meta, │   │  (cosine via L2-norm)    │       │
│  │  timestamp   │   │  upgrade: IVF / HNSW     │       │
│  └──────────────┘   └──────────────────────────┘       │
└────────────────────────────────────────────────────────┘



## Quick Start

# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialise DB & seed sample data
python scripts/seed.py

# 3. Run the demo (no server needed)
python scripts/demo.py

# 4. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# 5. Run tests
pytest tests/ -v


## API Endpoints : There are 6 API Endpoints

## Ingest


curl -X POST http://localhost:8001/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "We are looking for an ML engineer", "metadata": {"source": "crm"}}'


## Search


curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning talent acquisition", "top_k": 5}'


## Bulk Ingest Documents

curl -X POST http://localhost:8001/api/v1/ingest/bulk \
  -H "Content-Type: application/json" \
  -d @documents.json

## Get Document by ID
curl http://localhost:8001/api/v1/documents/123

## GET /health

curl http://localhost:8001/health

## Index Statistics

curl http://localhost:8001/api/v1/stats