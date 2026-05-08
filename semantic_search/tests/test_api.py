"""
Tests for the Semantic Search API.
Run: pytest tests/ -v
"""

import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect DB and FAISS paths to temp dir for isolation."""
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "test.index"))


@pytest.fixture
def client():
    # Import after env monkeypatching
    from app.core.config import settings
    settings.SQLITE_PATH = "/tmp/test_semantic.db"
    settings.FAISS_INDEX_PATH = "/tmp/test_semantic.index"

    from app.db.database import init_db
    init_db()

    from app.main import app
    return TestClient(app)


MOCK_VEC = np.random.rand(384).astype("float32")


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_ingest_valid(client):
    with patch("app.api.routes.embed_text", return_value=MOCK_VEC):
        r = client.post("/api/v1/ingest", json={"text": "Hello world"})
    assert r.status_code == 201
    assert "doc_id" in r.json()
    assert isinstance(r.json()["doc_id"], int)


def test_ingest_empty_text(client):
    r = client.post("/api/v1/ingest", json={"text": ""})
    assert r.status_code == 422


def test_ingest_too_long(client):
    r = client.post("/api/v1/ingest", json={"text": "x" * 6000})
    assert r.status_code == 422


def test_ingest_with_metadata(client):
    with patch("app.api.routes.embed_text", return_value=MOCK_VEC):
        r = client.post(
            "/api/v1/ingest",
            json={"text": "Product launch email", "metadata": {"user_id": 42, "source": "email"}},
        )
    assert r.status_code == 201


def test_ingest_model_timeout(client):
    with patch("app.api.routes.embed_text", side_effect=RuntimeError("timed out")):
        r = client.post("/api/v1/ingest", json={"text": "test"})
    assert r.status_code == 503


# ── Bulk Ingest ───────────────────────────────────────────────────────────────

def test_bulk_ingest(client):
    texts = ["Text one", "Text two", "Text three"]
    mock_vecs = np.random.rand(3, 384).astype("float32")
    with patch("app.api.routes.embed_batch", return_value=mock_vecs):
        r = client.post("/api/v1/ingest/bulk", json={"texts": texts})
    assert r.status_code == 201
    assert r.json()["count"] == 3


def test_bulk_ingest_empty(client):
    r = client.post("/api/v1/ingest/bulk", json={"texts": []})
    assert r.status_code == 422


def test_bulk_ingest_metadata_mismatch(client):
    r = client.post(
        "/api/v1/ingest/bulk",
        json={"texts": ["a", "b"], "metadatas": [{"k": 1}]},  # 1 meta for 2 texts
    )
    assert r.status_code == 422


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_empty_index(client):
    # Patch FAISS to return an empty index so this test is isolation-safe
    import faiss as _faiss
    from app.core.config import settings as _s
    empty = _faiss.IndexIDMap(_faiss.IndexFlatIP(_s.EMBEDDING_DIM))
    with patch("app.api.routes.embed_text", return_value=MOCK_VEC), \
         patch("app.db.database.load_or_create_index", return_value=empty):
        r = client.post("/api/v1/search", json={"query": "find me something"})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_search_returns_results(client):
    # Ingest first
    with patch("app.api.routes.embed_text", return_value=MOCK_VEC):
        client.post("/api/v1/ingest", json={"text": "Machine learning is great"})
        r = client.post("/api/v1/search", json={"query": "ML systems", "top_k": 1})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "latency_ms" in data


def test_search_invalid_top_k(client):
    r = client.post("/api/v1/search", json={"query": "test", "top_k": 0})
    assert r.status_code == 422


def test_search_empty_query(client):
    r = client.post("/api/v1/search", json={"query": "   "})
    assert r.status_code == 422


# ── Get Document ──────────────────────────────────────────────────────────────

def test_get_document(client):
    with patch("app.api.routes.embed_text", return_value=MOCK_VEC):
        ingest_r = client.post("/api/v1/ingest", json={"text": "Retrieve me"})
    doc_id = ingest_r.json()["doc_id"]
    r = client.get(f"/api/v1/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["text"] == "Retrieve me"


def test_get_document_not_found(client):
    r = client.get("/api/v1/documents/99999")
    assert r.status_code == 404


# ── Health + Stats ────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats(client):
    r = client.get("/api/v1/stats")
    assert r.status_code == 200
    assert "total_vectors" in r.json()
