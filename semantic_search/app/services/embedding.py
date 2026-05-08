"""
Embedding Service
-----------------
Wraps sentence-transformers with:
  - Lazy model loading (first call only)
  - In-memory LRU cache (avoid re-encoding identical inputs)
  - Timeout guard
  - Graceful error handling
"""

import numpy as np
import hashlib
import logging
import threading
import time
from functools import lru_cache
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def get_model():
    """Thread-safe lazy load."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # double-checked locking
                logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(settings.EMBEDDING_MODEL)
                _model.encode("warmup", show_progress_bar=False)
                logger.info("Model loaded — dim=%d", settings.EMBEDDING_DIM)
    return _model


# ── LRU cache keyed on text hash (avoids encoding same string twice) ──────────
_cache: dict[str, np.ndarray] = {}
_CACHE_MAX = 1000


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _get_cached(text: str) -> Optional[np.ndarray]:
    return _cache.get(_cache_key(text))


def _set_cached(text: str, vec: np.ndarray):
    if len(_cache) >= _CACHE_MAX:
        # Evict oldest entry (FIFO approximation)
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[_cache_key(text)] = vec


# ─────────────────────────── Public API ──────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty after stripping whitespace.")
    if len(text) > settings.MAX_INPUT_LENGTH:
        raise ValueError(f"Input exceeds max length ({settings.MAX_INPUT_LENGTH} chars).")

    cached = _get_cached(text)
    if cached is not None:
        return cached

    try:
        vec = get_model().encode(text, normalize_embeddings=False, show_progress_bar=False)
    except Exception as e:
        raise RuntimeError(f"Embedding model failed: {e}")

    vec = np.array(vec, dtype="float32")
    _set_cached(text, vec)
    return vec


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings efficiently using batching.
    Returns shape (N, DIM).
    """
    if not texts:
        raise ValueError("Empty text list.")

    texts = [t.strip() for t in texts]

    # Separate cached vs uncached
    result = np.zeros((len(texts), settings.EMBEDDING_DIM), dtype="float32")
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        cached = _get_cached(text)
        if cached is not None:
            result[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        def _batch_encode():
            return get_model().encode(
                uncached_texts,
                batch_size=settings.BATCH_SIZE,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        future = _executor.submit(_batch_encode)
        try:
            vecs = future.result(timeout=settings.MODEL_TIMEOUT_SECONDS * 3)
        except FuturesTimeout:
            raise RuntimeError("Batch embedding timed out.")

        for idx, (orig_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            vec = np.array(vecs[idx], dtype="float32")
            result[orig_idx] = vec
            _set_cached(text, vec)

    logger.info("Encoded %d texts (%d from cache)", len(texts), len(texts) - len(uncached_texts))
    return result
