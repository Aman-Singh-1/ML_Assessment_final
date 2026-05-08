"""
Central configuration — single source of truth.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # 384-dim, 80MB, fast + accurate
    EMBEDDING_DIM: int = 384
    MODEL_TIMEOUT_SECONDS: float = 60.0   # was 10.0
    # Database
    SQLITE_PATH: str = "semantic_search.db"       # metadata store
    FAISS_INDEX_PATH: str = "faiss.index"         # vector index

    # Retrieval
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 50

    # Performance
    BATCH_SIZE: int = 64                          # for bulk inserts
    MODEL_TIMEOUT_SECONDS: float = 10.0

    # Input validation
    MIN_INPUT_LENGTH: int = 1
    MAX_INPUT_LENGTH: int = 5000

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
