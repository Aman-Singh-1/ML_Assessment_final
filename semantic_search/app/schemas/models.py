"""
Request / Response schemas with validation.
"""

from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Any
from app.core.config import settings


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[dict[str, Any]] = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if len(v) < settings.MIN_INPUT_LENGTH:
            raise ValueError("text cannot be empty")
        if len(v) > settings.MAX_INPUT_LENGTH:
            raise ValueError(f"text exceeds {settings.MAX_INPUT_LENGTH} characters")
        return v


class BulkIngestRequest(BaseModel):
    texts: list[str]
    metadatas: Optional[list[Optional[dict]]] = None

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Max 1000 texts per bulk request")
        cleaned = []
        for t in v:
            t = t.strip()
            if not t:
                raise ValueError("texts list contains an empty string")
            cleaned.append(t)
        return cleaned

    @model_validator(mode="after")
    def check_lengths_match(self):
        if self.metadatas and len(self.metadatas) != len(self.texts):
            raise ValueError("metadatas length must match texts length")
        return self


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = settings.DEFAULT_TOP_K

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty")
        if len(v) > settings.MAX_INPUT_LENGTH:
            raise ValueError(f"query exceeds {settings.MAX_INPUT_LENGTH} characters")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v is None:
            return settings.DEFAULT_TOP_K
        if v < 1:
            raise ValueError("top_k must be >= 1")
        if v > settings.MAX_TOP_K:
            raise ValueError(f"top_k cannot exceed {settings.MAX_TOP_K}")
        return v


# ── Responses ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: int
    message: str = "Document stored successfully"


class BulkIngestResponse(BaseModel):
    doc_ids: list[int]
    count: int
    message: str = "Bulk ingest successful"


class SearchResult(BaseModel):
    id: int
    text: str
    metadata: Optional[str]
    score: float
    created_at: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]
    latency_ms: float


class DocumentResponse(BaseModel):
    id: int
    text: str
    metadata: Optional[str]
    created_at: str
    embedding_status: str
