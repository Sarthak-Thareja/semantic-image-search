"""Pydantic schemas for API request/response."""

from typing import Any

from pydantic import BaseModel, Field


# --- Request ---

class SearchTextRequest(BaseModel):
    """Request for text-to-image search."""

    query: str = Field(..., min_length=1, max_length=500, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")


# --- Response ---

class SearchResultItem(BaseModel):
    """Single search result."""

    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_base64: str | None = None  # Base64-encoded image for display


class SearchTextResponse(BaseModel):
    """Response for text-to-image search."""

    query: str
    results: list[SearchResultItem]
    latency_ms: float


class SearchImageResponse(BaseModel):
    """Response for image-to-image search."""

    results: list[SearchResultItem]
    latency_ms: float


class IndexStats(BaseModel):
    """Index statistics."""

    total_vectors: int
    dimension: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_db_connected: bool
    index_stats: IndexStats


# --- Error ---

class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_code: str | None = None
