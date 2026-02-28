"""API route handlers."""

import base64
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io

from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IndexStats,
    SearchImageResponse,
    SearchResultItem,
    SearchTextRequest,
    SearchTextResponse,
)
from app.config import settings
from app.services.encoder import EncoderService
from app.services.retriever import RetrieverService

router = APIRouter()

# Shared service instances (injected at startup)
encoder: EncoderService | None = None
retriever: RetrieverService | None = None

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = settings.max_image_size_mb * 1024 * 1024


def get_encoder() -> EncoderService:
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialized")
    return encoder


def get_retriever() -> RetrieverService:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever


def _add_image_data_to_results(results: list[dict]) -> list[dict]:
    """Add base64 image data to each result for inline display."""
    out = []
    for r in results:
        r = dict(r)
        meta = r.get("metadata") or {}
        path_str = meta.get("path") or meta.get("url") or meta.get("file_path")
        if path_str and Path(path_str).exists():
            try:
                data = Path(path_str).read_bytes()
                r["image_base64"] = base64.b64encode(data).decode()
            except Exception:
                r["image_base64"] = None
        else:
            r["image_base64"] = None
        out.append(r)
    return out


@router.post("/search-text", response_model=SearchTextResponse)
async def search_text(req: SearchTextRequest):
    """Search images by natural language query."""
    enc = get_encoder()
    ret = get_retriever()
    start = time.perf_counter()
    
    try:
        query_embedding = enc.encode_text(req.query.strip())
        results = ret.search(query_embedding, top_k=req.top_k)
        results = _add_image_data_to_results(results)
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchTextResponse(
            query=req.query,
            results=[SearchResultItem(**r) for r in results],
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search-image", response_model=SearchImageResponse)
async def search_image(
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form()] = 10,
):
    """Search images by uploaded image."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: JPEG, PNG, WebP",
        )
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_image_size_mb}MB",
        )
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {e}") from e

    top_k = max(1, min(top_k, settings.max_top_k))
    enc = get_encoder()
    ret = get_retriever()
    start = time.perf_counter()
    try:
        query_embedding = enc.encode_image(img)
        results = ret.search(query_embedding, top_k=top_k)
        results = _add_image_data_to_results(results)
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchImageResponse(
            results=[SearchResultItem(**r) for r in results],
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@router.get("/images/{image_id}")
async def serve_image(image_id: str):
    """Serve an indexed image by ID (for frontend display)."""
    ret = get_retriever()
    meta = ret.get_metadata_by_id(image_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Image not found")
    path_str = meta.get("path") or meta.get("url") or meta.get("file_path")
    if not path_str:
        raise HTTPException(status_code=404, detail="Image path not in metadata")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    return FileResponse(str(path), media_type=media_type)


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check with vector DB status and index stats."""
    ret = get_retriever()
    connected = ret.is_connected()
    stats = ret.get_index_stats()
    return HealthResponse(
        status="healthy" if connected else "degraded",
        vector_db_connected=connected,
        index_stats=IndexStats(**stats),
    )
