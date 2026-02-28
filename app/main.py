"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, encoder, retriever
from app.api.schemas import ErrorResponse
from app.config import settings
from app.services.encoder import EncoderService
from app.services.retriever import RetrieverService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global encoder, retriever
    enc = EncoderService()
    ret = RetrieverService(encoder=enc)
    # Inject into routes module
    import app.api.routes as routes_mod
    routes_mod.encoder = enc
    routes_mod.retriever = ret
    yield
    # Cleanup if any
    routes_mod.encoder = None
    routes_mod.retriever = None


app = FastAPI(
    title="Multimodal Semantic Image-Text Retrieval API",
    description="Search images by text or image using CLIP embeddings",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root redirect to docs."""
    return {"message": "Multimodal Image-Text Retrieval API", "docs": "/docs"}
