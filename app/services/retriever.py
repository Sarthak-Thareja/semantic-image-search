"""Vector retrieval service using ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from app.config import settings
from app.services.encoder import EncoderService


class RetrieverService:
    """Retrieve similar images via vector search."""

    def __init__(self, encoder: EncoderService | None = None):
        self.encoder = encoder or EncoderService()
        self._client = None
        self._collection = None

    def _ensure_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._ensure_client()
            self._collection = client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add embeddings to the index."""
        coll = self._get_collection()
        emb_list = embeddings.tolist()
        coll.add(ids=ids, embeddings=emb_list, metadatas=metadatas or [{}] * len(ids))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar embeddings. Returns list of {id, score, metadata}."""
        k = top_k or settings.default_top_k
        k = min(k, settings.max_top_k)
        coll = self._get_collection()
        results = coll.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        # ChromaDB returns distances (lower = more similar for cosine)
        # For cosine: distance = 1 - similarity, so score = 1 - distance
        out = []
        ids = results["ids"][0] or []
        metadatas = results["metadatas"][0] or []
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
        for i, (doc_id, meta, dist) in enumerate(zip(ids, metadatas, distances)):
            score = float(1.0 - dist) if dist is not None else 0.0
            out.append({"id": doc_id, "score": round(score, 4), "metadata": meta or {}})
        return out

    def count(self) -> int:
        """Return number of vectors in the collection."""
        coll = self._get_collection()
        return coll.count()

    def is_connected(self) -> bool:
        """Check if vector DB is accessible."""
        try:
            self._ensure_client()
            self._get_collection()
            return True
        except Exception:
            return False

    def get_index_stats(self) -> dict[str, Any]:
        """Return index statistics for health endpoint."""
        try:
            coll = self._get_collection()
            return {
                "total_vectors": coll.count(),
                "dimension": self.encoder.DIMENSION,
            }
        except Exception:
            return {"total_vectors": 0, "dimension": self.encoder.DIMENSION}

    def get_metadata_by_id(self, id: str) -> dict[str, Any] | None:
        """Get metadata for an indexed image by id."""
        try:
            coll = self._get_collection()
            res = coll.get(ids=[id], include=["metadatas"])
            # ChromaDB returns metadatas as flat list: [{"path": ..., "caption": ...}]
            if res["metadatas"] and len(res["metadatas"]) > 0:
                return res["metadatas"][0]
        except Exception:
            pass
        return None
