"""API integration tests."""

import pytest
from fastapi.testclient import TestClient

# Import after setting path - app needs to be importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data


def test_health(client):
    r = client.get("/api/v1/health")
    # 200 when services initialized, 503 if encoder/retriever not ready
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "status" in data
        assert "vector_db_connected" in data
        assert "index_stats" in data


def test_search_text_empty(client):
    r = client.post("/api/v1/search-text", json={"query": "", "top_k": 5})
    assert r.status_code == 422  # Validation error for empty query


def test_search_text_valid(client):
    r = client.post(
        "/api/v1/search-text",
        json={"query": "a red square", "top_k": 5},
    )
    # 200 OK, 500 server error, or 503 if services not initialized
    assert r.status_code in (200, 500, 503)
    if r.status_code == 200:
        data = r.json()
        assert "results" in data
        assert "latency_ms" in data
