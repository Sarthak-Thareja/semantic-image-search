"""Pytest configuration and fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure ChromaDB uses project data dir for tests."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv("RETRIEVAL_CHROMA_PERSIST_DIR", os.path.join(root, "data", "chroma"))
