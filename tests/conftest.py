"""Shared fixtures. Everything runs offline with the fake backend + in-memory store."""
from __future__ import annotations

import pytest

from contextual_rl_agentic_rag.config import settings as app_settings
from contextual_rl_agentic_rag.deps import RagContext
from contextual_rl_agentic_rag.llm.fake_backend import FakeEmbeddingBackend, FakeLLMBackend
from contextual_rl_agentic_rag.retrieval.bm25_store import BM25Store
from contextual_rl_agentic_rag.retrieval.memory_store import InMemoryVectorStore
from contextual_rl_agentic_rag.schema import Document


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Point all persistence at a temp dir so tests never touch ./data."""
    monkeypatch.setattr(app_settings, "base_dir", tmp_path)
    monkeypatch.setattr(app_settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(app_settings, "chroma_dir", tmp_path / "chroma")
    return app_settings


@pytest.fixture
def fake_ctx(tmp_settings) -> RagContext:
    llm = FakeLLMBackend()
    store = InMemoryVectorStore(FakeEmbeddingBackend())
    bm25 = BM25Store("test", base_dir=tmp_settings.base_dir)
    kg = None
    try:
        from contextual_rl_agentic_rag.retrieval.graph_store import KnowledgeGraph

        kg = KnowledgeGraph(llm=llm, path=tmp_settings.base_dir / "kg.pkl")
    except Exception:  # noqa: BLE001 - networkx optional in minimal envs
        kg = None
    return RagContext(llm=llm, vector_store=store, bm25=bm25, kg=kg, settings=tmp_settings)


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document("RAG retrieves relevant documents and then generates an answer with an LLM.", {"src": "a"}),
        Document("A multi-armed bandit balances exploration and exploitation of actions.", {"src": "b"}),
        Document("UCB1 adds an exploration bonus to the average reward of each arm.", {"src": "c"}),
    ]
