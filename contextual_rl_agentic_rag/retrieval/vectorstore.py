"""Vector store construction based on configuration."""
from __future__ import annotations

from contextual_rl_agentic_rag.config import Settings, settings as default_settings
from contextual_rl_agentic_rag.llm.factory import get_embeddings
from contextual_rl_agentic_rag.retrieval.base import BaseVectorStore
from contextual_rl_agentic_rag.retrieval.memory_store import InMemoryVectorStore


def build_vector_store(
    settings: Settings | None = None, embeddings=None
) -> BaseVectorStore:
    settings = settings or default_settings
    embeddings = embeddings if embeddings is not None else get_embeddings(settings)
    kind = (settings.vector_store or "chroma").lower()

    if kind == "memory":
        return InMemoryVectorStore(embeddings)
    if kind == "chroma":
        settings.ensure_dirs()
        from contextual_rl_agentic_rag.retrieval.chroma_store import ChromaVectorStore

        return ChromaVectorStore(settings.chroma_dir, embeddings)
    raise ValueError(f"Unknown VECTOR_STORE: {kind!r} (expected chroma|memory)")
