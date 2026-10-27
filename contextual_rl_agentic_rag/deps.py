"""Dependency wiring.

A single :class:`RagContext` bundles the shared collaborators (LLM backend,
vector store, BM25 store, knowledge graph). It is built once and injected into
every agent so they share the same underlying stores and clients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from contextual_rl_agentic_rag.config import Settings, settings as default_settings
from contextual_rl_agentic_rag.llm.base import LLMBackend
from contextual_rl_agentic_rag.llm.factory import get_llm
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.retrieval.base import BaseVectorStore
from contextual_rl_agentic_rag.retrieval.bm25_store import BM25Store
from contextual_rl_agentic_rag.retrieval.graph_store import KnowledgeGraph
from contextual_rl_agentic_rag.retrieval.vectorstore import build_vector_store

logger = get_logger(__name__)


@dataclass
class RagContext:
    llm: LLMBackend
    vector_store: BaseVectorStore
    bm25: Optional[BM25Store] = None
    kg: Optional[KnowledgeGraph] = None
    settings: Settings = field(default_factory=lambda: default_settings)


def build_context(
    settings: Settings | None = None,
    *,
    llm: LLMBackend | None = None,
    vector_store: BaseVectorStore | None = None,
    bm25: BM25Store | None = None,
    kg: KnowledgeGraph | None = None,
) -> RagContext:
    settings = settings or default_settings
    llm = llm or get_llm(settings)
    vector_store = vector_store or build_vector_store(settings)

    if bm25 is None:
        bm25 = BM25Store("hybrid", base_dir=settings.base_dir)

    if kg is None:
        try:
            kg = KnowledgeGraph(llm=llm, path=settings.base_dir / "knowledge_graph.pkl")
        except Exception as exc:  # noqa: BLE001 - networkx may be unavailable
            logger.warning("KnowledgeGraph unavailable (%s); graph_rag will degrade.", exc)
            kg = None

    return RagContext(
        llm=llm, vector_store=vector_store, bm25=bm25, kg=kg, settings=settings
    )
