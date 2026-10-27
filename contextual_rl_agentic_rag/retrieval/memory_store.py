"""In-process vector store.

Uses the injected embedding backend and brute-force cosine similarity. It needs
no native dependencies (no Chroma/sqlite), which makes it ideal for unit tests
and fully offline runs.
"""
from __future__ import annotations

import math
from typing import Optional

from contextual_rl_agentic_rag.retrieval.base import BaseVectorStore, content_hash
from contextual_rl_agentic_rag.schema import Document


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings
        # collection -> {content_hash: (Document, vector)}
        self._collections: dict[str, dict[str, tuple[Document, list[float]]]] = {}

    def add_documents(self, collection: str, docs: list[Document]) -> int:
        if not docs:
            return 0
        col = self._collections.setdefault(collection, {})
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        for doc, vec in zip(docs, vectors):
            col[content_hash(doc.page_content)] = (doc, vec)
        return len(docs)

    def search(
        self, collection: str, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ) -> list[Document]:
        return [doc for doc, _ in self.search_with_scores(collection, query, k, filter_dict)]

    def search_with_scores(
        self, collection: str, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        col = self._collections.get(collection, {})
        if not col:
            return []
        qv = self.embeddings.embed_query(query)
        scored: list[tuple[Document, float]] = []
        for doc, vec in col.values():
            if filter_dict and not all(doc.metadata.get(kk) == vv for kk, vv in filter_dict.items()):
                continue
            scored.append((doc, _cosine(qv, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def list_collections(self) -> list[dict]:
        return [{"name": name, "count": len(col)} for name, col in self._collections.items()]
