"""Sparse (BM25) retrieval for hybrid search.

Dense embeddings miss exact keyword matches; BM25 catches them. ``rank-bm25`` is
imported lazily, the index is rebuilt lazily (only when stale), and the corpus is
persisted with an atomic write.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from contextual_rl_agentic_rag.retrieval.base import content_hash
from contextual_rl_agentic_rag.schema import Document
from contextual_rl_agentic_rag.utils import atomic_write_bytes


class BM25Store:
    def __init__(self, name: str, *, base_dir: str | Path = "./data") -> None:
        self.name = name
        self.path = Path(base_dir) / f"bm25_{name}.pkl"
        self.documents: list[Document] = []
        self._bm25 = None
        self._dirty = True
        self._seen: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.documents = data.get("docs", [])
            self._seen = {content_hash(d.page_content) for d in self.documents}
            self._dirty = True
        except (OSError, pickle.UnpicklingError, EOFError):
            # Corrupt/incompatible cache: start clean rather than crash.
            self.documents = []
            self._seen = set()

    def _save(self) -> None:
        atomic_write_bytes(self.path, pickle.dumps({"docs": self.documents}))

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        if self.documents:
            from rank_bm25 import BM25Okapi

            tokenized = [d.page_content.lower().split() for d in self.documents]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        self._dirty = False

    def add_documents(self, docs: list[Document]) -> int:
        added = 0
        for d in docs:
            h = content_hash(d.page_content)
            if h in self._seen:
                continue
            self._seen.add(h)
            self.documents.append(d)
            added += 1
        if added:
            self._dirty = True
            self._save()
        return added

    def search(self, query: str, k: int = 5) -> list[Document]:
        self._ensure_index()
        if not self._bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top if scores[i] > 0]
