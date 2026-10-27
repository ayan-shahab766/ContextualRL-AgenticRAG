"""Vector store abstraction shared by the Chroma and in-memory backends."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

from contextual_rl_agentic_rag.schema import Document


def content_hash(text: str) -> str:
    """Stable id used to deduplicate documents by content."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class BaseVectorStore(ABC):
    """Minimal vector store interface used by every agent."""

    @abstractmethod
    def add_documents(self, collection: str, docs: list[Document]) -> int: ...

    @abstractmethod
    def search(
        self, collection: str, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ) -> list[Document]: ...

    @abstractmethod
    def search_with_scores(
        self, collection: str, query: str, k: int = 5
    ) -> list[tuple[Document, float]]: ...

    @abstractmethod
    def list_collections(self) -> list[dict]: ...
