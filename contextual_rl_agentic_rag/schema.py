"""Lightweight document type shared across the system.

Deliberately decoupled from LangChain's ``Document`` so that core retrieval,
agent, and RL logic can be imported and unit-tested without installing the
heavy LangChain / Chroma / OpenAI stack. Adapters convert to/from LangChain
documents only inside the Chroma-backed store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A chunk of retrievable content plus arbitrary metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy_with(self, *, page_content: str | None = None, **extra_metadata: Any) -> "Document":
        """Return a shallow copy, optionally overriding content / adding metadata."""
        merged = {**self.metadata, **extra_metadata}
        return Document(
            page_content=self.page_content if page_content is None else page_content,
            metadata=merged,
        )
