"""Base RAG agent with dependency injection and consistent source sanitization."""
from __future__ import annotations

from abc import ABC, abstractmethod

from contextual_rl_agentic_rag.deps import RagContext
from contextual_rl_agentic_rag.schema import Document

# Metadata keys that must never be echoed back to API clients.
SENSITIVE_METADATA_KEYS = {
    "parent_content",
    "window_text",
    "image_b64_preview",
    "csv_path",
    "image_path",
}


def sanitize_sources(docs: list[Document], content_chars: int = 300) -> list[dict]:
    """Trim content and strip sensitive/large metadata from retrieved docs."""
    out = []
    for d in docs:
        out.append(
            {
                "content": d.page_content[:content_chars],
                "metadata": {
                    k: v for k, v in d.metadata.items() if k not in SENSITIVE_METADATA_KEYS
                },
            }
        )
    return out


class BaseRAGAgent(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self, ctx: RagContext) -> None:
        self.ctx = ctx

    @property
    def llm(self):
        return self.ctx.llm

    @property
    def store(self):
        return self.ctx.vector_store

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]: ...

    @abstractmethod
    async def generate(self, query: str, context: list[Document]) -> str: ...

    async def run(self, query: str, top_k: int = 5) -> dict:
        docs = await self.retrieve(query, top_k)
        answer = await self.generate(query, docs)
        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": sanitize_sources(docs),
            "num_sources": len(docs),
        }
