"""Sentence Window RAG: embed sentences, return the surrounding window."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


class SentenceWindowRAGAgent(BaseRAGAgent):
    name = "sentence_window_rag"
    description = "Embeds sentences, retrieves surrounding window for better context."

    COLLECTION = "sentence_window_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        results = self.store.search(self.COLLECTION, query, k=top_k)
        expanded: list[Document] = []
        seen_windows: set[str] = set()
        for doc in results:
            window = doc.metadata.get("window_text", doc.page_content)
            if window not in seen_windows:
                seen_windows.add(window)
                expanded.append(Document(page_content=window, metadata=doc.metadata))
        return expanded

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "Answer using the context. Each context block is a window around a relevant sentence.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )
