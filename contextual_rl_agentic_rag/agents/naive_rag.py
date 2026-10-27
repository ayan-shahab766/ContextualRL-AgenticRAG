"""Naive RAG (baseline): chunk -> embed -> retrieve top-k -> generate."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


class NaiveRAGAgent(BaseRAGAgent):
    name = "naive_rag"
    description = "Basic chunk-retrieve-generate. The baseline RAG approach."

    COLLECTION = "naive_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "Answer based on the context provided. If unsure, say so.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )
