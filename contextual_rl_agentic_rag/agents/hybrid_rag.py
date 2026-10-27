"""Hybrid RAG: dense (embeddings) + sparse (BM25) retrieval fused with RRF."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


def reciprocal_rank_fusion(dense: list[Document], sparse: list[Document], k: int = 60) -> list[Document]:
    """RRF: score = sum(1 / (k + rank)) across both ranked lists."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in (dense, sparse):
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ordered]


class HybridRAGAgent(BaseRAGAgent):
    name = "hybrid_rag"
    description = "Combines dense (embedding) + sparse (BM25) retrieval with rank fusion."

    COLLECTION = "naive_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        dense = self.store.search(self.COLLECTION, query, k=top_k * 2)
        sparse = self.ctx.bm25.search(query, k=top_k * 2) if self.ctx.bm25 else []
        return reciprocal_rank_fusion(dense, sparse)[:top_k]

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "Answer based on context retrieved via hybrid search (semantic + keyword matching).",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )
