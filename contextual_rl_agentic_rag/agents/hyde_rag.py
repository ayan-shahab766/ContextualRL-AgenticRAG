"""HyDE RAG: generate a hypothetical answer, then retrieve with it."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


class HyDERAGAgent(BaseRAGAgent):
    name = "hyde_rag"
    description = "Generates hypothetical answer first, uses it for retrieval. Bridges query-document gap."

    COLLECTION = "naive_rag"

    def _generate_hypothesis(self, query: str) -> str:
        return self.llm.complete(
            "Write a detailed, factual paragraph that would perfectly answer this question. "
            "This will be used for document retrieval, so make it specific and information-dense. "
            "Don't hedge - write as if you know the answer.",
            query,
        )

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        hypothesis = self._generate_hypothesis(query)
        hyde_results = self.store.search(self.COLLECTION, hypothesis, k=top_k)
        original_results = self.store.search(self.COLLECTION, query, k=top_k)

        seen: set[str] = set()
        merged: list[Document] = []
        for doc in hyde_results + original_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged[:top_k]

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "Answer using the provided context. Ignore any prior assumptions.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )
