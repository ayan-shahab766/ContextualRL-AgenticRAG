"""Multimodal RAG: text + image-derived descriptions in a unified prompt."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


class MultimodalRAGAgent(BaseRAGAgent):
    name = "multimodal_rag"
    description = "Handles images and text together. Images are described by vision LLM for retrieval."

    COLLECTION = "multimodal_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query: str, context: list[Document]) -> str:
        text_parts, image_parts = [], []
        for doc in context:
            if doc.metadata.get("modality") == "image":
                image_parts.append(f"[IMAGE: {doc.page_content}]")
            else:
                text_parts.append(doc.page_content)

        ctx = "\n\n---\n\n".join(text_parts + image_parts)
        return self.llm.complete(
            "Answer using the context which may include image descriptions marked with [IMAGE]. "
            "Reference visual elements when relevant.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )
