"""LoRA RAG: naive retrieval, generation via local LoRA adapters when available.

Falls back to the configured LLM backend if the training package / adapters are
not present.
"""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.schema import Document

logger = get_logger(__name__)


class LoraRAGAgent(BaseRAGAgent):
    name = "lora_rag"
    description = "RAG with optional local LoRA-generated answers; falls back to backend LLM if no adapters."

    COLLECTION = "naive_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        prompt = f"Answer based on the context provided. If unsure, say so.\nContext:\n{ctx}\n\nQuestion: {query}"
        try:
            from training.inference import generate as local_generate

            out = local_generate(prompt)
            if out:
                return out
        except Exception as exc:  # noqa: BLE001 - training deps/adapters optional
            logger.debug("Local LoRA generation unavailable (%s); using backend LLM.", exc)
        return self.llm.complete("Answer based on the context provided. If unsure, say so.", prompt)
