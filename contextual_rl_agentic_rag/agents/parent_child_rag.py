"""Parent-Child RAG: search small child chunks, return larger parent documents."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


class ParentChildRAGAgent(BaseRAGAgent):
    name = "parent_child_rag"
    description = "Searches small child chunks but returns their larger parent documents."

    CHILD_COLLECTION = "parent_child_rag_children"
    PARENT_COLLECTION = "parent_child_rag_parents"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        children = self.store.search(self.CHILD_COLLECTION, query, k=top_k * 2)
        seen_parents: dict[str, Document] = {}
        for child in children:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_parents:
                parent_content = child.metadata.get("parent_content", child.page_content)
                seen_parents[pid] = Document(
                    page_content=parent_content,
                    metadata={**child.metadata, "retrieval_method": "parent_from_child"},
                )
        return list(seen_parents.values())[:top_k]

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "Answer using the full document sections provided as context.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )
