"""Table RAG: answer analytical questions over structured data.

SECURITY NOTE: the original implementation executed LLM-generated pandas code via
``exec()``, which is a remote-code-execution vector (uploaded data can steer the
generated code). That has been removed. This agent now answers from the schema,
summary statistics, and row-group text that were captured at ingestion time. For
true executable analytics, run generated code in an isolated, resource-limited
sandbox (separate process/container) - out of scope here and intentionally not
done in-process.
"""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.schema import Document


class TableRAGAgent(BaseRAGAgent):
    name = "table_rag"
    description = "Answers questions over structured data (CSV/Excel) from schema, stats, and row groups."

    COLLECTION = "table_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query: str, context: list[Document]) -> str:
        schema_parts = [d.page_content for d in context if d.metadata.get("chunk_type") == "schema"]
        row_parts = [d.page_content for d in context if d.metadata.get("chunk_type") != "schema"]
        ctx = "\n\n".join(schema_parts + row_parts) or "\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "You are a careful data analyst. Answer the question using ONLY the table schema, "
            "summary statistics, and row samples provided. Show the reasoning behind any figure you "
            "cite. If the data provided is insufficient to compute the answer exactly, say what "
            "additional data or aggregation would be required.",
            f"Tabular context:\n{ctx}\n\nQuestion: {query}",
        )
