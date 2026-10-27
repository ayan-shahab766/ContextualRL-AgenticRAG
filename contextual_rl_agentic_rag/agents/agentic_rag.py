"""Agentic RAG (self-correcting): retrieve -> grade -> reformulate -> generate -> check."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent, sanitize_sources
from contextual_rl_agentic_rag.llm.base import LLMError
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.schema import Document

logger = get_logger(__name__)


class AgenticRAGAgent(BaseRAGAgent):
    name = "agentic_rag"
    description = "Self-correcting RAG with relevance grading, query reformulation, and hallucination checks."

    COLLECTION = "naive_rag"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.search(self.COLLECTION, query, k=top_k)

    def _grade_relevance(self, query: str, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        relevant, irrelevant = [], []
        for doc in docs:
            try:
                result = self.llm.complete_json(
                    "Grade if this document is relevant to the question. "
                    'Return: {"relevant": true/false, "reason": "brief reason"}',
                    f"Question: {query}\nDocument: {doc.page_content[:600]}",
                )
            except LLMError as exc:
                logger.warning("Relevance grading failed, treating as relevant: %s", exc)
                relevant.append(doc)
                continue
            if result.get("relevant", False):
                doc.metadata["grade_reason"] = result.get("reason", "")
                relevant.append(doc)
            else:
                irrelevant.append(doc)
        return relevant, irrelevant

    def _reformulate(self, query: str, attempt: int) -> str:
        return self.llm.complete(
            "Rewrite this search query to find better, more relevant results. "
            "Try a different angle or phrasing. Return ONLY the new query.",
            f"Original query (attempt {attempt}): {query}",
        ).strip()

    def _check_hallucination(self, answer: str, docs: list[Document]) -> dict:
        ctx = "\n".join(d.page_content[:400] for d in docs)
        try:
            return self.llm.complete_json(
                "Check if the answer is fully supported by the context. "
                'Return: {"grounded": true/false, "unsupported_claims": ["..."]}',
                f"Context:\n{ctx}\n\nAnswer: {answer}",
            )
        except LLMError as exc:
            logger.warning("Hallucination check failed: %s", exc)
            return {"grounded": True, "unsupported_claims": []}

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        return self.llm.complete(
            "Answer ONLY based on the context. Cite specific parts that support your answer. "
            "If context is insufficient, clearly state what's missing.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )

    async def run(self, query: str, top_k: int = 5) -> dict:
        trace: list[dict] = []
        current_query = query
        max_retries = self.ctx.settings.max_agentic_retries
        docs: list[Document] = []
        relevant: list[Document] = []
        attempt = 0

        for attempt in range(max_retries):
            docs = await self.retrieve(current_query, top_k)
            trace.append(
                {"step": "retrieve", "attempt": attempt + 1, "query": current_query, "docs_found": len(docs)}
            )
            relevant, irrelevant = self._grade_relevance(query, docs)
            trace.append({"step": "grade", "relevant": len(relevant), "irrelevant": len(irrelevant)})

            if len(relevant) >= 2 or attempt == max_retries - 1:
                break
            current_query = self._reformulate(current_query, attempt + 1)
            trace.append({"step": "reformulate", "new_query": current_query})

        final_docs = relevant if relevant else docs
        answer = await self.generate(query, final_docs)
        trace.append({"step": "generate"})

        hall_check = self._check_hallucination(answer, final_docs)
        trace.append({"step": "hallucination_check", **hall_check})

        if not hall_check.get("grounded", True):
            answer = await self.generate(query, final_docs)
            answer += "\n\n[Note: Answer was regenerated after hallucination detection]"
            trace.append({"step": "regenerate"})

        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": sanitize_sources(final_docs),
            "num_sources": len(final_docs),
            "grounded": hall_check.get("grounded", True),
            "retrieval_attempts": attempt + 1,
            "trace": trace,
        }
