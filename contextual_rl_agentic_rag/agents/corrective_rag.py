"""Corrective RAG (CRAG): fall back to web search when local retrieval is weak."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent, sanitize_sources
from contextual_rl_agentic_rag.llm.base import LLMError
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.schema import Document

logger = get_logger(__name__)


class CorrectiveRAGAgent(BaseRAGAgent):
    name = "corrective_rag"
    description = "Falls back to web search when local retrieval is insufficient."

    COLLECTION = "naive_rag"
    WEB_TIMEOUT = 10.0

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.search(self.COLLECTION, query, k=top_k)

    def _assess_retrieval(self, query: str, docs: list[Document]) -> str:
        if not docs:
            return "incorrect"
        ctx = "\n".join(d.page_content[:300] for d in docs[:3])
        try:
            result = self.llm.complete_json(
                "Assess if these documents can answer the question. "
                'Return: {"assessment": "correct|ambiguous|incorrect", "reason": "..."}',
                f"Question: {query}\n\nDocuments:\n{ctx}",
            )
        except LLMError as exc:
            logger.warning("Retrieval assessment failed, assuming ambiguous: %s", exc)
            return "ambiguous"
        return result.get("assessment", "ambiguous")

    async def _web_search_fallback(self, query: str) -> list[Document]:
        """Best-effort web fallback (DuckDuckGo). Swap for Tavily/Serper in prod."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.WEB_TIMEOUT) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": 1},
                )
                response.raise_for_status()
                data = response.json()
        except ImportError:
            logger.warning("httpx not installed; skipping web fallback.")
            return []
        except Exception as exc:  # noqa: BLE001 - external call, log and degrade
            logger.warning("Web fallback failed for %r: %s", query, exc)
            return []

        results: list[Document] = []
        for item in data.get("RelatedTopics", [])[:3]:
            if "Text" in item:
                results.append(
                    Document(
                        page_content=item["Text"],
                        metadata={"source": "web_search", "url": item.get("FirstURL", "")},
                    )
                )
        return results

    async def generate(self, query: str, context: list[Document]) -> str:
        ctx = "\n\n---\n\n".join(d.page_content for d in context)
        sources = {d.metadata.get("source", "local") for d in context}
        source_note = " (includes web results)" if "web_search" in sources else ""
        return self.llm.complete(
            f"Answer the question using context{source_note}. Clearly indicate if info comes from web search.",
            f"Context:\n{ctx}\n\nQuestion: {query}",
        )

    async def run(self, query: str, top_k: int = 5) -> dict:
        docs = await self.retrieve(query, top_k)
        assessment = self._assess_retrieval(query, docs)

        web_docs: list[Document] = []
        if assessment in ("incorrect", "ambiguous"):
            web_docs = await self._web_search_fallback(query)

        if assessment == "incorrect":
            final_docs = web_docs or docs
        elif assessment == "ambiguous":
            final_docs = docs + web_docs
        else:
            final_docs = docs

        answer = await self.generate(query, final_docs)
        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": sanitize_sources(final_docs),
            "num_sources": len(final_docs),
            "retrieval_assessment": assessment,
            "web_fallback_used": bool(web_docs),
        }
