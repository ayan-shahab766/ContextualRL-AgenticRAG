"""Graph RAG: extract query entities, traverse the knowledge graph, then answer."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.base import BaseRAGAgent, sanitize_sources
from contextual_rl_agentic_rag.llm.base import LLMError
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.schema import Document

logger = get_logger(__name__)

_EMPTY_SUBGRAPH = {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}


class GraphRAGAgent(BaseRAGAgent):
    name = "graph_rag"
    description = "Uses knowledge graph to answer questions about entities and relationships."

    COLLECTION = "naive_rag"

    def _extract_entities(self, query: str) -> list[str]:
        try:
            result = self.llm.complete_json(
                'Extract key entities from this question. Return: {"entities": ["entity1", "entity2"]}',
                query,
            )
        except LLMError as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return []
        return result.get("entities", [])

    async def retrieve(self, query: str, top_k: int = 5) -> tuple[list[Document], dict]:
        entities = self._extract_entities(query)
        if self.ctx.kg is not None:
            subgraph = self.ctx.kg.query_subgraph(entities, depth=2)
        else:
            subgraph = dict(_EMPTY_SUBGRAPH)
        text_docs = self.store.search(self.COLLECTION, query, k=top_k)
        return text_docs, subgraph

    async def generate(self, query: str, context: tuple[list[Document], dict]) -> str:
        text_docs, subgraph = context
        text_ctx = "\n\n".join(d.page_content for d in text_docs)

        graph_ctx = "Knowledge Graph:\n"
        for node in subgraph.get("nodes", []):
            graph_ctx += f"  Entity: {node['name']} (type: {node.get('type', '?')}): {node.get('description', '')}\n"
        for edge in subgraph.get("edges", []):
            graph_ctx += f"  Relation: {edge['source']} --[{edge.get('relation', '')}]--> {edge['target']}\n"

        return self.llm.complete(
            "Answer using both the text context and the knowledge graph. "
            "The graph shows entities and their relationships. Prefer graph data for relationship questions.",
            f"Text Context:\n{text_ctx}\n\n{graph_ctx}\n\nQuestion: {query}",
        )

    async def run(self, query: str, top_k: int = 5) -> dict:
        text_docs, subgraph = await self.retrieve(query, top_k)
        answer = await self.generate(query, (text_docs, subgraph))
        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": sanitize_sources(text_docs),
            "graph_context": subgraph,
            "num_sources": len(text_docs),
        }
