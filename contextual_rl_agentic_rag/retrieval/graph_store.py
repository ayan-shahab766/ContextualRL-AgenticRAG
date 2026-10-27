"""Knowledge graph for Graph RAG.

Entities + relationships capture structure that flat chunks miss. Entity/relation
extraction is delegated to an injected LLM backend (no global client), and the
graph is persisted atomically. ``networkx`` is imported lazily.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from contextual_rl_agentic_rag.llm.base import LLMBackend, LLMError
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.utils import atomic_write_bytes

logger = get_logger(__name__)

_EXTRACT_SYSTEM = (
    "Extract entities and relationships from this text. Return JSON: {"
    '"entities": [{"name": "...", "type": "person|org|concept|location|event", "description": "..."}], '
    '"relationships": [{"source": "...", "target": "...", "relation": "...", "description": "..."}]}. '
    "Extract ALL meaningful entities and relationships. Be thorough."
)


class KnowledgeGraph:
    def __init__(self, *, llm: LLMBackend, path: str | Path = "./data/knowledge_graph.pkl") -> None:
        import networkx as nx

        self._nx = nx
        self.llm = llm
        self.path = Path(path)
        self.graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "rb") as f:
                self.graph = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.warning("Could not load knowledge graph at %s; starting empty.", self.path)
            self.graph = self._nx.DiGraph()

    def _save(self) -> None:
        atomic_write_bytes(self.path, pickle.dumps(self.graph))

    def extract_and_add(self, text: str, source: str) -> None:
        """Extract entities/relations via the LLM and merge into the graph."""
        try:
            result = self.llm.complete_json(_EXTRACT_SYSTEM, text[:3000])
        except LLMError as exc:
            logger.warning("Graph extraction failed for %s: %s", source, exc)
            return

        for entity in result.get("entities", []):
            name = (entity.get("name") or "").strip().lower()
            if not name:
                continue
            self.graph.add_node(
                name,
                type=entity.get("type", "unknown"),
                description=entity.get("description", ""),
                source=source,
            )

        for rel in result.get("relationships", []):
            src = (rel.get("source") or "").strip().lower()
            tgt = (rel.get("target") or "").strip().lower()
            if not src or not tgt:
                continue
            self.graph.add_edge(
                src,
                tgt,
                relation=rel.get("relation", "related_to"),
                description=rel.get("description", ""),
                source=source,
            )
        self._save()

    def query_subgraph(self, entities: list[str], depth: int = 2) -> dict:
        relevant_nodes: set[str] = set()
        for entity in entities:
            entity_lower = entity.lower()
            matches = [n for n in self.graph.nodes if entity_lower in n or n in entity_lower]
            for match in matches:
                for d in range(depth + 1):
                    neighbors = self._nx.single_source_shortest_path_length(
                        self.graph, match, cutoff=d
                    )
                    relevant_nodes.update(neighbors.keys())

        subgraph = self.graph.subgraph(relevant_nodes)
        nodes = [{"name": n, **self.graph.nodes[n]} for n in subgraph.nodes]
        edges = [{"source": u, "target": v, **d} for u, v, d in subgraph.edges(data=True)]
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def get_stats(self) -> dict:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
