"""RAG agents and a registry for building them from a shared context."""
from __future__ import annotations

from contextual_rl_agentic_rag.agents.agentic_rag import AgenticRAGAgent
from contextual_rl_agentic_rag.agents.base import BaseRAGAgent
from contextual_rl_agentic_rag.agents.corrective_rag import CorrectiveRAGAgent
from contextual_rl_agentic_rag.agents.graph_rag import GraphRAGAgent
from contextual_rl_agentic_rag.agents.hybrid_rag import HybridRAGAgent
from contextual_rl_agentic_rag.agents.hyde_rag import HyDERAGAgent
from contextual_rl_agentic_rag.agents.lora_rag import LoraRAGAgent
from contextual_rl_agentic_rag.agents.multimodal_rag import MultimodalRAGAgent
from contextual_rl_agentic_rag.agents.naive_rag import NaiveRAGAgent
from contextual_rl_agentic_rag.agents.parent_child_rag import ParentChildRAGAgent
from contextual_rl_agentic_rag.agents.sentence_window_rag import SentenceWindowRAGAgent
from contextual_rl_agentic_rag.agents.table_rag import TableRAGAgent
from contextual_rl_agentic_rag.deps import RagContext

AGENT_CLASSES: tuple[type[BaseRAGAgent], ...] = (
    NaiveRAGAgent,
    SentenceWindowRAGAgent,
    ParentChildRAGAgent,
    AgenticRAGAgent,
    MultimodalRAGAgent,
    TableRAGAgent,
    GraphRAGAgent,
    HybridRAGAgent,
    HyDERAGAgent,
    CorrectiveRAGAgent,
    LoraRAGAgent,
)

# Canonical agent name ordering used by the RL router.
AGENT_NAMES: list[str] = [cls.name for cls in AGENT_CLASSES]


def build_agents(ctx: RagContext) -> dict[str, BaseRAGAgent]:
    """Instantiate every agent wired to the shared context."""
    return {cls.name: cls(ctx) for cls in AGENT_CLASSES}


__all__ = [
    "AGENT_CLASSES",
    "AGENT_NAMES",
    "BaseRAGAgent",
    "build_agents",
    "AgenticRAGAgent",
    "CorrectiveRAGAgent",
    "GraphRAGAgent",
    "HybridRAGAgent",
    "HyDERAGAgent",
    "LoraRAGAgent",
    "MultimodalRAGAgent",
    "NaiveRAGAgent",
    "ParentChildRAGAgent",
    "SentenceWindowRAGAgent",
    "TableRAGAgent",
]
