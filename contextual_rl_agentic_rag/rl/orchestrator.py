"""RL orchestrator: classify -> route (bandit) -> run agent -> log -> learn from feedback."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from typing import Optional

from contextual_rl_agentic_rag.agents import build_agents
from contextual_rl_agentic_rag.deps import RagContext, build_context
from contextual_rl_agentic_rag.logging_config import get_logger
from contextual_rl_agentic_rag.rl.bandit_router import BanditRouter
from contextual_rl_agentic_rag.rl.query_signals import derive_query_signals
from contextual_rl_agentic_rag.rl.retrieval_policy import build_retrieval_plan
from contextual_rl_agentic_rag.rl.reward_store import load_all_interactions, log_interaction
from contextual_rl_agentic_rag.rl.self_improver import SelfImprover
from contextual_rl_agentic_rag.evidence import build_evidence

logger = get_logger(__name__)

# Cap on the in-memory interaction lookup cache (avoids full-log rescan on feedback).
_CACHE_SIZE = 2048
_AGENT_COST_UNITS = {
    "naive_rag": 1.0, "sentence_window_rag": 1.2, "parent_child_rag": 1.3,
    "hybrid_rag": 1.6, "table_rag": 1.7, "multimodal_rag": 2.0,
    "graph_rag": 2.4, "hyde_rag": 2.2, "corrective_rag": 2.5,
    "agentic_rag": 3.5, "lora_rag": 2.0,
}


class RLOrchestrator:
    def __init__(self, ctx: Optional[RagContext] = None, *, bandit: BanditRouter | None = None) -> None:
        self.ctx = ctx or build_context()
        self.ctx.settings.ensure_dirs()
        self.agents = build_agents(self.ctx)
        self.bandit = bandit or BanditRouter(base_dir=self.ctx.settings.base_dir)
        self.improver = SelfImprover(self.bandit)
        self._interaction_cache: "OrderedDict[str, dict]" = OrderedDict()

    def list_agents(self) -> list[dict]:
        return [{"name": a.name, "description": a.description} for a in self.agents.values()]

    def _cache_interaction(self, record: dict) -> None:
        self._interaction_cache[record["interaction_id"]] = record
        while len(self._interaction_cache) > _CACHE_SIZE:
            self._interaction_cache.popitem(last=False)

    async def query(self, question: str, agent_name: str | None = None, top_k: int | None = None) -> dict:
        signals = derive_query_signals(question)
        category = signals.category
        retrieval_plan = build_retrieval_plan(signals, top_k, self.ctx.settings.default_top_k)

        if agent_name:
            selected = agent_name
            bandit_info = {"mode": "manual_override", "category": category, "context_key": signals.context_key}
        else:
            selected, bandit_info = self.bandit.select_agent(category, context_key=signals.context_key)

        agent = self.agents.get(selected)
        if not agent:
            return {"error": f"Unknown agent: {selected}", "available": list(self.agents.keys())}

        try:
            started_at = time.perf_counter()
            result = await agent.run(question, retrieval_plan.top_k)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
        except Exception as exc:  # noqa: BLE001 - surface agent failures as structured errors
            logger.exception("Agent %s failed for question %r", selected, question[:80])
            return {"error": f"Agent {selected} failed: {exc}", "agent": selected, "category": category}

        interaction_id = str(uuid.uuid4())[:8]
        result["interaction_id"] = interaction_id
        result["category"] = category
        result["routing"] = bandit_info
        result["query_signals"] = signals.as_dict()
        result["retrieval_plan"] = retrieval_plan.as_dict()
        result["routed_to"] = selected
        result["execution"] = {
            "latency_ms": latency_ms,
            "estimated_cost_units": _AGENT_COST_UNITS.get(selected, 1.0),
        }
        result["evidence"] = build_evidence(result.get("answer", ""), result.get("sources", []))
        result["feedback_hint"] = (
            f"Rate this with POST /feedback: interaction_id={interaction_id}, reward=0.0-1.0"
        )

        record = {
            "interaction_id": interaction_id,
            "question": question,
            "agent": selected,
            "category": category,
            "context_key": signals.context_key,
            "query_signals": signals.as_dict(),
            "top_k": retrieval_plan.top_k,
            "num_sources": result.get("num_sources", 0),
            "latency_ms": latency_ms,
            "estimated_cost_units": _AGENT_COST_UNITS.get(selected, 1.0),
            "grounding_status": result["evidence"]["grounding"]["status"],
        }
        self._cache_interaction(record)
        log_interaction(record)
        return result

    async def compare(self, question: str, agent_names: list[str], top_k: int | None = None) -> dict:
        signals = derive_query_signals(question)
        category = signals.category
        retrieval_plan = build_retrieval_plan(signals, top_k, self.ctx.settings.default_top_k)
        valid = [n for n in agent_names if n in self.agents]
        tasks = [self.agents[n].run(question, retrieval_plan.top_k) for n in valid]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        comparison: dict[str, dict] = {}
        for name, result in zip(valid, results):
            if isinstance(result, Exception):
                logger.warning("Compare: agent %s failed: %s", name, result)
                comparison[name] = {"error": str(result)}
            else:
                result["evidence"] = build_evidence(result.get("answer", ""), result.get("sources", []))
                comparison[name] = result
        return {
            "question": question, "category": category, "query_signals": signals.as_dict(),
            "retrieval_plan": retrieval_plan.as_dict(), "comparisons": comparison,
        }

    def _find_interaction(self, interaction_id: str) -> dict | None:
        cached = self._interaction_cache.get(interaction_id)
        if cached:
            return cached
        for record in load_all_interactions():
            if record.get("interaction_id") == interaction_id and "reward" not in record:
                return record
        return None

    def submit_feedback(
        self,
        interaction_id: str,
        reward: float,
        *,
        faithfulness: float | None = None,
        citation_quality: float | None = None,
        usefulness: float | None = None,
    ) -> dict:
        original = self._find_interaction(interaction_id)
        if not original:
            return {"error": f"Interaction {interaction_id} not found"}

        quality = (
            0.5 * (usefulness if usefulness is not None else reward)
            + 0.3 * (faithfulness if faithfulness is not None else reward)
            + 0.2 * (citation_quality if citation_quality is not None else reward)
        )
        latency_penalty = min(float(original.get("latency_ms", 0)) / 30_000, 0.15)
        cost_penalty = min(float(original.get("estimated_cost_units", 1.0)) / 100, 0.08)
        routing_reward = max(0.0, min(1.0, quality - latency_penalty - cost_penalty))
        self.bandit.update(
            category=original["category"], agent=original["agent"], reward=routing_reward,
            context_key=original.get("context_key"),
        )
        log_interaction(
            {
                "interaction_id": interaction_id,
                "type": "feedback",
                "agent": original["agent"],
                "category": original["category"],
                "question": original.get("question", ""),
                "reward": reward,
                "routing_reward": routing_reward,
                "feedback_dimensions": {
                    "usefulness": usefulness, "faithfulness": faithfulness,
                    "citation_quality": citation_quality,
                },
            }
        )
        return {
            "status": "feedback_recorded",
            "interaction_id": interaction_id,
            "agent": original["agent"],
            "category": original["category"],
            "reward": reward,
            "routing_reward": round(routing_reward, 3),
            "reward_components": {
                "quality": round(quality, 3), "latency_penalty": round(latency_penalty, 3),
                "cost_penalty": round(cost_penalty, 3),
            },
            "updated_agent_stats": self.bandit.state.get(original["category"], {}).get(
                original["agent"], {}
            ),
        }
