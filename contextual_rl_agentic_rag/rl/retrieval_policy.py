"""Adaptive retrieval planning driven by explainable query signals."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from contextual_rl_agentic_rag.rl.query_signals import QuerySignals


@dataclass(frozen=True)
class RetrievalPlan:
    top_k: int
    rerank_required: bool
    corrective_fallback_eligible: bool
    rationale: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def build_retrieval_plan(signals: QuerySignals, requested_top_k: int | None, default_top_k: int) -> RetrievalPlan:
    """Choose a bounded retrieval depth while respecting an explicit API override."""
    rationale: list[str] = []
    if requested_top_k is not None:
        top_k = max(1, min(requested_top_k, 12))
        rationale.append("caller_override")
    else:
        top_k = default_top_k
        if signals.complexity == "high":
            top_k += 2
            rationale.append("high_complexity_expansion")
        elif signals.complexity == "low":
            top_k -= 1
            rationale.append("low_complexity_budget")
        if signals.modality == "tabular":
            top_k += 1
            rationale.append("tabular_context_expansion")
        top_k = max(3, min(top_k, 8))

    return RetrievalPlan(
        top_k=top_k,
        rerank_required=signals.complexity == "high" or signals.category in {"analytical", "relational"},
        corrective_fallback_eligible=signals.complexity != "low" or signals.category == "vague",
        rationale=rationale or ["default_retrieval_budget"],
    )
