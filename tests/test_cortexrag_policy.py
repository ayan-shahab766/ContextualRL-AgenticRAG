import asyncio

from contextual_rl_agentic_rag.rl.bandit_router import BanditRouter
from contextual_rl_agentic_rag.rl.orchestrator import RLOrchestrator
from contextual_rl_agentic_rag.rl.query_signals import derive_query_signals
from contextual_rl_agentic_rag.rl.retrieval_policy import build_retrieval_plan


def test_query_signals_form_a_stable_context_key():
    signals = derive_query_signals("Compare the Python API latency metrics in this table")
    assert signals.domain == "technical"
    assert signals.modality == "tabular"
    assert signals.context_key == "analytical|technical|tabular|medium"


def test_adaptive_plan_expands_high_complexity_queries():
    signals = derive_query_signals("Analyze the relationship between API latency, cost, and quality trade-offs across the dataset")
    plan = build_retrieval_plan(signals, None, 5)
    assert plan.top_k >= 7
    assert plan.rerank_required is True


def test_contextual_bandit_persists_separate_policy_state(tmp_path):
    bandit = BanditRouter(base_dir=tmp_path)
    agent, info = bandit.select_agent("factual", context_key="factual|technical|text|low")
    bandit.update("factual", agent, 0.8, context_key=info["context_key"])
    assert info["policy"] == "contextual_ucb1"
    assert info["context_key"] in bandit.get_contextual_stats()


def test_query_returns_evidence_and_execution_diagnostics(fake_ctx, sample_docs):
    fake_ctx.vector_store.add_documents("naive_rag", sample_docs)
    orch = RLOrchestrator(ctx=fake_ctx)
    response = asyncio.run(orch.query("What is RAG?", agent_name="naive_rag", top_k=2))
    assert response["evidence"]["items"]
    assert response["evidence"]["grounding"]["method"] == "retrieval_lexical_overlap"
    assert response["execution"]["latency_ms"] >= 0
