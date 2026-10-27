import asyncio

from contextual_rl_agentic_rag.rl.orchestrator import RLOrchestrator
from contextual_rl_agentic_rag.schema import Document


def test_forced_agent_query_and_feedback(fake_ctx, sample_docs):
    fake_ctx.vector_store.add_documents("naive_rag", sample_docs)
    orch = RLOrchestrator(ctx=fake_ctx)

    res = asyncio.run(orch.query("What is RAG?", agent_name="naive_rag", top_k=2))
    assert res["routed_to"] == "naive_rag"
    assert res["num_sources"] >= 1
    assert res["category"] == "factual"
    assert "interaction_id" in res

    fb = orch.submit_feedback(res["interaction_id"], 0.9)
    assert fb["status"] == "feedback_recorded"
    assert fb["reward"] == 0.9
    assert fb["agent"] == "naive_rag"


def test_bandit_routes_to_registered_agent(fake_ctx, sample_docs):
    fake_ctx.vector_store.add_documents("naive_rag", sample_docs)
    orch = RLOrchestrator(ctx=fake_ctx)
    res = asyncio.run(orch.query("What is a bandit?", top_k=2))
    assert res["routed_to"] in orch.agents
    assert res["routing"]["category"] == "factual"


def test_unknown_forced_agent_returns_error(fake_ctx):
    orch = RLOrchestrator(ctx=fake_ctx)
    res = asyncio.run(orch.query("hello", agent_name="does_not_exist"))
    assert "error" in res


def test_feedback_unknown_interaction(fake_ctx):
    orch = RLOrchestrator(ctx=fake_ctx)
    res = orch.submit_feedback("deadbeef", 1.0)
    assert "error" in res


def test_compare_runs_multiple_agents(fake_ctx, sample_docs):
    fake_ctx.vector_store.add_documents("naive_rag", sample_docs)
    orch = RLOrchestrator(ctx=fake_ctx)
    res = asyncio.run(orch.compare("What is RAG?", ["naive_rag", "hyde_rag"], top_k=2))
    assert set(res["comparisons"].keys()) == {"naive_rag", "hyde_rag"}
    for payload in res["comparisons"].values():
        assert "answer" in payload


def test_sources_metadata_is_sanitized(fake_ctx):
    doc = Document(
        "sensitive body text",
        {"parent_content": "HUGE PARENT", "window_text": "win", "image_b64_preview": "AAAA", "page": 7},
    )
    fake_ctx.vector_store.add_documents("naive_rag", [doc])
    orch = RLOrchestrator(ctx=fake_ctx)
    res = asyncio.run(orch.query("body?", agent_name="naive_rag", top_k=1))
    md = res["sources"][0]["metadata"]
    assert "parent_content" not in md
    assert "window_text" not in md
    assert "image_b64_preview" not in md
    assert md.get("page") == 7


def test_agentic_agent_produces_trace(fake_ctx, sample_docs):
    fake_ctx.vector_store.add_documents("naive_rag", sample_docs)
    orch = RLOrchestrator(ctx=fake_ctx)
    res = asyncio.run(orch.query("Explain how bandits work", agent_name="agentic_rag", top_k=3))
    assert res["routed_to"] == "agentic_rag"
    assert isinstance(res.get("trace"), list) and res["trace"]
    assert "grounded" in res
