from contextual_rl_agentic_rag.agents.hybrid_rag import reciprocal_rank_fusion
from contextual_rl_agentic_rag.llm.fake_backend import FakeEmbeddingBackend
from contextual_rl_agentic_rag.retrieval.memory_store import InMemoryVectorStore
from contextual_rl_agentic_rag.schema import Document


def test_rrf_prefers_docs_in_both_lists():
    a = Document("alpha content here", {})
    b = Document("beta content here", {})
    c = Document("gamma content here", {})
    fused = reciprocal_rank_fusion(dense=[a, b], sparse=[b, c])
    assert fused[0].page_content == "beta content here"  # appears in both lists
    assert len(fused) == 3


def test_rrf_handles_empty_sparse():
    a = Document("only dense", {})
    fused = reciprocal_rank_fusion(dense=[a], sparse=[])
    assert [d.page_content for d in fused] == ["only dense"]


def test_memory_store_dedup_by_content():
    store = InMemoryVectorStore(FakeEmbeddingBackend())
    added = store.add_documents(
        "c", [Document("bandit exploration exploitation", {}), Document("bandit exploration exploitation", {})]
    )
    assert added == 2  # add reports processed count
    counts = {c["name"]: c["count"] for c in store.list_collections()}
    assert counts["c"] == 1  # but only one unique doc is stored


def test_memory_store_search_ranks_by_similarity():
    store = InMemoryVectorStore(FakeEmbeddingBackend())
    store.add_documents(
        "c",
        [
            Document("bandit exploration exploitation reward", {}),
            Document("completely unrelated cooking recipe", {}),
        ],
    )
    res = store.search("c", "bandit exploration reward", k=1)
    assert res and "bandit" in res[0].page_content


def test_memory_store_filter():
    store = InMemoryVectorStore(FakeEmbeddingBackend())
    store.add_documents(
        "c",
        [Document("schema info", {"chunk_type": "schema"}), Document("row info", {"chunk_type": "rows"})],
    )
    res = store.search_with_scores("c", "info", k=5, filter_dict={"chunk_type": "schema"})
    assert len(res) == 1
    assert res[0][0].metadata["chunk_type"] == "schema"
