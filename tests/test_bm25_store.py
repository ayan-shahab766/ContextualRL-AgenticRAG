import pytest

pytest.importorskip("rank_bm25")

from contextual_rl_agentic_rag.retrieval.bm25_store import BM25Store  # noqa: E402
from contextual_rl_agentic_rag.schema import Document  # noqa: E402


def test_bm25_keyword_search(tmp_path):
    # A few docs so BM25 IDF is positive for the rare query terms.
    store = BM25Store("t", base_dir=tmp_path)
    store.add_documents(
        [
            Document("the bandit explores and exploits arms to maximize reward", {}),
            Document("a pasta cooking recipe with tomatoes", {}),
            Document("weather forecast for the weekend", {}),
            Document("stock market prices rose today", {}),
            Document("gardening tips for spring planting", {}),
        ]
    )
    res = store.search("bandit exploits arms", k=1)
    assert res and "bandit" in res[0].page_content


def test_bm25_dedup_and_persistence(tmp_path):
    store = BM25Store("t", base_dir=tmp_path)
    assert store.add_documents([Document("alpha beta gamma", {})]) == 1
    assert store.add_documents([Document("alpha beta gamma", {})]) == 0  # duplicate ignored

    reloaded = BM25Store("t", base_dir=tmp_path)
    assert len(reloaded.documents) == 1
