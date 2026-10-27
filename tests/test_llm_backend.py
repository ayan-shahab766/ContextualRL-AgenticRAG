import pytest

from contextual_rl_agentic_rag.llm.base import LLMError, extract_json
from contextual_rl_agentic_rag.llm.fake_backend import FakeEmbeddingBackend, FakeLLMBackend


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_code_fenced():
    assert extract_json('```json\n{"a": 1, "b": 2}\n```') == {"a": 1, "b": 2}


def test_extract_json_embedded_in_prose():
    assert extract_json('Sure, here it is: {"ok": true} - hope that helps') == {"ok": True}


def test_extract_json_empty_raises():
    with pytest.raises(LLMError):
        extract_json("")


def test_extract_json_garbage_raises():
    with pytest.raises(LLMError):
        extract_json("no json here at all")


def test_fake_complete_json_relevance_schema():
    result = FakeLLMBackend().complete_json("Grade if this document is relevant.", "q")
    assert result["relevant"] is True


def test_fake_complete_json_hallucination_schema():
    result = FakeLLMBackend().complete_json("Check if the answer is grounded.", "ctx")
    assert result["grounded"] is True


def test_fake_complete_text_is_deterministic():
    b = FakeLLMBackend()
    prompt = "Context:\nfoo bar\n\nQuestion: what is foo?"
    assert b.complete("sys", prompt) == b.complete("sys", prompt)


def test_fake_embeddings_shared_tokens_more_similar():
    emb = FakeEmbeddingBackend()
    v_query = emb.embed_query("bandit exploration reward")
    docs = emb.embed_documents(["bandit exploration reward signal", "totally different cooking topic"])
    # cosine with itself-ish text should beat the unrelated one
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    assert cos(v_query, docs[0]) > cos(v_query, docs[1])
