"""Deterministic, offline, no-cost backend for tests and skip-live runs.

- ``FakeLLMBackend`` returns stable, structured responses. For ``json_mode`` it
  inspects the system prompt and returns schema-appropriate JSON so the agent
  control flow (relevance grading, hallucination checks, routing, etc.) exercises
  its real code paths without any network call.
- ``FakeEmbeddingBackend`` produces deterministic bag-of-words hash vectors, so
  documents that share tokens get higher cosine similarity - enough signal for
  the in-memory vector store to return sensible results in tests.
"""
from __future__ import annotations

import hashlib
import math

from contextual_rl_agentic_rag.llm.base import BaseLLMBackend


def _extract_question(user: str) -> str:
    marker = "Question:"
    if marker in user:
        return user.split(marker, 1)[1].strip()
    return user.strip()


class FakeLLMBackend(BaseLLMBackend):
    tag = "fake"

    def complete(
        self, system: str, user: str, *, json_mode: bool = False, model: str | None = None
    ) -> str:
        if json_mode:
            import json

            return json.dumps(self._fake_json(system, user))
        question = _extract_question(user)
        snippet = user.strip().replace("\n", " ")[:160]
        return (
            f"[{self.tag}-llm] Answer to '{question[:120]}' grounded in context: {snippet}"
        )

    def vision(self, image_b64: str, prompt: str, *, model: str | None = None) -> str:
        digest = hashlib.md5(image_b64.encode()).hexdigest()[:8]
        return f"[{self.tag}-vision] Synthetic description of image {digest}."

    def _fake_json(self, system: str, user: str) -> dict:
        s = system.lower()
        if "relevant" in s:
            return {"relevant": True, "reason": f"{self.tag}: matches query"}
        if "grounded" in s or "hallucinat" in s:
            return {"grounded": True, "unsupported_claims": []}
        if "assess" in s:
            return {"assessment": "correct", "reason": f"{self.tag}: sufficient"}
        if "entities" in s and "relationship" in s:
            return {"entities": [], "relationships": []}
        if "entities" in s:
            return {"entities": []}
        if "agent" in s and "route" in s:
            return {"agent": "naive_rag", "reason": f"{self.tag}: default route"}
        if "scores" in s:
            return {"scores": {}, "winner": "", "explanation": f"{self.tag}: no judgement"}
        return {}


class FakeEmbeddingBackend:
    """Deterministic hash-based embeddings with real token-overlap signal."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
