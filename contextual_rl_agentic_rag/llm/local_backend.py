"""Local, no-API backend for offline runs.

- ``LocalEmbeddingBackend`` uses ``sentence-transformers`` when installed and
  transparently falls back to the deterministic hash embeddings otherwise, so
  imports never hard-fail in minimal environments.
- ``LocalLLMBackend`` is an extractive "LLM": for free-text answers it selects
  the context sentences most relevant to the question (token overlap). JSON-mode
  responses reuse the deterministic fake schema logic.
"""
from __future__ import annotations

import re

from contextual_rl_agentic_rag.llm.fake_backend import FakeEmbeddingBackend, FakeLLMBackend
from contextual_rl_agentic_rag.logging_config import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LocalLLMBackend(FakeLLMBackend):
    tag = "local"

    def complete(
        self, system: str, user: str, *, json_mode: bool = False, model: str | None = None
    ) -> str:
        if json_mode:
            import json

            return json.dumps(self._fake_json(system, user))

        context, question = self._split(user)
        if not context:
            return f"[{self.tag}-llm] {question[:200]}"

        q_tokens = set(question.lower().split())
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(context) if s.strip()]
        ranked = sorted(
            sentences,
            key=lambda s: len(q_tokens & set(s.lower().split())),
            reverse=True,
        )
        top = [s for s in ranked[:3] if s]
        if not top:
            top = sentences[:2]
        return " ".join(top) if top else f"[{self.tag}-llm] No relevant context found."

    @staticmethod
    def _split(user: str) -> tuple[str, str]:
        question = ""
        context = user
        if "Question:" in user:
            context, question = user.rsplit("Question:", 1)
        context = context.replace("Context:", "").strip()
        return context, question.strip()


class LocalEmbeddingBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._fallback = FakeEmbeddingBackend()

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sentence-transformers unavailable (%s); using hash embeddings fallback.",
                    exc,
                )
                self._model = False  # sentinel: use fallback
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model is False:
            return self._fallback.embed_documents(texts)
        return [list(map(float, v)) for v in self.model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:
        if self.model is False:
            return self._fallback.embed_query(text)
        return list(map(float, self.model.encode([text])[0]))
