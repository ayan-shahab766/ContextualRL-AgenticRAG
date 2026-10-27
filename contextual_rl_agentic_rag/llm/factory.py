"""Backend selection based on configuration.

``LLM_BACKEND`` / ``EMBEDDING_BACKEND`` env vars choose between ``openai``,
``local``, and ``fake`` implementations. Callers depend on the protocols in
:mod:`contextual_rl_agentic_rag.llm.base`, never on the concrete classes.
"""
from __future__ import annotations

from contextual_rl_agentic_rag.config import Settings, settings as default_settings
from contextual_rl_agentic_rag.llm.base import EmbeddingBackend, LLMBackend
from contextual_rl_agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


def get_llm(settings: Settings | None = None) -> LLMBackend:
    settings = settings or default_settings
    backend = (settings.llm_backend or "openai").lower()

    if backend == "fake":
        from contextual_rl_agentic_rag.llm.fake_backend import FakeLLMBackend

        return FakeLLMBackend()
    if backend == "local":
        from contextual_rl_agentic_rag.llm.local_backend import LocalLLMBackend

        return LocalLLMBackend()
    if backend == "openai":
        from contextual_rl_agentic_rag.llm.openai_backend import OpenAILLMBackend

        return OpenAILLMBackend(
            settings.openai_api_key,
            model=settings.llm_model,
            vision_model=settings.vision_model,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected openai|local|fake)")


def get_embeddings(settings: Settings | None = None) -> EmbeddingBackend:
    settings = settings or default_settings
    backend = settings.effective_embedding_backend.lower()

    if backend == "fake":
        from contextual_rl_agentic_rag.llm.fake_backend import FakeEmbeddingBackend

        return FakeEmbeddingBackend()
    if backend == "local":
        from contextual_rl_agentic_rag.llm.local_backend import LocalEmbeddingBackend

        return LocalEmbeddingBackend()
    if backend == "openai":
        from contextual_rl_agentic_rag.llm.openai_backend import OpenAIEmbeddingBackend

        return OpenAIEmbeddingBackend(
            settings.openai_api_key,
            model=settings.embedding_model,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} (expected openai|local|fake)")
