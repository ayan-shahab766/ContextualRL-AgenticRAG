"""OpenAI-backed LLM and embedding implementations.

The OpenAI client is created lazily on first use (never at import time) and all
network calls are wrapped with an explicit timeout and bounded retry/backoff.
"""
from __future__ import annotations

from contextual_rl_agentic_rag.llm.base import BaseLLMBackend, LLMError, call_with_retries
from contextual_rl_agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """Return retryable OpenAI exception types (import lazily)."""
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    except Exception:  # noqa: BLE001 - openai may be absent
        return (Exception,)


class OpenAILLMBackend(BaseLLMBackend):
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        vision_model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._vision_model = vision_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError(
                    "OPENAI_API_KEY is not set. Set it in the environment/.env or "
                    "use LLM_BACKEND=fake / LLM_BACKEND=local for offline runs."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def complete(
        self, system: str, user: str, *, json_mode: bool = False, model: str | None = None
    ) -> str:
        messages = [
            {"role": "system", "content": system + (" Respond in JSON." if json_mode else "")},
            {"role": "user", "content": user},
        ]
        kwargs: dict = {"model": model or self._model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _do():
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return call_with_retries(
            _do,
            retries=self._max_retries,
            exceptions=_transient_exceptions(),
            description="openai.chat.completions",
        )

    def vision(self, image_b64: str, prompt: str, *, model: str | None = None) -> str:
        def _do():
            resp = self.client.chat.completions.create(
                model=model or self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=800,
            )
            return resp.choices[0].message.content or ""

        return call_with_retries(
            _do,
            retries=self._max_retries,
            exceptions=_transient_exceptions(),
            description="openai.vision",
        )


class OpenAIEmbeddingBackend:
    def __init__(
        self, api_key: str, *, model: str, timeout: float = 30.0, max_retries: int = 3
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError("OPENAI_API_KEY is not set for embeddings.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _do():
            resp = self.client.embeddings.create(model=self._model, input=texts)
            return [d.embedding for d in resp.data]

        return call_with_retries(
            _do,
            retries=self._max_retries,
            exceptions=_transient_exceptions(),
            description="openai.embeddings",
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
