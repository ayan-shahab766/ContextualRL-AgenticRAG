"""LLM / embedding backend protocols and shared helpers.

The rest of the system depends only on these abstractions, never on a concrete
provider. This is what makes the system testable (inject a fake backend) and
runnable offline (inject a local backend).
"""
from __future__ import annotations

import json
import time
from typing import Callable, Protocol, TypeVar, runtime_checkable

from contextual_rl_agentic_rag.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class LLMError(RuntimeError):
    """Raised when an LLM/embedding call fails or returns unusable output."""


@runtime_checkable
class LLMBackend(Protocol):
    """Text + vision generation."""

    def complete(
        self, system: str, user: str, *, json_mode: bool = False, model: str | None = None
    ) -> str: ...

    def complete_json(self, system: str, user: str, *, model: str | None = None) -> dict: ...

    def vision(self, image_b64: str, prompt: str, *, model: str | None = None) -> str: ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Embedding provider compatible with LangChain's duck-typed interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def extract_json(text: str | None) -> dict:
    """Parse a JSON object from raw model output, tolerating code fences / prose.

    Raises :class:`LLMError` if no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise LLMError("LLM returned empty output where JSON was expected")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` fences.
        cleaned = cleaned.split("```", 2)
        cleaned = cleaned[1] if len(cleaned) > 1 else text
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Could not parse JSON from LLM output: {text[:200]!r}")


def call_with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    description: str = "call",
) -> T:
    """Run ``fn`` with bounded exponential backoff on transient failures."""
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: BLE001 - retry policy is intentional
            last_exc = exc
            if attempt == retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                description, attempt, retries, exc, delay,
            )
            time.sleep(delay)
    raise LLMError(f"{description} failed after {retries} attempts: {last_exc}") from last_exc


class BaseLLMBackend:
    """Mixin implementing ``complete_json`` in terms of ``complete``."""

    def complete(
        self, system: str, user: str, *, json_mode: bool = False, model: str | None = None
    ) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def vision(self, image_b64: str, prompt: str, *, model: str | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_json(self, system: str, user: str, *, model: str | None = None) -> dict:
        raw = self.complete(system, user, json_mode=True, model=model)
        return extract_json(raw)
