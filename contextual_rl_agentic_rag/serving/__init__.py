"""FastAPI serving layer.

``create_app`` is imported lazily via ``__getattr__`` so that importing pure
submodules (e.g. ``contextual_rl_agentic_rag.serving.validation``) does not require FastAPI.
"""
from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from contextual_rl_agentic_rag.serving.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
