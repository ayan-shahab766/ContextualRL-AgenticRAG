"""Application configuration.

Settings are read from environment variables (optionally seeded from a ``.env``
file). This module has **no import-time side effects** beyond building the
settings object: it does not create directories, open network clients, or read
secrets eagerly. Call :meth:`Settings.ensure_dirs` explicitly when you actually
need the data directories to exist.

We intentionally avoid a hard dependency on ``pydantic-settings`` so the config
(and everything that imports it) stays importable in minimal/offline
environments used for testing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _load_dotenv(path: str = ".env") -> None:
    """Best-effort ``.env`` loader. Only sets keys that are not already set."""
    p = Path(path)
    if not p.exists():
        return
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)
    except OSError:
        # A malformed/unreadable .env should never crash startup.
        return


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    # --- Secrets / backends ---
    openai_api_key: str = ""
    # Backend selectors: "openai" | "local" | "fake"
    llm_backend: str = "openai"
    embedding_backend: str = ""  # falls back to llm_backend when empty
    # Vector store: "chroma" (persistent) | "memory" (in-process, no native deps)
    vector_store: str = "chroma"

    # --- Directories ---
    base_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    chroma_dir: Path = Path("./data/chroma")
    lora_adapters_path: Optional[Path] = None

    # --- Models ---
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vision_model: str = "gpt-4o-mini"

    # --- Network reliability ---
    request_timeout: float = 30.0  # seconds per LLM/embedding call
    max_retries: int = 3

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    sentence_window_size: int = 3

    # --- Retrieval ---
    default_top_k: int = 5
    relevance_threshold: float = 0.7
    max_agentic_retries: int = 3

    # --- Serving / security ---
    api_key: str = ""  # if set, required via X-API-Key header; empty => auth disabled
    cors_allow_origins: str = ""  # comma-separated; empty => same-origin only
    max_upload_mb: int = 25
    allowed_upload_extensions: tuple[str, ...] = (
        ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".csv", ".xlsx", ".xls",
    )
    log_level: str = "INFO"

    @property
    def effective_embedding_backend(self) -> str:
        return self.embedding_backend or self.llm_backend

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv()
        lora = os.getenv("LORA_ADAPTERS_PATH")
        exts = os.getenv("ALLOWED_UPLOAD_EXTENSIONS")
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_backend=os.getenv("LLM_BACKEND", "openai"),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", ""),
            vector_store=os.getenv("VECTOR_STORE", "chroma"),
            base_dir=Path(os.getenv("BASE_DIR", "./data")),
            upload_dir=Path(os.getenv("UPLOAD_DIR", "./data/uploads")),
            chroma_dir=Path(os.getenv("CHROMA_DIR", "./data/chroma")),
            lora_adapters_path=Path(lora) if lora else None,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            vision_model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
            request_timeout=_get_float("REQUEST_TIMEOUT", 30.0),
            max_retries=_get_int("MAX_RETRIES", 3),
            chunk_size=_get_int("CHUNK_SIZE", 512),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 50),
            sentence_window_size=_get_int("SENTENCE_WINDOW_SIZE", 3),
            default_top_k=_get_int("DEFAULT_TOP_K", 5),
            relevance_threshold=_get_float("RELEVANCE_THRESHOLD", 0.7),
            max_agentic_retries=_get_int("MAX_AGENTIC_RETRIES", 3),
            api_key=os.getenv("API_KEY", ""),
            cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", ""),
            max_upload_mb=_get_int("MAX_UPLOAD_MB", 25),
            allowed_upload_extensions=(
                tuple(e.strip().lower() for e in exts.split(",") if e.strip())
                if exts
                else cls.allowed_upload_extensions
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def ensure_dirs(self) -> None:
        for d in (self.base_dir, self.upload_dir, self.chroma_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings.load()
