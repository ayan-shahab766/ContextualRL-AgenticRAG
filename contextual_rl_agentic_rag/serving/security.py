"""FastAPI security helpers: API-key auth and safe upload handling."""
from __future__ import annotations

from fastapi import Header, HTTPException, UploadFile

from contextual_rl_agentic_rag.config import Settings
from contextual_rl_agentic_rag.serving.validation import (
    UploadValidationError,
    check_size,
    sanitize_filename,
    validate_extension,
)


def make_api_key_dependency(settings: Settings):
    """Build a dependency that enforces the X-API-Key header when a key is configured.

    If ``settings.api_key`` is empty, auth is disabled (useful for local dev) - but the
    app logs a warning at startup in that case.
    """

    async def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _require_api_key


async def read_upload(file: UploadFile, settings: Settings) -> tuple[bytes, str, str]:
    """Validate and read an upload within limits.

    Returns ``(data, safe_filename, file_type)``. Raises ``HTTPException`` on any
    validation failure.
    """
    safe_name = sanitize_filename(file.filename)
    try:
        ext = validate_extension(safe_name, settings.allowed_upload_extensions)
    except UploadValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    max_bytes = settings.max_upload_mb * 1024 * 1024
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        try:
            check_size(len(buf), settings.max_upload_mb)
        except UploadValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        if len(buf) > max_bytes:  # defensive
            raise HTTPException(status_code=413, detail="File too large")

    from contextual_rl_agentic_rag.serving.validation import resolve_file_type

    return bytes(buf), safe_name, resolve_file_type(ext)
