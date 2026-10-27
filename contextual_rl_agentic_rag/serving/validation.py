"""Framework-agnostic upload validation (no FastAPI import, so it is unit-testable)."""
from __future__ import annotations

import os
import re

EXT_TYPE_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".csv": "csv",
    ".xlsx": "csv",
    ".xls": "csv",
}


class UploadValidationError(ValueError):
    """Validation failure with an associated HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def sanitize_filename(name: str | None) -> str:
    """Strip any directory components and unsafe characters (prevents path traversal)."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = base.split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return cleaned or "upload"


def validate_extension(filename: str, allowed: tuple[str, ...]) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise UploadValidationError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(allowed))}",
            status_code=415,
        )
    return ext


def resolve_file_type(ext: str) -> str:
    return EXT_TYPE_MAP.get(ext.lower(), "pdf")


def check_size(num_bytes: int, max_mb: int) -> None:
    if num_bytes > max_mb * 1024 * 1024:
        raise UploadValidationError(f"File exceeds the {max_mb} MB limit", status_code=413)
