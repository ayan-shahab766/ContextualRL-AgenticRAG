"""Central logging setup.

Provides a single ``configure_logging`` entrypoint and a ``get_logger`` helper so
every module logs through the same, consistently formatted handlers instead of
using ``print`` or staying silent.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure root logging to stdout."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
