"""Interaction + feedback log (JSON-lines). No database required."""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

from contextual_rl_agentic_rag.config import settings as default_settings

_lock = Lock()


def _log_path() -> Path:
    return default_settings.base_dir / "reward_log.jsonl"


def log_interaction(data: dict) -> None:
    """Append an interaction/feedback record to the log."""
    data = {**data, "timestamp": time.time()}
    path = _log_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")


def load_all_interactions() -> list[dict]:
    path = _log_path()
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_interactions_for_agent(agent_name: str) -> list[dict]:
    return [i for i in load_all_interactions() if i.get("agent") == agent_name]
