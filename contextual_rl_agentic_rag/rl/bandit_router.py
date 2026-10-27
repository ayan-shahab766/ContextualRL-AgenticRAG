"""UCB1 routing with a global policy and contextual policy overlays."""
from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock

from contextual_rl_agentic_rag.config import settings as default_settings
from contextual_rl_agentic_rag.utils import atomic_write_text

# Keep in sync with contextual_rl_agentic_rag.agents.AGENT_NAMES.
AGENTS: list[str] = [
    "naive_rag",
    "sentence_window_rag",
    "parent_child_rag",
    "agentic_rag",
    "multimodal_rag",
    "table_rag",
    "graph_rag",
    "hybrid_rag",
    "hyde_rag",
    "corrective_rag",
    "lora_rag",
]

CATEGORIES: list[str] = ["factual", "analytical", "relational", "procedural", "vague", "visual"]

# Smart priors: (prior_pulls, prior_total_reward) => a warm-start average reward.
PRIOR_REWARDS: dict[tuple[str, str], tuple[int, float]] = {
    ("analytical", "table_rag"): (3, 2.4),
    ("relational", "graph_rag"): (3, 2.4),
    ("visual", "multimodal_rag"): (3, 2.4),
    ("vague", "hyde_rag"): (3, 2.1),
    ("factual", "naive_rag"): (2, 1.4),
    ("factual", "hybrid_rag"): (2, 1.4),
    ("procedural", "agentic_rag"): (3, 2.4),
    ("procedural", "parent_child_rag"): (2, 1.4),
}


class BanditRouter:
    """UCB1 multi-armed bandit for agent selection."""

    def __init__(self, exploration_constant: float = 1.41, *, base_dir=None) -> None:
        self.C = exploration_constant  # ~sqrt(2), theoretically optimal
        self.lock = Lock()
        base_dir = Path(base_dir) if base_dir is not None else default_settings.base_dir
        self.state_path = Path(base_dir) / "bandit_state.json"
        self.context_state_path = Path(base_dir) / "contextual_bandit_state.json"
        self.state: dict = {}
        self.context_state: dict = {}
        self._load()
        self._load_context_state()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
                return
            except (OSError, json.JSONDecodeError):
                pass
        self._initialize()

    def _save(self) -> None:
        atomic_write_text(self.state_path, json.dumps(self.state, indent=2))

    def _load_context_state(self) -> None:
        if not self.context_state_path.exists():
            return
        try:
            self.context_state = json.loads(self.context_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.context_state = {}

    def _save_context_state(self) -> None:
        atomic_write_text(self.context_state_path, json.dumps(self.context_state, indent=2))

    def _initialize(self) -> None:
        self.state = {}
        for cat in CATEGORIES:
            self.state[cat] = {}
            for agent in AGENTS:
                prior = PRIOR_REWARDS.get((cat, agent), (1, 0.5))
                self.state[cat][agent] = {"pulls": prior[0], "total_reward": prior[1]}
        self._save()

    def _ensure_context(self, category: str, context_key: str) -> dict:
        """Warm-start a context from category priors, keeping sparse contexts stable."""
        if context_key not in self.context_state:
            self.context_state[context_key] = {
                agent: dict(data) for agent, data in self.state[category].items()
            }
            self._save_context_state()
        return self.context_state[context_key]

    def _score(self, arm_state: dict) -> tuple[str, dict, int]:
        total_pulls = sum(a["pulls"] for a in arm_state.values())
        scores: dict[str, dict] = {}
        for agent, data in arm_state.items():
            n = data["pulls"]
            avg_reward = data["total_reward"] / n if n > 0 else 0
            exploration_bonus = self.C * math.sqrt(math.log(total_pulls + 1) / (n + 1))
            scores[agent] = {
                "ucb_score": round(avg_reward + exploration_bonus, 4),
                "avg_reward": round(avg_reward, 4),
                "exploration_bonus": round(exploration_bonus, 4),
                "pulls": n,
                "total_reward": round(data["total_reward"], 2),
            }
        return max(scores, key=lambda a: scores[a]["ucb_score"]), scores, total_pulls

    def select_agent(self, category: str, *, context_key: str | None = None) -> tuple[str, dict]:
        if category not in self.state:
            category = "factual"
        use_context = bool(context_key and context_key != category)
        arm_state = self._ensure_context(category, context_key) if use_context else self.state[category]
        best_agent, scores, total_pulls = self._score(arm_state)
        return best_agent, {
            "category": category,
            "context_key": context_key or category,
            "policy": "contextual_ucb1" if use_context else "ucb1",
            "selected": best_agent,
            "all_scores": scores,
            "total_interactions": total_pulls,
        }

    def update(self, category: str, agent: str, reward: float, *, context_key: str | None = None) -> None:
        with self.lock:
            if category not in self.state or agent not in self.state[category]:
                return
            self.state[category][agent]["pulls"] += 1
            self.state[category][agent]["total_reward"] += reward
            self._save()
            if context_key and context_key != category:
                contextual_arms = self._ensure_context(category, context_key)
                contextual_arms[agent]["pulls"] += 1
                contextual_arms[agent]["total_reward"] += reward
                self._save_context_state()

    def get_stats(self) -> dict:
        stats: dict = {}
        for cat in CATEGORIES:
            cat_stats = {}
            for agent in AGENTS:
                data = self.state.get(cat, {}).get(agent, {"pulls": 0, "total_reward": 0})
                n = data["pulls"]
                cat_stats[agent] = {
                    "pulls": n,
                    "avg_reward": round(data["total_reward"] / n, 3) if n > 0 else 0,
                    "total_reward": round(data["total_reward"], 2),
                }
            cat_stats = dict(
                sorted(cat_stats.items(), key=lambda x: x[1]["avg_reward"], reverse=True)
            )
            stats[cat] = cat_stats
        return stats

    def get_leaderboard(self) -> list[dict]:
        agent_totals = {a: {"pulls": 0, "reward": 0.0, "categories_won": 0} for a in AGENTS}
        for cat in CATEGORIES:
            best_avg = 0.0
            best_agent = None
            for agent in AGENTS:
                data = self.state.get(cat, {}).get(agent, {"pulls": 0, "total_reward": 0})
                agent_totals[agent]["pulls"] += data["pulls"]
                agent_totals[agent]["reward"] += data["total_reward"]
                avg = data["total_reward"] / data["pulls"] if data["pulls"] > 0 else 0
                if avg > best_avg:
                    best_avg = avg
                    best_agent = agent
            if best_agent:
                agent_totals[best_agent]["categories_won"] += 1

        leaderboard = []
        for agent, totals in agent_totals.items():
            n = totals["pulls"]
            leaderboard.append(
                {
                    "agent": agent,
                    "total_interactions": n,
                    "avg_reward": round(totals["reward"] / n, 3) if n > 0 else 0,
                    "categories_won": totals["categories_won"],
                }
            )
        leaderboard.sort(key=lambda x: x["avg_reward"], reverse=True)
        return leaderboard

    def get_contextual_stats(self) -> dict:
        """Return policy stats per observed context without exposing internal state shape."""
        return {
            context: {
                agent: {
                    "pulls": data["pulls"],
                    "avg_reward": round(data["total_reward"] / data["pulls"], 3) if data["pulls"] else 0,
                }
                for agent, data in arms.items()
            }
            for context, arms in self.context_state.items()
        }

    def reset(self) -> None:
        self._initialize()
        self.context_state = {}
        self._save_context_state()
