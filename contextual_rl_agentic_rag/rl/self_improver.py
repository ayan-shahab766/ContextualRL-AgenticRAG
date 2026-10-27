"""Offline analysis of the interaction log to surface insights and parameter suggestions."""
from __future__ import annotations

from contextual_rl_agentic_rag.rl.bandit_router import BanditRouter
from contextual_rl_agentic_rag.rl.reward_store import load_all_interactions


class SelfImprover:
    def __init__(self, bandit: BanditRouter) -> None:
        self.bandit = bandit

    def analyze(self) -> dict:
        interactions = load_all_interactions()

        if len(interactions) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 interactions with feedback. Currently have {len(interactions)}.",
                "interactions_logged": len(interactions),
            }

        report: dict = {
            "total_interactions": len(interactions),
            "with_feedback": len([i for i in interactions if "reward" in i]),
            "insights": [],
            "recommendations": [],
            "parameter_suggestions": {},
        }

        # --- Agent performance by category ---
        agent_cat_perf: dict[tuple[str, str], list[float]] = {}
        for i in interactions:
            if "reward" not in i:
                continue
            key = (i.get("agent", "unknown"), i.get("category", "unknown"))
            agent_cat_perf.setdefault(key, []).append(i["reward"])

        for (agent, cat), rewards in agent_cat_perf.items():
            avg = sum(rewards) / len(rewards)
            if avg < 0.4 and len(rewards) >= 3:
                report["insights"].append(
                    {
                        "type": "low_performance",
                        "agent": agent,
                        "category": cat,
                        "avg_reward": round(avg, 3),
                        "sample_size": len(rewards),
                        "message": f"{agent} struggles with {cat} questions (avg reward: {avg:.2f})",
                    }
                )
            elif avg > 0.8 and len(rewards) >= 3:
                report["insights"].append(
                    {
                        "type": "high_performance",
                        "agent": agent,
                        "category": cat,
                        "avg_reward": round(avg, 3),
                        "sample_size": len(rewards),
                        "message": f"{agent} excels at {cat} questions (avg reward: {avg:.2f})",
                    }
                )

        # --- Top-K effectiveness ---
        topk_rewards: dict[int, list[float]] = {}
        for i in interactions:
            if "reward" not in i or "top_k" not in i:
                continue
            topk_rewards.setdefault(i["top_k"], []).append(i["reward"])
        if topk_rewards:
            best_k = max(topk_rewards, key=lambda k: sum(topk_rewards[k]) / len(topk_rewards[k]))
            report["parameter_suggestions"]["recommended_top_k"] = best_k
            report["insights"].append(
                {
                    "type": "top_k_analysis",
                    "best_k": best_k,
                    "all_k_performance": {
                        k: {"avg_reward": round(sum(v) / len(v), 3), "count": len(v)}
                        for k, v in topk_rewards.items()
                    },
                }
            )

        # --- Category difficulty ---
        cat_rewards: dict[str, list[float]] = {}
        for i in interactions:
            if "reward" not in i:
                continue
            cat_rewards.setdefault(i.get("category", "unknown"), []).append(i["reward"])

        hardest, hardest_avg = None, 1.0
        for cat, rewards in cat_rewards.items():
            avg = sum(rewards) / len(rewards)
            if avg < hardest_avg:
                hardest_avg, hardest = avg, cat
        if hardest:
            report["insights"].append(
                {
                    "type": "hardest_category",
                    "category": hardest,
                    "avg_reward": round(hardest_avg, 3),
                    "message": f"'{hardest}' questions are hardest (avg reward: {hardest_avg:.2f}). "
                    "Consider uploading more relevant documents.",
                }
            )

        # --- Query length vs reward ---
        short_q = [i for i in interactions if "reward" in i and len(i.get("question", "").split()) <= 5]
        long_q = [i for i in interactions if "reward" in i and len(i.get("question", "").split()) > 10]
        if short_q and long_q:
            short_avg = sum(i["reward"] for i in short_q) / len(short_q)
            long_avg = sum(i["reward"] for i in long_q) / len(long_q)
            report["insights"].append(
                {
                    "type": "query_length",
                    "short_queries_avg": round(short_avg, 3),
                    "long_queries_avg": round(long_avg, 3),
                    "message": f"Short queries avg reward: {short_avg:.2f}, Long queries: {long_avg:.2f}",
                }
            )

        # --- Recommendations from bandit stats ---
        stats = self.bandit.get_stats()
        for cat, agents in stats.items():
            best = max(agents, key=lambda a: agents[a]["avg_reward"])
            worst = min(agents, key=lambda a: agents[a]["avg_reward"])
            if agents[best]["avg_reward"] - agents[worst]["avg_reward"] > 0.3:
                report["recommendations"].append(
                    f"For '{cat}' questions, strongly prefer {best} "
                    f"(avg: {agents[best]['avg_reward']}) over {worst} "
                    f"(avg: {agents[worst]['avg_reward']})"
                )

        return report
