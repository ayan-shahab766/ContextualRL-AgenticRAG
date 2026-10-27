from contextual_rl_agentic_rag.rl.bandit_router import AGENTS, CATEGORIES, BanditRouter


def test_initializes_with_priors(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    # analytical/table_rag prior is (3 pulls, 2.4 reward) => 0.8 avg.
    stats = b.get_stats()
    assert stats["analytical"]["table_rag"]["avg_reward"] == 0.8
    assert (tmp_path / "bandit_state.json").exists()


def test_select_agent_returns_valid_agent(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    agent, info = b.select_agent("visual")
    assert agent in AGENTS
    assert info["category"] == "visual"
    assert set(info["all_scores"].keys()) == set(AGENTS)


def test_unknown_category_falls_back(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    agent, info = b.select_agent("does_not_exist")
    assert info["category"] == "factual"
    assert agent in AGENTS


def test_update_changes_average_and_persists(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    before = b.state["factual"]["naive_rag"]["pulls"]
    b.update("factual", "naive_rag", 1.0)
    assert b.state["factual"]["naive_rag"]["pulls"] == before + 1

    # New instance should load the persisted update from disk.
    b2 = BanditRouter(base_dir=tmp_path)
    assert b2.state["factual"]["naive_rag"]["pulls"] == before + 1


def test_update_ignores_unknown_keys(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    b.update("nope", "naive_rag", 1.0)  # should not raise
    b.update("factual", "nope", 1.0)


def test_high_reward_raises_agent_average(tmp_path):
    # UCB1 keeps exploring under-pulled arms, so select_agent may still explore;
    # what learning guarantees is that the rewarded arm's *average* becomes best.
    b = BanditRouter(base_dir=tmp_path)
    for _ in range(30):
        b.update("factual", "hyde_rag", 1.0)
    stats = b.get_stats()["factual"]
    best = max(stats, key=lambda a: stats[a]["avg_reward"])
    assert best == "hyde_rag"
    assert stats["hyde_rag"]["avg_reward"] > 0.9


def test_reset_restores_priors(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    b.update("factual", "naive_rag", 1.0)
    b.reset()
    # After reset, factual/naive_rag prior is (2 pulls, 1.4 reward).
    assert b.state["factual"]["naive_rag"]["pulls"] == 2


def test_leaderboard_ranks_all_agents(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    lb = b.get_leaderboard()
    assert len(lb) == len(AGENTS)
    scores = [row["avg_reward"] for row in lb]
    assert scores == sorted(scores, reverse=True)


def test_categories_and_agents_populated(tmp_path):
    b = BanditRouter(base_dir=tmp_path)
    assert set(b.state.keys()) == set(CATEGORIES)
