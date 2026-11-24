"""Environment checks. Run: uv run pytest -q"""

import json

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (
    HIRE, MENTOR_0, SAVE, SchoolEnv, make_config, obs_from_state,
)
from training.baselines import (
    EVAL_SEEDS, RandomAgent, RuleBasedAgent, ScriptedPlanAgent, evaluate,
)


def test_gymnasium_check_env():
    check_env(SchoolEnv(), skip_render_check=True)


def test_spaces():
    env = SchoolEnv()
    assert env.observation_space.shape == (32,)
    assert env.action_space.n == 19


def test_same_seed_same_trajectory():
    def rollout():
        env = SchoolEnv()
        obs, _ = env.reset(seed=123)
        rng = np.random.default_rng(7)
        out = [obs]
        for _ in range(18):
            obs, r, term, trunc, _ = env.step(int(rng.integers(19)))
            out.append(np.append(obs, r))
            if term:
                break
        return out

    a, b = rollout(), rollout()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_observations_in_range_and_no_nan():
    env = SchoolEnv()
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            assert obs.dtype == np.float32
            assert not np.isnan(obs).any()
            assert obs.min() >= 0.0 and obs.max() <= 1.0
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc


def test_terminal_term_18():
    env = SchoolEnv()
    env.reset(seed=1)
    env.cfg["p_quit_normal"] = 0.0
    env.cfg["p_quit_low_morale"] = 0.0
    done, info = False, {}
    while not done:
        _, _, done, _, info = env.step(MENTOR_0)
    assert info["end_reason"] == "Term 18 reached" and env.term == 18


def test_terminal_budget_collapse():
    env = SchoolEnv()
    env.reset(seed=1)
    done, info = False, {}
    while not done:
        _, r, done, _, info = env.step(HIRE)
    assert info["end_reason"].startswith("Budget collapse")
    assert r <= -10 + 2  # collapse penalty is applied


def test_terminal_all_teachers_quit():
    cfg = make_config()
    cfg["p_quit_normal"] = 1.0
    cfg["p_quit_low_morale"] = 1.0
    env = SchoolEnv(config=cfg)
    env.reset(seed=1)
    done, info = False, {}
    while not done:
        _, _, done, _, info = env.step(SAVE)
    assert info["end_reason"] == "All teachers quit"


def test_terminal_four_dropouts():
    env = SchoolEnv()
    env.reset(seed=1)
    for c in env.children[:4]:
        c.update(attendance=0.05, confidence=0.05, poverty=1.0, distance=1.0)
    done, info = False, {}
    while not done:
        _, _, done, _, info = env.step(SAVE)
    assert info["end_reason"] == "4 or more children dropped out"
    assert info["dropouts"] >= 4


def test_state_json_round_trip():
    env = SchoolEnv()
    obs, _ = env.reset(seed=5)
    env.step(MENTOR_0 + 2)
    state = json.loads(env.get_state_json())
    assert len(state["children"]) == 6
    np.testing.assert_allclose(obs_from_state(state), env._obs(), atol=1e-4)


def test_hard_world_is_harder():
    rule = RuleBasedAgent()
    std = evaluate(rule, EVAL_SEEDS[:20], "standard")["reward"].mean()
    hard = evaluate(rule, EVAL_SEEDS[:20], "hard")["reward"].mean()
    assert hard < std


def test_reward_ordering_plan_beats_rule_beats_random():
    """A sensible plan must score more than the fixed rule, and the fixed rule
    more than random play. If not, the reward is broken."""
    plan = evaluate(ScriptedPlanAgent())["reward"].mean()
    rule = evaluate(RuleBasedAgent())["reward"].mean()
    rand = evaluate(RandomAgent(seed=0))["reward"].mean()
    assert plan > rule > rand
