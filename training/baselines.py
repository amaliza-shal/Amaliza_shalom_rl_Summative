"""Baseline agents and a shared evaluation loop.

- RandomAgent: picks any of the 19 actions.
- SaveAgent: always saves money (checks that doing nothing slowly fails).
- RuleBasedAgent: the fixed rule from the plan.
- ScriptedPlanAgent: a hand-written balanced plan. Used only in tests to check
  that a sensible plan beats the simple rule (so the reward makes sense).

Run: uv run python -m training.baselines
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from environment.custom_env import (
    ENGAGE, HIRE, LOBBY, MENTOR_0, N_CHILDREN, REMOTE, SAVE, SUPPORT_0, TECH, TRAIN,
    SchoolEnv,
)

# Fixed seeds so every agent is scored on the same 50 worlds.
EVAL_SEEDS = list(range(1000, 1050))
# 100 seeds never used in training or sweep evaluation (generalization test).
UNSEEN_SEEDS = list(range(50_000, 50_100))


def _children(env):
    return [c for c in env.children if c["enrolled"]]


class RandomAgent:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return int(self.rng.integers(0, 19))


class SaveAgent:
    name = "save_only"

    def act(self, obs, env):
        return SAVE


class RuleBasedAgent:
    """Hire when under 4 teachers and money allows, lobby when funding is low,
    otherwise mentor the least confident child."""

    name = "rule_based"

    def act(self, obs, env):
        # "Budget allows" = a reserve of 0.25 is left after paying for the hire.
        if env.teachers < 4 and env.budget - 0.20 >= 0.25:
            return HIRE
        if env.gov_funding < 0.25:
            return LOBBY
        kids = _children(env)
        least = min(kids, key=lambda c: c["confidence"])
        return MENTOR_0 + env.children.index(least)


class ScriptedPlanAgent:
    """A balanced hand-made plan: funding first, then teachers, then targeted
    help for the child who needs it most."""

    name = "scripted_plan"

    def act(self, obs, env):
        kids = _children(env)
        idx = env.children.index
        t = env.term
        if env.budget < 0.10:
            return LOBBY
        if t in (0, 3) :
            return LOBBY
        if t == 1:
            return ENGAGE
        if t in (2, 8) and env.budget > 0.15:
            return TRAIN
        if env.teachers < 3 and env.budget > 0.40:
            return HIRE
        # Child in danger of dropping out: support the family.
        risky = [c for c in kids if c["attendance"] < 0.45]
        if risky:
            c = min(risky, key=lambda c: c["attendance"])
            if env.budget > 0.12:
                return SUPPORT_0 + idx(c)
        # Child below the confidence bar: mentor.
        low = [c for c in kids if c["confidence"] < 0.6]
        if low and t % 3 != 2:
            return MENTOR_0 + idx(min(low, key=lambda c: c["confidence"]))
        if env.trust < 0.4:
            return ENGAGE
        if env.gov_funding < 0.3:
            return LOBBY
        weakest = min(kids, key=lambda c: c["learning"])
        return MENTOR_0 + idx(weakest)


def run_episode(agent, seed, scenario="standard", env=None):
    """Play one full episode. Return a dict of results."""
    env = env or SchoolEnv(scenario=scenario)
    obs, info = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, env))
        total += r
        done = term or trunc
    return {
        "seed": seed,
        "reward": total,
        "terms": env.term,
        "dropouts": info["dropouts"],
        "finished_strong": info["finished_strong"],
        "finished_strong_share": info["finished_strong"] / N_CHILDREN,
        "teacher_quits": info["teacher_quits"],
        "invalid_actions": info["invalid_actions"],
        "end_reason": info["end_reason"],
    }


def evaluate(agent, seeds=EVAL_SEEDS, scenario="standard") -> pd.DataFrame:
    env = SchoolEnv(scenario=scenario)
    return pd.DataFrame([run_episode(agent, s, scenario, env) for s in seeds])


def summarise(df: pd.DataFrame) -> dict:
    return {
        "mean_reward": df["reward"].mean(),
        "std_reward": df["reward"].std(),
        "finished_strong_share": df["finished_strong_share"].mean(),
        "mean_dropouts": df["dropouts"].mean(),
        "full_episodes": (df["terms"] == 18).mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="write CSVs to logs/baselines")
    args = parser.parse_args()
    agents = [RandomAgent(), SaveAgent(), RuleBasedAgent(), ScriptedPlanAgent()]
    rows = []
    for scenario, seeds in [("standard", EVAL_SEEDS), ("standard_unseen", UNSEEN_SEEDS), ("hard", UNSEEN_SEEDS)]:
        scen = "hard" if scenario == "hard" else "standard"
        for agent in agents:
            df = evaluate(agent, seeds, scen)
            if args.save:
                os.makedirs("logs/baselines", exist_ok=True)
                df.to_csv(f"logs/baselines/{agent.name}_{scenario}.csv", index=False)
            rows.append({"agent": agent.name, "scenario": scenario, **summarise(df)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))
    if args.save:
        table.to_csv("logs/baselines/summary.csv", index=False)


if __name__ == "__main__":
    main()
