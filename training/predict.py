"""Integration path: JSON state in, action name out.

A web dashboard or a Flutter app can send the school state as JSON (the same
format as SchoolEnv.get_state_json()) and get the director's next action back.

    from training.predict import predict
    predict(env.get_state_json())            # -> "Mentor Larrissa"
    predict(state_json, algo="dqn")          # use a specific trained agent

Command line:  uv run python -m training.predict logs/episodes/<file>.json
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

from environment.custom_env import ACTION_NAMES, obs_from_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATHS = {
    "dqn": os.path.join(ROOT, "models", "dqn", "best_dqn.pt"),
    "reinforce": os.path.join(ROOT, "models", "pg", "best_reinforce.pt"),
    "ppo": os.path.join(ROOT, "models", "pg", "best_ppo.pt"),
    "a2c": os.path.join(ROOT, "models", "pg", "best_a2c.pt"),
}
BEST_AGENT_FILE = os.path.join(ROOT, "models", "best_agent.json")


def best_algo() -> str:
    """The winner named by the leaderboard rule (written by training/plots.py)."""
    if os.path.exists(BEST_AGENT_FILE):
        with open(BEST_AGENT_FILE) as f:
            return json.load(f)["algo"]
    for algo in ["ppo", "dqn", "a2c", "reinforce"]:
        if os.path.exists(MODEL_PATHS[algo]):
            return algo
    raise FileNotFoundError("No trained model found in models/. Train first (see README).")


@lru_cache(maxsize=None)
def load_agent(algo: str):
    from training.common import load_policy
    path = MODEL_PATHS[algo]
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model for {algo} at {path}")
    return load_policy(path)


def predict_index(state, algo: str | None = None, deterministic: bool = False) -> int:
    from training.common import policy_action
    net = load_agent(algo or best_algo())
    return policy_action(net, obs_from_state(state), net.stochastic and not deterministic)


def predict(state_json, algo: str | None = None, deterministic: bool = False) -> str:
    """Return the name of the action the trained director would take.
    Policy-gradient agents sample from their policy unless deterministic=True."""
    return ACTION_NAMES[predict_index(state_json, algo, deterministic)]


def predict_proba(state_json, algo: str | None = None) -> dict:
    """Action probabilities (policy-gradient agents) - useful for a dashboard."""
    from training.common import action_probs
    net = load_agent(algo or best_algo())
    p = action_probs(net, obs_from_state(state_json))
    return {ACTION_NAMES[i]: round(float(x), 4) for i, x in enumerate(p)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        from environment.custom_env import SchoolEnv
        env = SchoolEnv()
        env.reset(seed=0)
        state = env.get_state_json()
        print("Example state (term 0):", state[:160], "...")
        print("Predicted action:", predict(state))
    else:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        state = data["terms"][0]["state_before"] if "terms" in data else data
        print("Predicted action:", predict(state))
