"""Shared pieces for every algorithm: the TorchRL env wrapper, networks,
evaluation, logging and saving. All four algorithms use this same file, so the
comparison is fair.
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchrl.envs import Compose, GymWrapper, RewardSum, StepCounter, TransformedEnv
from torchrl.modules import MLP

from environment.custom_env import N_ACTIONS, OBS_SIZE, SchoolEnv

# Seeds. Training never resets with these seeds.
VAL_SEEDS = list(range(2000, 2010))        # pick the best checkpoint during training
EVAL_SEEDS = list(range(1000, 1050))       # final score of every run (50 episodes)
UNSEEN_SEEDS = list(range(50_000, 50_100))  # generalization test (100 episodes)

EVAL_EVERY = 5_000  # steps between validation checks

torch.set_num_threads(1)  # many small runs in parallel are faster than one big one


def make_env(seed: int = 0, scenario: str = "standard"):
    """The one TorchRL wrapper all four algorithms train on."""
    base = GymWrapper(SchoolEnv(scenario=scenario), device="cpu")
    env = TransformedEnv(base, Compose(StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def mlp(out_features: int, hidden: int, activation=nn.ReLU) -> nn.Module:
    return MLP(in_features=OBS_SIZE, out_features=out_features,
               num_cells=[hidden, hidden], activation_class=activation)


# ---------------------------------------------------------------------------
# Evaluation: greedy actions on the raw Gymnasium env (fast and simple).
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_action(net: nn.Module, obs: np.ndarray) -> int:
    out = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(out.argmax(dim=-1).item())


@torch.no_grad()
def policy_action(net: nn.Module, obs: np.ndarray, stochastic: bool,
                  gen: torch.Generator | None = None) -> int:
    """DQN acts greedily on Q-values. Policy-gradient agents learn a probability
    over actions, so they act by sampling from it (that IS their policy)."""
    if not stochastic:
        return greedy_action(net, obs)
    logits = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen).item())


def action_probs(net: nn.Module, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return torch.softmax(logits, dim=-1).squeeze(0).numpy()


@torch.no_grad()
def evaluate_net(net: nn.Module, seeds, scenario: str = "standard",
                 stochastic: bool = False) -> list[dict]:
    """Play one episode per seed. Return one row per seed. Sampling uses a
    generator seeded with the episode seed, so results are reproducible."""
    env = SchoolEnv(scenario=scenario)
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        gen = torch.Generator().manual_seed(seed)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy_action(net, obs, stochastic, gen))
            total += r
            done = term or trunc
        rows.append({
            "seed": seed,
            "reward": round(total, 4),
            "terms": env.term,
            "dropouts": info["dropouts"],
            "finished_strong": info["finished_strong"],
            "teacher_quits": info["teacher_quits"],
            "invalid_actions": info["invalid_actions"],
            "end_reason": info["end_reason"],
            "actions": json.dumps(env.action_counts),
        })
    return rows


def steps_to_threshold(steps, values, frac: float = 0.8, smooth: int = 3):
    """First step where the curve reaches frac of the way from its start to its
    best value AND stays there for the rest of training ("reach and hold").
    The validation curve (10 seeds) is noisy, so it is first smoothed with a
    rolling mean over `smooth` checks."""
    if len(values) == 0:
        return None
    v = pd.Series(np.asarray(values, dtype=float)).rolling(smooth, min_periods=1).mean().to_numpy()
    start, best = v[0], v.max()
    if best <= start:
        return int(steps[0])
    thr = start + frac * (best - start)
    for i in range(len(v)):
        if np.all(v[i:] >= thr - 1e-9):
            return int(steps[i])
    return int(steps[-1])


# ---------------------------------------------------------------------------
# Logging and saving
# ---------------------------------------------------------------------------

class RunLogger:
    """Writes the CSVs every run needs for the tables and figures."""

    def __init__(self, algo: str, run_id: str, config: dict, root: str = "logs"):
        self.dir = os.path.join(root, algo, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.algo, self.run_id, self.config = algo, run_id, config
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        self._files = {}
        self.start = time.time()

    def write(self, name: str, row: dict):
        path = os.path.join(self.dir, f"{name}.csv")
        new = name not in self._files
        if new:
            f = open(path, "w", newline="")
            w = csv.DictWriter(f, fieldnames=["run_id", *row.keys()])
            w.writeheader()
            self._files[name] = (f, w)
        f, w = self._files[name]
        w.writerow({"run_id": self.run_id, **row})
        f.flush()

    def write_rows(self, name: str, rows: list[dict]):
        for r in rows:
            self.write(name, r)

    def close(self):
        for f, _ in self._files.values():
            f.close()

    def elapsed(self) -> float:
        return time.time() - self.start


def save_policy(path: str, net: nn.Module, algo: str, config: dict, score: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "algo": algo,
        "state_dict": net.state_dict(),
        "hidden": config["hidden"],
        "activation": "relu" if algo == "dqn" else "tanh",
        "stochastic": algo != "dqn",
        "config": config,
        "score": score,
    }, path)


def load_policy(path: str) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    act = nn.ReLU if ckpt["activation"] == "relu" else nn.Tanh
    net = mlp(N_ACTIONS, ckpt["hidden"], act)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.stochastic = ckpt.get("stochastic", ckpt["algo"] != "dqn")
    return net


def steps_to_80_for_run(run_dir: str):
    """Recompute steps-to-80% from a run's logged validation curve."""
    v = pd.read_csv(os.path.join(run_dir, "validation.csv"))
    return steps_to_threshold(v["step"].tolist(), v["mean_reward"].tolist())


def summarise_eval(rows: list[dict]) -> dict:
    r = np.array([x["reward"] for x in rows])
    fs = np.array([x["finished_strong"] for x in rows]) / 6
    return {
        "mean_reward": float(r.mean()),
        "std_reward": float(r.std()),
        "finished_strong_share": float(fs.mean()),
        "mean_dropouts": float(np.mean([x["dropouts"] for x in rows])),
        "full_episodes": float(np.mean([x["terms"] == 18 for x in rows])),
    }


class Validator:
    """Checks the greedy policy on the validation seeds every EVAL_EVERY steps,
    keeps the best weights, and logs the curve (used for convergence)."""

    def __init__(self, logger: RunLogger, net: nn.Module, stochastic: bool = False):
        self.logger, self.net, self.stochastic = logger, net, stochastic
        self.best = -float("inf")
        self.best_state = None
        self.next_check = 0
        self.steps, self.values = [], []
        self.last_step = 0

    def maybe_check(self, step: int, force: bool = False):
        self.last_step = step
        if step < self.next_check and not force:
            return
        if self.steps and self.steps[-1] == step:
            return
        self.next_check = step + EVAL_EVERY
        s = summarise_eval(evaluate_net(self.net, VAL_SEEDS, stochastic=self.stochastic))
        self.steps.append(step)
        self.values.append(s["mean_reward"])
        self.logger.write("validation", {"step": step, **{k: round(v, 4) for k, v in s.items()}})
        if s["mean_reward"] > self.best:
            self.best = s["mean_reward"]
            self.best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

    def finish(self, algo: str, config: dict) -> dict:
        """Load the best weights, score them on the 50 eval seeds, save the model."""
        self.maybe_check(self.last_step, force=True)
        self.net.load_state_dict(self.best_state)
        rows = evaluate_net(self.net, EVAL_SEEDS, stochastic=self.stochastic)
        self.logger.write_rows("eval", rows)
        score = summarise_eval(rows)
        score["eval_mode"] = "sample" if self.stochastic else "greedy"
        if self.stochastic:  # also report the greedy score, for transparency
            score["greedy_mean_reward"] = summarise_eval(evaluate_net(self.net, EVAL_SEEDS))["mean_reward"]
        score["steps_to_80"] = steps_to_threshold(self.steps, self.values)
        score["train_time_s"] = round(self.logger.elapsed(), 1)
        score["best_val_reward"] = self.best
        with open(os.path.join(self.logger.dir, "summary.json"), "w") as f:
            json.dump({**config, **score}, f, indent=2)
        save_policy(os.path.join(self.logger.dir, "policy.pt"), self.net, algo, config, score)
        self.logger.close()
        return score
