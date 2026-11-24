"""Policy-gradient methods with TorchRL: REINFORCE, PPO and A2C.

- REINFORCE: ReinforceLoss with no learned baseline. The advantage is the Monte Carlo
  return of each full episode (optionally normalised), plus an entropy bonus.
- PPO: ClipPPOLoss with GAE, several epochs of minibatch updates per rollout.
- A2C: A2CLoss with GAE, one update per short rollout.

Run one config:
  uv run python -m training.pg_training --algo ppo --run-id test --steps 20000
"""

from __future__ import annotations

import argparse
import json

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from environment.custom_env import N_ACTIONS
from training.common import RunLogger, Validator, make_env, mlp

BASELINES = {
    "reinforce": {
        "lr": 1e-3, "gamma": 0.99, "entropy_coef": 0.01, "episodes_per_update": 8,
        "normalize_advantage": True, "hidden": 128,
    },
    "ppo": {
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "entropy_coef": 0.01,
        "epochs": 4, "rollout": 1024, "minibatch": 64, "value_coef": 0.5, "hidden": 128,
    },
    "a2c": {
        "lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.01,
        "value_coef": 0.5, "rollout": 128, "hidden": 128,
    },
}
COMMON = {"total_steps": 100_000, "seed": 0, "max_grad_norm": 0.5}


def build_actor_critic(env, hidden: int):
    actor_net = mlp(N_ACTIONS, hidden, nn.Tanh)
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], out_keys=["action"], spec=env.action_spec,
        distribution_class=OneHotCategorical, return_log_prob=True,
    )
    critic = ValueOperator(mlp(1, hidden, nn.Tanh), in_keys=["observation"])
    return actor_net, actor, critic


def _log_episodes(logger, batch, steps, counter):
    done = batch["next", "done"].squeeze(-1)
    for ep_r in batch["next", "episode_reward"][done].flatten().tolist():
        counter[0] += 1
        logger.write("episodes", {"episode": counter[0], "step": steps, "episode_reward": round(ep_r, 4)})


def _step(optim, params, loss, max_norm):
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm)
    optim.step()


# ---------------------------------------------------------------------------
# PPO and A2C (rollout-based, GAE advantages)
# ---------------------------------------------------------------------------

def _train_actor_critic(algo: str, cfg: dict, run_id: str, verbose: bool) -> dict:
    env = make_env(cfg["seed"])
    actor_net, actor, critic = build_actor_critic(env, cfg["hidden"])
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
    if algo == "ppo":
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"], entropy_coeff=cfg["entropy_coef"],
                              critic_coeff=cfg["value_coef"], normalize_advantage=True)
    else:
        loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["value_coef"])
    params = list(loss_fn.parameters())
    optim = torch.optim.Adam(params, lr=cfg["lr"])
    collector = Collector(env, actor, frames_per_batch=cfg["rollout"], total_frames=cfg["total_steps"])

    logger = RunLogger(algo, run_id, cfg)
    validator = Validator(logger, actor_net, stochastic=True)
    steps, counter = 0, [0]
    for batch in collector:
        n = batch.numel()
        steps += n
        _log_episodes(logger, batch, steps, counter)
        logs = {"loss_objective": [], "loss_critic": [], "entropy": []}
        epochs = cfg.get("epochs", 1)
        for _ in range(epochs):
            with torch.no_grad():
                gae(batch)  # recompute advantages with the latest critic
            flat = batch.reshape(-1)
            if algo == "ppo":
                parts = torch.randperm(n).split(cfg["minibatch"])
            else:
                parts = [torch.arange(n)]
            for idx in parts:
                out = loss_fn(flat[idx])
                loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                _step(optim, params, loss, cfg["max_grad_norm"])
                logs["loss_objective"].append(out["loss_objective"].item())
                logs["loss_critic"].append(out["loss_critic"].item())
                logs["entropy"].append(out["entropy"].mean().item())
        logger.write("train", {
            "step": steps,
            "loss": round(sum(logs["loss_objective"]) / len(logs["loss_objective"]), 6),
            "value_loss": round(sum(logs["loss_critic"]) / len(logs["loss_critic"]), 6),
            "entropy": round(sum(logs["entropy"]) / len(logs["entropy"]), 6),
            "lr": cfg["lr"],
        })
        validator.maybe_check(steps)
        if verbose and steps % 10_000 < n:
            print(f"[{algo} {run_id}] step {steps:>7}  episodes {counter[0]:>5}  "
                  f"entropy {logs['entropy'][-1]:.3f}  val reward {validator.values[-1]:+.2f}", flush=True)
    collector.shutdown()
    return validator.finish(algo, cfg)


# ---------------------------------------------------------------------------
# REINFORCE (full episodes, Monte Carlo returns)
# ---------------------------------------------------------------------------

def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    out = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def _train_reinforce(cfg: dict, run_id: str, verbose: bool) -> dict:
    env = make_env(cfg["seed"])
    actor_net, actor, critic = build_actor_critic(env, cfg["hidden"])
    # TorchRL 0.15 requires a critic argument. It is NOT trained (not in the
    # optimizer) and NOT used for the advantage, which we set ourselves to the
    # Monte Carlo return. Its value loss is ignored. This keeps REINFORCE pure.
    loss_fn = ReinforceLoss(actor, critic_network=critic)
    params = list(actor_net.parameters())
    optim = torch.optim.Adam(params, lr=cfg["lr"])

    logger = RunLogger("reinforce", run_id, cfg)
    validator = Validator(logger, actor_net, stochastic=True)
    steps, episodes = 0, 0
    while steps < cfg["total_steps"]:
        trajs = []
        with set_exploration_type(ExplorationType.RANDOM):
            for _ in range(cfg["episodes_per_update"]):
                td = env.rollout(max_steps=18, policy=actor, break_when_any_done=True)
                returns = _discounted_returns(td["next", "reward"], cfg["gamma"])
                td["advantage"] = returns
                td["value_target"] = returns.clone()
                trajs.append(td)
                episodes += 1
                steps += td.numel()
                logger.write("episodes", {"episode": episodes, "step": steps,
                                          "episode_reward": round(td["next", "reward"].sum().item(), 4)})
        batch = torch.cat(trajs, 0)
        adv = batch["advantage"]
        if cfg["normalize_advantage"]:
            batch["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        out = loss_fn(batch)
        entropy = actor.get_dist(batch).entropy().mean()
        loss = out["loss_actor"] - cfg["entropy_coef"] * entropy
        _step(optim, params, loss, cfg["max_grad_norm"])
        logger.write("train", {"step": steps, "loss": round(out["loss_actor"].item(), 6),
                               "entropy": round(entropy.item(), 6), "lr": cfg["lr"]})
        validator.maybe_check(steps)
        if verbose and steps % 10_000 < cfg["episodes_per_update"] * 18:
            print(f"[reinforce {run_id}] step {steps:>7}  episodes {episodes:>5}  "
                  f"entropy {entropy.item():.3f}  val reward {validator.values[-1]:+.2f}", flush=True)
    return validator.finish("reinforce", cfg)


def train_pg(algo: str, config: dict, run_id: str, verbose: bool = True) -> dict:
    cfg = {**COMMON, **BASELINES[algo], **config, "algo": algo}
    torch.manual_seed(cfg["seed"])
    score = _train_reinforce(cfg, run_id, verbose) if algo == "reinforce" \
        else _train_actor_critic(algo, cfg, run_id, verbose)
    if verbose:
        print(f"[{algo} {run_id}] done: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in score.items()})}")
    return score


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(BASELINES), required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--steps", type=int, default=COMMON["total_steps"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="{}", help="JSON overrides, e.g. '{\"clip\": 0.1}'")
    a = p.parse_args()
    train_pg(a.algo, {**json.loads(a.config), "total_steps": a.steps, "seed": a.seed},
             a.run_id or f"{a.algo}_manual")
