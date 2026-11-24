"""DQN with TorchRL: DQNLoss, EGreedyModule, a replay buffer and a soft target
network update.

Run one config:  uv run python -m training.dqn_training --run-id test --steps 20000
"""

from __future__ import annotations

import argparse
import json

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from environment.custom_env import N_ACTIONS
from training.common import RunLogger, Validator, make_env, mlp

DQN_BASELINE = {
    "lr": 1e-3,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 50_000,
    "eps_decay_steps": 30_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "tau": 0.005,
    "hidden": 128,
    "frames_per_batch": 16,   # collect 16 steps ...
    "updates_per_batch": 8,   # ... then do 8 gradient updates
    "warmup_steps": 1_000,
    "total_steps": 100_000,
    "seed": 0,
}


def train_dqn(config: dict, run_id: str, verbose: bool = True) -> dict:
    cfg = {**DQN_BASELINE, **config}
    torch.manual_seed(cfg["seed"])
    env = make_env(cfg["seed"])

    q_net = mlp(N_ACTIONS, cfg["hidden"], torch.nn.ReLU)
    actor = QValueActor(q_net, in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                            annealing_num_steps=cfg["eps_decay_steps"])
    policy = TensorDictSequential(actor, egreedy)

    collector = Collector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_steps"], init_random_frames=cfg["warmup_steps"])
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]),
                                    batch_size=cfg["batch_size"])

    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True)
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    target_updater = SoftUpdate(loss_fn, tau=cfg["tau"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    logger = RunLogger("dqn", run_id, cfg)
    validator = Validator(logger, q_net)
    steps, episodes = 0, 0
    for batch in collector:
        n = batch.numel()
        steps += n
        buffer.extend(batch.reshape(-1))

        done = batch["next", "done"].squeeze(-1)
        for ep_r in batch["next", "episode_reward"][done].flatten().tolist():
            episodes += 1
            logger.write("episodes", {"episode": episodes, "step": steps, "episode_reward": round(ep_r, 4)})

        if steps >= cfg["warmup_steps"]:
            losses = []
            for _ in range(cfg["updates_per_batch"]):
                sample = buffer.sample()
                loss = loss_fn(sample)["loss"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optim.step()
                target_updater.step()
                losses.append(loss.item())
            egreedy.step(n)
            logger.write("train", {
                "step": steps,
                "loss": round(sum(losses) / len(losses), 6),
                "epsilon": round(float(egreedy.eps), 4),
                "lr": cfg["lr"],
            })
        validator.maybe_check(steps)
        if verbose and steps % 10_000 < n:
            v = validator.values[-1] if validator.values else float("nan")
            print(f"[dqn {run_id}] step {steps:>7}  episodes {episodes:>5}  "
                  f"eps {float(egreedy.eps):.2f}  val reward {v:+.2f}", flush=True)
    collector.shutdown()
    score = validator.finish("dqn", cfg)
    if verbose:
        print(f"[dqn {run_id}] done: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in score.items()})}")
    return score


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="dqn_manual")
    p.add_argument("--steps", type=int, default=DQN_BASELINE["total_steps"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="{}", help="JSON overrides, e.g. '{\"lr\": 3e-4}'")
    a = p.parse_args()
    train_dqn({**json.loads(a.config), "total_steps": a.steps, "seed": a.seed}, a.run_id)
