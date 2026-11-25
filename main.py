"""Run (play) a trained School Director agent with the Pygame GUI and a
one-line-per-term terminal log. This is the project's play script.

  uv sync
  uv run main.py                            # the best agent (leaderboard winner)
  uv run main.py --algo ppo                 # any trained agent: dqn, reinforce, ppo, a2c
  uv run main.py --algo dqn --seed 7 --delay 3
  uv run main.py --algo rule                # the rule-based baseline
  uv run main.py --algo ppo --scenario hard # the harder world
  uv run main.py --algo ppo --gif assets/demo.gif --no-window
  uv run main.py --algo dqn --view3d        # also open a three.js 3D replay in the browser

Every episode is saved as JSON in logs/episodes/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from environment.custom_env import ACTION_NAMES, SchoolEnv
from environment.rendering import fmt_value

ALGOS = ["dqn", "reinforce", "ppo", "a2c", "rule", "random", "best"]


def make_actor(algo: str):
    """Return (name, function obs, env -> action)."""
    if algo == "rule":
        from training.baselines import RuleBasedAgent
        a = RuleBasedAgent()
        return "rule-based", a.act
    if algo == "random":
        from training.baselines import RandomAgent
        a = RandomAgent(seed=0)
        return "random", a.act
    import torch

    from training.predict import best_algo, load_agent
    from training.common import policy_action
    algo = best_algo() if algo == "best" else algo
    net = load_agent(algo)
    gen = torch.Generator()

    def act(obs, env):
        if env.term == 0:  # same sampling stream as the evaluation for this seed
            gen.manual_seed(env.seed_used)
        return policy_action(net, obs, net.stochastic, gen)
    return algo.upper(), act


def wait(env, seconds: float, window: bool):
    """Keep the window alive (and animated) while we pause between terms."""
    end = time.time() + seconds
    while time.time() < end:
        if window:
            env.render()
        time.sleep(1 / 30)


def hold_until_key(env):
    """Keep the final report card on screen until a key is pressed or the window is closed."""
    env.end_hint = "Press any key or close the window to exit"
    renderer = env._renderer
    renderer.key_pressed = False
    try:
        while not renderer.key_pressed:
            env.render()
            time.sleep(1 / 30)
    except SystemExit:
        pass


def print_report(env, cfg):
    """Clean per-child summary at the end of an episode."""
    print("\n  Child      | Learning | Attendance | Confidence | Result")
    print("  " + "-" * 66)
    for c in env.children:
        learn_ok = c["learning"] >= cfg["finish_learning"]
        conf_ok = c["confidence"] >= cfg["finish_confidence"]
        if not c["enrolled"]:
            why = "Dropped out"
        elif c["finished_strong"]:
            why = "Finished strong"
        elif env.end_reason != "Term 18 reached":
            why = "School closed early"
        elif not learn_ok and not conf_ok:
            why = "Learning and confidence below target"
        elif not learn_ok:
            why = "Learning below 0.65"
        else:
            why = "Confidence below 0.55"
        learn = fmt_value(c["learning"], cfg["finish_learning"])
        conf = fmt_value(c["confidence"], cfg["finish_confidence"])
        print(f"  {c['name']:<10} |   {learn:<5}  |    {c['attendance']:.2f}    |    {conf:<5}   | {why}")
    used = sorted(((n, ACTION_NAMES[i]) for i, n in enumerate(env.action_counts) if n), reverse=True)
    print("\n  Decisions: " + ", ".join(f"{name} x{n}" for n, name in used))


def on_track(env) -> int:
    return sum(c["enrolled"] and c["learning"] >= 0.65 and c["confidence"] >= 0.55
               for c in env.children)


def play(algo="best", seed=0, delay=1.5, scenario="standard", window=True, gif=None,
         episodes=1, log=True, view3d=False, hold=True):
    name, act = make_actor(algo)
    render_mode = "human" if window else ("rgb_array" if gif else None)
    env = SchoolEnv(render_mode=render_mode, scenario=scenario)
    env.agent_label = name if name in ("rule-based", "random") else f"{name} agent"
    frames = []
    results = []
    for ep in range(episodes):
        ep_seed = seed + ep
        env.end_hint = ""
        env.agent_label = env.agent_label.split("  |")[0] + f"  |  seed {ep_seed}"
        obs, info = env.reset(seed=ep_seed)
        env.seed_used = ep_seed
        print(f"\n=== {name} director | seed {ep_seed} | {scenario} world ===")
        print(f"{'term':>4} | {'action':<26} | {'reward':>7} | {'total':>7} | budget | tchr | trust "
              f"| on-track | events")
        record = {"agent": name, "seed": ep_seed, "scenario": scenario, "terms": []}
        if gif:
            frames.append(env.render() if render_mode == "rgb_array" else None)
        wait(env, delay, window)
        done = False
        while not done:
            before = env.get_state()
            action = act(obs, env)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            events = "; ".join(info["events"]) or "-"
            print(f"{env.term:>4} | {ACTION_NAMES[action]:<26} | {r:+7.2f} | {env.total_reward:+7.2f} "
                  f"|  {env.budget:.2f}  |  {env.teachers}   | {env.trust:.2f} |   {on_track(env)}/6    | {events}",
                  flush=True)
            record["terms"].append({"term": env.term, "action": ACTION_NAMES[action], "reward": round(r, 4),
                                    "state_before": before, "events": info["events"]})
            if gif and render_mode == "rgb_array":
                frames.append(env.render())
            wait(env, delay, window)
        record["final"] = env.get_state()
        print(f"--- End: {info['end_reason']}. Finished strong: {info['finished_strong']}/6, "
              f"dropouts: {info['dropouts']}, teacher quits: {info['teacher_quits']}, "
              f"total reward: {env.total_reward:+.2f}")
        print_report(env, env.cfg)
        results.append(record)
        if log:
            os.makedirs("logs/episodes", exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"logs/episodes/{name.lower()}_seed{ep_seed}_{scenario}_{stamp}.json"
            with open(path, "w") as f:
                json.dump(record, f, indent=1)
            print(f"Episode saved to {path}")
        if view3d:
            from environment.rendering import export_3d_replay
            html = export_3d_replay(record, f"logs/episodes/{name.lower()}_seed{ep_seed}_{scenario}_3d.html")
            print(f"3D replay (three.js) opened in your browser: {html}")
        if gif and render_mode == "rgb_array":  # hold the report card in the GIF
            frames.extend([env.render()] * 4)
        if window and ep == episodes - 1 and hold:
            print("Report card on screen. Press any key in the window (or close it) to exit.")
            hold_until_key(env)
        else:
            wait(env, max(3.0, 3 * delay), window)
    if gif and frames and frames[0] is not None:
        import imageio
        imageio.mimsave(gif, frames, duration=1.0, loop=0)
        print(f"GIF saved to {gif}")
    env.close()
    return results


def main():
    p = argparse.ArgumentParser(description="Run a trained School Director agent.")
    p.add_argument("--algo", default="best", choices=ALGOS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=1.5, help="seconds per term")
    p.add_argument("--scenario", default="standard", choices=["standard", "hard"])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-window", action="store_true", help="no GUI (terminal only)")
    p.add_argument("--gif", default=None, help="save the episode as a GIF")
    p.add_argument("--view3d", action="store_true", help="open a three.js 3D replay in the browser")
    p.add_argument("--no-hold", action="store_true",
                   help="close the window by itself at the end instead of waiting for a key")
    a = p.parse_args()
    play(a.algo, a.seed, a.delay if not a.no_window else 0.0, a.scenario,
         window=not a.no_window, gif=a.gif, episodes=a.episodes, view3d=a.view3d, hold=not a.no_hold)


if __name__ == "__main__":
    main()
