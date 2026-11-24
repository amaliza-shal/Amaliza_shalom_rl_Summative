"""Figures, generalization tests and the leaderboard.

Run after the sweeps and the 3-seed runs:
  uv run python -m training.plots

Writes to assets/ (PNG figures) and logs/results/ (CSV + markdown tables), and
names the winner in models/best_agent.json.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.baselines import RandomAgent, RuleBasedAgent, evaluate as eval_baseline
from training.common import (EVAL_SEEDS, UNSEEN_SEEDS, evaluate_net, load_policy, steps_to_80_for_run,
                             summarise_eval)

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C",
         "random": "Random", "rule_based": "Rule-based"}
# Fixed categorical order (reference palette slots 1-4). Colour follows the algorithm.
COLOUR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
SEEDS = [0, 1, 2]
OUT = "assets"
RES = os.path.join("logs", "results")

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#bdbbb4", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False,
})


def run_path(algo, seed, name):
    return os.path.join("logs", algo, f"{algo}_best_seed{seed}", name)


def read(algo, seed, name):
    p = run_path(algo, seed, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def on_grid(df, x, y, grid, window):
    """Rolling mean of y, interpolated onto a common step grid."""
    s = df[y].rolling(window, min_periods=1).mean()
    return np.interp(grid, df[x].values, s.values)


def band(ax, grid, curves, colour, label):
    c = np.vstack(curves)
    m, sd = c.mean(0), c.std(0)
    ax.plot(grid, m, color=colour, label=label)
    ax.fill_between(grid, m - sd, m + sd, color=colour, alpha=0.18, linewidth=0)


def save(fig, name, caption):
    """Save a figure. The caption is printed (the report and README carry it under the image),
    not drawn on the figure, so it can never collide with the axes."""
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved assets/{name}  ({caption})")


# ---------------------------------------------------------------------------
# Baselines and generalization numbers
# ---------------------------------------------------------------------------

def baseline_scores():
    rows = {}
    for agent in [RandomAgent(seed=0), RuleBasedAgent()]:
        rows[agent.name] = {
            "eval": eval_baseline(agent, EVAL_SEEDS),
            "unseen": eval_baseline(agent, UNSEEN_SEEDS),
            "hard": eval_baseline(agent, UNSEEN_SEEDS, "hard"),
        }
    return rows


def agent_generalization(algo):
    """Each of the 3 seed models on 100 unseen seeds and on the hard world."""
    out = []
    for s in SEEDS:
        p = run_path(algo, s, "policy.pt")
        if not os.path.exists(p):
            continue
        net = load_policy(p)
        u = summarise_eval(evaluate_net(net, UNSEEN_SEEDS, "standard", net.stochastic))
        h = summarise_eval(evaluate_net(net, UNSEEN_SEEDS, "hard", net.stochastic))
        out.append({"seed": s, "unseen_reward": u["mean_reward"], "unseen_strong": u["finished_strong_share"],
                    "unseen_dropouts": u["mean_dropouts"], "hard_reward": h["mean_reward"],
                    "hard_strong": h["finished_strong_share"], "hard_dropouts": h["mean_dropouts"]})
    return pd.DataFrame(out)


def steps_to_episodes(algo, seed, step):
    ep = read(algo, seed, "episodes")
    return int((ep["step"] <= step).sum()) if ep is not None and step is not None else None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_rewards(base):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    rnd = base["random"]["eval"]["reward"].mean()
    rule = base["rule_based"]["eval"]["reward"].mean()
    for ax, algo in zip(axes.flat, ALGOS):
        grid = np.linspace(2_000, 100_000, 200)
        curves = [on_grid(df, "step", "episode_reward", grid, 100)
                  for s in SEEDS if (df := read(algo, s, "episodes")) is not None]
        if curves:
            band(ax, grid, curves, COLOUR[algo], f"{LABEL[algo]} (mean of 3 seeds)")
        ax.axhline(rule, color=INK, linestyle="--", linewidth=1.3, label=f"Rule-based ({rule:.1f})")
        ax.axhline(rnd, color=MUTED, linestyle=":", linewidth=1.3, label=f"Random ({rnd:.1f})")
        ax.set_title(LABEL[algo])
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.legend(fontsize=8, loc="center right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward (sum over 18 terms)")
    fig.suptitle("Figure 1. Cumulative reward per episode during training", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig1_cumulative_rewards.png",
         "Rolling mean of 100 training episodes; band = +/-1 std across seeds 0-2 (best config of each algorithm). "
         "Training rewards include exploration (epsilon for DQN, sampling for policy-gradient). "
         "Dashed/dotted lines: baselines on the 50 evaluation seeds.")


def fig2_dqn_loss():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    grid = np.linspace(2_000, 100_000, 200)
    curves = [on_grid(df, "step", "loss", grid, 50) for s in SEEDS if (df := read("dqn", s, "train")) is not None]
    if curves:
        band(ax, grid, curves, COLOUR["dqn"], "DQN TD loss (mean of 3 seeds)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("TD loss (MSE, rolling mean)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend()
    fig.suptitle("Figure 2. DQN training objective (temporal-difference loss)", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig2_dqn_loss.png", "Best DQN config, seeds 0-2; band = +/-1 std across seeds.")


def fig3_entropy():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    grid = np.linspace(2_000, 100_000, 200)
    for ax, algo in zip(axes, ["reinforce", "ppo", "a2c"]):
        curves = [on_grid(df, "step", "entropy", grid, 20) for s in SEEDS
                  if (df := read(algo, s, "train")) is not None]
        if curves:
            band(ax, grid, curves, COLOUR[algo], f"{LABEL[algo]} policy entropy")
        ax.axhline(np.log(19), color=MUTED, linestyle=":", linewidth=1.2, label="Uniform over 19 actions (2.94)")
        ax.set_title(LABEL[algo])
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Policy entropy (nats)")
    fig.suptitle("Figure 3. Policy entropy during training (policy-gradient methods)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "fig3_policy_entropy.png", "Best config of each method, seeds 0-2; band = +/-1 std across seeds.")


def fig4_convergence():
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    conv = []
    for i, algo in enumerate(ALGOS):
        ax = fig.add_subplot(gs[0, i])
        for s in SEEDS:
            v = read(algo, s, "validation")
            if v is None:
                continue
            ax.plot(v["step"], v["mean_reward"], color=COLOUR[algo], alpha=0.45 + 0.25 * s, linewidth=1.6)
            st = steps_to_80_for_run(os.path.dirname(run_path(algo, s, "x")))
            y = np.interp(st, v["step"], v["mean_reward"])
            ax.plot(st, y, marker="o", markersize=8, color=COLOUR[algo], markeredgecolor="#fcfcfb",
                    markeredgewidth=2, zorder=5)
            conv.append({"algo": algo, "seed": s, "steps_to_80": st,
                         "episodes_to_80": steps_to_episodes(algo, s, st)})
        ax.set_title(LABEL[algo])
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        if i == 0:
            ax.set_ylabel("Validation reward\n(10 held-out seeds)")
    df = pd.DataFrame(conv)
    ax = fig.add_subplot(gs[1, :])
    width = 0.26
    for j, s in enumerate(SEEDS):
        d = df[df.seed == s].set_index("algo").reindex(ALGOS)
        xs = np.arange(len(ALGOS)) + (j - 1) * width
        ax.bar(xs, d["episodes_to_80"], width=width - 0.03, color=[COLOUR[a] for a in ALGOS],
               alpha=0.55 + 0.2 * j, edgecolor="#fcfcfb", linewidth=2)
        for x, val in zip(xs, d["episodes_to_80"]):
            if pd.notna(val):
                ax.text(x, val, f"{int(val)}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(ALGOS)), [f"{LABEL[a]}\n(seeds 0 | 1 | 2)" for a in ALGOS])
    ax.set_ylabel("Episodes to reach and\nhold 80% of best")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Figure 4. Convergence: when each method reaches and holds 80% of its best validation reward",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "fig4_convergence.png",
         "Top: validation reward every 5,000 steps (lighter = seed 0, darker = seed 2); dot = first point after which the "
         "curve stays above start + 80% of (best - start). Bottom: the same point in training episodes.")
    df.to_csv(os.path.join(RES, "convergence.csv"), index=False)
    return df


def fig5_generalization(gen, base):
    rows = []
    for algo in ALGOS:
        g = gen[algo]
        if g.empty:
            continue
        rows.append({"agent": LABEL[algo], "key": algo,
                     "standard_reward": g["unseen_reward"].mean(), "standard_std": g["unseen_reward"].std(),
                     "standard_strong": g["unseen_strong"].mean(),
                     "hard_reward": g["hard_reward"].mean(), "hard_std": g["hard_reward"].std(),
                     "hard_strong": g["hard_strong"].mean()})
    for name in ["rule_based", "random"]:
        u, h = summarise_eval(base[name]["unseen"].to_dict("records")), \
            summarise_eval(base[name]["hard"].to_dict("records"))
        rows.append({"agent": LABEL[name], "key": name, "standard_reward": u["mean_reward"], "standard_std": 0,
                     "standard_strong": u["finished_strong_share"], "hard_reward": h["mean_reward"], "hard_std": 0,
                     "hard_strong": h["finished_strong_share"]})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    colours = [COLOUR.get(k, "#a9a79f" if k == "rule_based" else "#d6d4cc") for k in t["key"]]
    x = np.arange(len(t))
    for ax, metric, title, fmt in [(axes[0], "reward", "Mean episode reward", "{:+.1f}"),
                                   (axes[1], "strong", "Share of children who finish strong", "{:.0%}")]:
        w = 0.38
        for k, (scen, alpha) in enumerate([("standard", 1.0), ("hard", 0.45)]):
            vals = t[f"{scen}_{metric}"]
            err = t[f"{scen}_std"] if metric == "reward" else None
            ax.bar(x + (k - 0.5) * w, vals, width=w - 0.03, color=colours, alpha=alpha, yerr=err,
                   error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3}, edgecolor="#fcfcfb", linewidth=2,
                   label="100 unseen seeds, standard world" if scen == "standard" else "Harder world")
            tops = vals + (err if err is not None else 0)
            pad = 0.4 if metric == "reward" else 0.01
            for xi, v, top in zip(x + (k - 0.5) * w, vals, tops):
                ax.text(xi, max(top, 0) + pad, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x, t["agent"], fontsize=9)
        ax.set_title(title)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.18)
        ax.axhline(0, color="#bdbbb4", linewidth=1)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Figure 5. Generalization: unseen starting states and a harder world", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "fig5_generalization.png",
         "RL bars: mean of the 3 seed models (error bar = std across seeds), each on 100 unseen seeds. Harder world: "
         "funding-cut chance 25% (not 10%), +0.15 poverty for every child, 1 starting teacher. Solid = standard, faded = hard.")
    out = t.drop(columns=["key"]).round(3)
    out.to_csv(os.path.join(RES, "generalization.csv"), index=False)
    with open(os.path.join(RES, "generalization.md"), "w") as f:
        f.write(out.to_markdown(index=False))
    return t


def fig6_sweeps():
    """Extra: every sweep run per algorithm (supports the hyperparameter tables)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    for ax, algo in zip(axes.flat, ALGOS):
        p = os.path.join("logs", algo, "sweep_results.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        labels = [f"{r.run}. {r.changed} {r.value}" if r.changed != "baseline" else "1. baseline"
                  for r in df.itertuples()]
        y = np.arange(len(df))[::-1]
        bars = ax.barh(y, df["mean_reward"], color=COLOUR[algo], height=0.7,
                       edgecolor="#fcfcfb", linewidth=2)
        for bar, b in zip(bars, df["best"].fillna("")):
            bar.set_alpha(1.0 if b == "BEST" else 0.5)
        ax.errorbar(df["mean_reward"], y, xerr=df["std_reward"] / np.sqrt(50), fmt="none", ecolor=MUTED,
                    elinewidth=1, capsize=2)
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.set_title(f"{LABEL[algo]} - {len(df)} runs (solid = best)")
        ax.set_xlabel("Mean reward on 50 evaluation seeds (error bar = standard error)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 6. Hyperparameter sweeps: every run, same environment, same 100k-step budget",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "fig6_sweeps.png", "One training seed per sweep run; the best run was re-trained with seeds 0-2.")


# ---------------------------------------------------------------------------
# Leaderboard and winner
# ---------------------------------------------------------------------------

def leaderboard(base, gen, conv):
    rows = []
    for name in ["random", "rule_based"]:
        e = summarise_eval(base[name]["eval"].to_dict("records"))
        u = summarise_eval(base[name]["unseen"].to_dict("records"))
        h = summarise_eval(base[name]["hard"].to_dict("records"))
        rows.append({"agent": LABEL[name], "best_run": "-", "mean_reward_3_seeds": round(e["mean_reward"], 2),
                     "std_3_seeds": "-", "steps_to_80": "-", "finished_strong": round(e["finished_strong_share"], 3),
                     "unseen_reward": round(u["mean_reward"], 2), "unseen_std": "-",
                     "hard_reward": round(h["mean_reward"], 2)})
    for algo in ALGOS:
        sums = [json.load(open(run_path(algo, s, "summary.json"))) for s in SEEDS
                if os.path.exists(run_path(algo, s, "summary.json"))]
        if not sums:
            continue
        with open(os.path.join("logs", algo, "best_config.json")) as f:
            best_run = json.load(f)["run"]
        m = [x["mean_reward"] for x in sums]
        g = gen[algo]
        rows.append({"agent": LABEL[algo], "best_run": best_run, "mean_reward_3_seeds": round(np.mean(m), 2),
                     "std_3_seeds": round(float(np.std(m, ddof=1)), 2),
                     "steps_to_80": int(np.mean([steps_to_80_for_run(os.path.dirname(run_path(algo, s, "x")))
                                                 for s in SEEDS if os.path.exists(run_path(algo, s, "summary.json"))])),
                     "finished_strong": round(np.mean([x["finished_strong_share"] for x in sums]), 3),
                     "unseen_reward": round(g["unseen_reward"].mean(), 2),
                     "unseen_std": round(g["unseen_reward"].std(ddof=1), 2),
                     "hard_reward": round(g["hard_reward"].mean(), 2)})
    lb = pd.DataFrame(rows)
    lb.to_csv(os.path.join(RES, "leaderboard.csv"), index=False)
    with open(os.path.join(RES, "leaderboard.md"), "w") as f:
        f.write(lb.to_markdown(index=False))

    # The written rule.
    rl = lb[lb["agent"].isin([LABEL[a] for a in ALGOS])].copy()
    rl = rl.sort_values("unseen_reward", ascending=False)
    first, second = rl.iloc[0], rl.iloc[1]
    gap = first["unseen_reward"] - second["unseen_reward"]
    # Clear only if the gap is larger than the seed-to-seed spread of BOTH leaders.
    spread = max(float(first["unseen_std"]), float(second["unseen_std"]))
    clear = gap >= spread
    gs = {a: gen[a]["unseen_reward"].tolist() for a in ALGOS}
    k1 = {LABEL[a]: a for a in ALGOS}
    a1, a2 = k1[first["agent"]], k1[second["agent"]]
    seed_wins = sum(x > y for x in gs[a1] for y in gs[a2])
    robust = rl.sort_values("hard_reward", ascending=False).iloc[0]
    key = {LABEL[a]: a for a in ALGOS}
    verdict = {
        "algo": key[first["agent"]],
        "best_on_average": first["agent"],
        "unseen_reward": float(first["unseen_reward"]),
        "runner_up": second["agent"],
        "gap": round(float(gap), 3),
        "larger_seed_std": round(spread, 3),
        "gap_vs_seed_std": "clear winner" if clear else "no clear winner (gap < 1 std across seeds)",
        "seed_model_pairs_won": f"{seed_wins}/{len(gs[a1]) * len(gs[a2])}",
        "most_robust_hard_world": robust["agent"],
    }
    with open("models/best_agent.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("\nLEADERBOARD\n" + lb.to_string(index=False))
    print("\nWINNER:", json.dumps(verdict, indent=2))
    return lb, verdict


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    print("Scoring baselines ...")
    base = baseline_scores()
    print("Generalization tests (3 seeds x 100 unseen + 100 hard) ...")
    gen = {a: agent_generalization(a) for a in ALGOS}
    pd.concat([g.assign(algo=a) for a, g in gen.items()]).round(3).to_csv(
        os.path.join(RES, "generalization_per_seed.csv"), index=False)
    print("Figures ...")
    fig1_rewards(base)
    fig2_dqn_loss()
    fig3_entropy()
    conv = fig4_convergence()
    fig5_generalization(gen, base)
    fig6_sweeps()
    leaderboard(base, gen, conv)


if __name__ == "__main__":
    main()
