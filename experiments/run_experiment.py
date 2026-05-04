"""
experiments/run_experiment.py
==============================
Trains DQN and PPO, evaluates all agents, and saves results.

Usage
-----
    python -m experiments.run_experiment

Output (saved to results/)
--------------------------
    results/dqn_model.pt          trained DQN weights
    results/ppo_model.pt          trained PPO weights
    results/training_curves.png   reward over training episodes
    results/comparison.png        bar charts for all metrics
    results/metrics.txt           mean ± std table
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # no display needed

from envs.electricity_env import ElectricityPricingEnv
from agents.baselines import FixedPriceAgent, HeuristicAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from evaluation.evaluate import evaluate_agent, compare_agents

# ---------------------------------------------------------------------------
# Config — change these to tune your experiments
# ---------------------------------------------------------------------------

ENV_KWARGS = dict(
    alpha   = 0.3,    # price elasticity
    sigma   = 0.03,   # consumer noise
    lambda_ = 2.0,    # overload penalty weight
    D_max   = 0.95,   # grid capacity
    p_min   = 0.5,
    p_max   = 2.0,
    p_ref   = 1.0,
    n_price_levels = 5,
)

DQN_KWARGS = dict(
    lr             = 1e-3,
    gamma          = 0.97,
    epsilon_start  = 1.0,
    epsilon_end    = 0.05,
    epsilon_decay  = 5_000,
    buffer_size    = 10_000,
    batch_size     = 64,
    target_update  = 200,
    hidden_dim     = 64,
    seed           = 42,
)

PPO_KWARGS = dict(
    lr             = 3e-4,
    gamma          = 0.97,
    gae_lambda     = 0.95,
    clip_epsilon   = 0.2,
    n_epochs       = 4,
    batch_size     = 64,
    vf_coef        = 0.5,
    ent_coef       = 0.01,
    max_grad_norm  = 0.5,
    rollout_steps  = 512,
    hidden_dim     = 64,
    seed           = 42,
)

N_TRAIN_EPISODES = 600   # episodes to train each agent
N_EVAL_EPISODES  = 100   # episodes for final evaluation
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(values: list, window: int = 20) -> np.ndarray:
    """Moving average for smoother training curves."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def save_training_curves(dqn_history: dict, ppo_history: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, history, name, color in zip(
        axes,
        [dqn_history, ppo_history],
        ["DQN", "PPO"],
        ["steelblue", "darkorange"],
    ):
        raw    = history["episode_rewards"]
        smoothed = smooth(raw, window=20)
        episodes_raw      = np.arange(1, len(raw) + 1)
        episodes_smoothed = np.arange(len(raw) - len(smoothed) + 1, len(raw) + 1)

        ax.plot(episodes_raw, raw, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(episodes_smoothed, smoothed, color=color, linewidth=2.0, label="smoothed (w=20)")
        ax.set_title(f"{name} — Training Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_comparison_chart(all_metrics: dict, path: str) -> None:
    metrics_to_plot = [
        ("total_reward",    "Total Reward",       "higher is better"),
        ("total_revenue",   "Total Revenue",      "higher is better"),
        ("peak_demand",     "Peak Demand",        "lower is better"),
        ("n_overloads",     "Overload Events",    "lower is better"),
        ("demand_variance", "Demand Variance",    "lower is better"),
    ]

    n   = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    agent_names = list(all_metrics.keys())
    colors = ["slategray", "seagreen", "steelblue", "darkorange"]
    x      = np.arange(len(agent_names))

    for ax, (key, title, note) in zip(axes, metrics_to_plot):
        means = [all_metrics[name][key].mean() for name in agent_names]
        stds  = [all_metrics[name][key].std()  for name in agent_names]

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors[:len(agent_names)], alpha=0.85)
        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_metrics_txt(all_metrics: dict, path: str) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS  (mean ± std over 100 episodes)")
    lines.append("=" * 60)

    for name, metrics in all_metrics.items():
        lines.append(f"\n  {name}")
        lines.append("  " + "-" * 40)
        for key, vals in metrics.items():
            lines.append(f"  {key:22s}: {vals.mean():8.4f}  ±  {vals.std():.4f}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- environments ---
    # separate env instances to avoid state leakage between agents
    env_train_dqn = ElectricityPricingEnv(**ENV_KWARGS, seed=0)
    env_train_ppo = ElectricityPricingEnv(**ENV_KWARGS, seed=1)
    env_eval      = ElectricityPricingEnv(**ENV_KWARGS, seed=99)

    # -----------------------------------------------------------------------
    # 1. Train DQN
    # -----------------------------------------------------------------------
    print("\n" + "="*52)
    print("  Training DQN")
    print("="*52)
    dqn_agent = DQNAgent(env_train_dqn, **DQN_KWARGS)
    dqn_history = dqn_agent.train(n_episodes=N_TRAIN_EPISODES, print_every=100)
    dqn_agent.save(os.path.join(RESULTS_DIR, "dqn_model.pt"))

    # -----------------------------------------------------------------------
    # 2. Train PPO
    # -----------------------------------------------------------------------
    print("\n" + "="*52)
    print("  Training PPO")
    print("="*52)
    ppo_agent = PPOAgent(env_train_ppo, **PPO_KWARGS)
    ppo_history = ppo_agent.train(n_episodes=N_TRAIN_EPISODES, print_every=100)
    ppo_agent.save(os.path.join(RESULTS_DIR, "ppo_model.pt"))

    # -----------------------------------------------------------------------
    # 3. Training curves
    # -----------------------------------------------------------------------
    print("\nSaving training curves...")
    save_training_curves(
        dqn_history, ppo_history,
        os.path.join(RESULTS_DIR, "training_curves.png"),
    )

    # -----------------------------------------------------------------------
    # 4. Final evaluation — all agents on the same env
    # -----------------------------------------------------------------------
    print("\n" + "="*52)
    print("  Final Evaluation")
    print("="*52)

    agents = {
        "Fixed Price": FixedPriceAgent(env_eval),
        "Heuristic":   HeuristicAgent(env_eval),
        "DQN":         dqn_agent,
        "PPO":         ppo_agent,
    }

    all_metrics = compare_agents(agents, env_eval, n_episodes=N_EVAL_EPISODES, seed=42)

    # -----------------------------------------------------------------------
    # 5. Save charts and metrics table
    # -----------------------------------------------------------------------
    print("\nSaving comparison charts and metrics...")
    save_comparison_chart(all_metrics, os.path.join(RESULTS_DIR, "comparison.png"))
    save_metrics_txt(all_metrics,      os.path.join(RESULTS_DIR, "metrics.txt"))

    print("\nDone. Check the results/ folder.")


if __name__ == "__main__":
    main()