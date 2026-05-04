"""
evaluation/evaluate.py
======================
Evaluation utilities for all agents (baselines, DQN, PPO).

Usage
-----
    from evaluation.evaluate import evaluate_agent, print_metrics, compare_agents

The agent only needs to implement  .act(obs) -> int.
DQN and PPO wrappers in their respective files expose this same interface.
"""

import numpy as np
from envs.electricity_env import ElectricityPricingEnv


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent,
    env: ElectricityPricingEnv,
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run agent for n_episodes and collect per-episode metrics.

    Returns
    -------
    dict with np.ndarray of length n_episodes:
        total_reward    : sum of rewards in the episode
        total_revenue   : sum of p_t * d_t  (before penalties)
        total_penalty   : sum of lambda * overload²
        peak_demand     : max demand seen in the episode
        n_overloads     : number of hours where d_t > D_max
        demand_variance : variance of demand across the episode
    """
    rng = np.random.default_rng(seed)

    metrics = {
        "total_reward":    np.zeros(n_episodes),
        "total_revenue":   np.zeros(n_episodes),
        "total_penalty":   np.zeros(n_episodes),
        "peak_demand":     np.zeros(n_episodes),
        "n_overloads":     np.zeros(n_episodes),
        "demand_variance": np.zeros(n_episodes),
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        done = False
        ep_reward = ep_revenue = ep_penalty = 0.0
        demands = []

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward  += reward
            ep_revenue += info["revenue"]
            ep_penalty += info["penalty"]
            demands.append(info["demand"])
            if info["overload"]:
                metrics["n_overloads"][ep] += 1

        demands = np.array(demands)
        metrics["total_reward"][ep]    = ep_reward
        metrics["total_revenue"][ep]   = ep_revenue
        metrics["total_penalty"][ep]   = ep_penalty
        metrics["peak_demand"][ep]     = demands.max()
        metrics["demand_variance"][ep] = demands.var()

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics(name: str, metrics: dict) -> None:
    """Print mean ± std for each metric."""
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    for key, vals in metrics.items():
        print(f"  {key:22s}: {vals.mean():8.4f}  ±  {vals.std():.4f}")


def compare_agents(agents: dict, env: ElectricityPricingEnv, n_episodes: int = 100, seed: int = 42) -> dict:
    """
    Evaluate multiple agents and print a comparison table.

    Parameters
    ----------
    agents : dict  {name: agent}  — agent must implement .act(obs)
    env    : ElectricityPricingEnv
    n_episodes : int
    seed   : int   — same seed for all agents so results are comparable

    Returns
    -------
    dict {name: metrics_dict}
    """
    all_metrics = {}
    for name, agent in agents.items():
        print(f"Evaluating: {name} ...")
        m = evaluate_agent(agent, env, n_episodes=n_episodes, seed=seed)
        all_metrics[name] = m
        print_metrics(name, m)

    # --- summary table ---
    print(f"\n{'='*52}")
    print("  SUMMARY  (mean over episodes)")
    print(f"{'='*52}")
    header = f"{'Agent':22s} {'Reward':>10} {'Revenue':>10} {'Peak D':>8} {'Overloads':>10}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(
            f"{name:22s} "
            f"{m['total_reward'].mean():10.3f} "
            f"{m['total_revenue'].mean():10.3f} "
            f"{m['peak_demand'].mean():8.3f} "
            f"{m['n_overloads'].mean():10.2f}"
        )

    return all_metrics