"""
experiments/run_robustness.py
==============================
Robustness experiment: trains specialist and robust agents across
multiple demand profiles and evaluates them in a full cross-matrix.

Experiment structure
--------------------
1. Specialist agents  — DQN and PPO each trained on ONE fixed profile
2. Robust agents      — DQN and PPO each trained on ALL profiles
                        (profile randomised every episode via MultiDemandEnv)
3. Baselines          — FixedPrice and Heuristic (no training needed)
4. Evaluation         — every agent evaluated on every profile separately
5. Outputs            — cross-matrix heatmaps + per-profile bar charts
                        + robustness summary table (metrics.txt)

Output (saved to results/robustness/)
--------------------------------------
    specialist_dqn_<profile>.pt
    specialist_ppo_<profile>.pt
    robust_dqn.pt
    robust_ppo.pt
    heatmap_<metric>.png          cross-matrix heatmaps
    barplot_<profile>.png         per-profile comparison charts
    robustness_metrics.txt        mean ± std table + worst-case column

Usage
-----
    python -m experiments.run_robustness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from envs.electricity_env import ElectricityPricingEnv
from envs.multi_demand_env import MultiDemandEnv
from envs.demand_profiles import ALL_PROFILES
from agents.baselines import FixedPriceAgent, HeuristicAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from evaluation.evaluate import evaluate_agent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_KWARGS = dict(
    alpha=0.3,
    sigma=0.03,
    lambda_=2.0,
    D_max=0.95,
    p_min=0.5,
    p_max=2.0,
    p_ref=1.0,
    n_price_levels=5,
)

DQN_KWARGS = dict(
    lr=1e-3,
    gamma=0.97,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5_000,
    buffer_size=10_000,
    batch_size=64,
    target_update=200,
    hidden_dim=64,
    seed=42,
)

PPO_KWARGS = dict(
    lr=3e-4,
    gamma=0.97,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    n_epochs=4,
    batch_size=64,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    rollout_steps=512,
    hidden_dim=64,
    seed=42,
)

N_TRAIN_EPISODES = 600
N_EVAL_EPISODES  = 100
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "robustness")

PROFILE_NAMES = list(ALL_PROFILES.keys())

# Metrics shown in heatmaps and bar plots
PLOT_METRICS = [
    ("total_reward",    "Total Reward",    "↑ higher is better"),
    ("total_revenue",   "Total Revenue",   "↑ higher is better"),
    ("n_overloads",     "Overload Events", "↓ lower is better"),
    ("peak_demand",     "Peak Demand",     "↓ lower is better"),
    ("demand_variance", "Demand Variance", "↓ lower is better"),
]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_env(profile_name: str | None, seed: int) -> ElectricityPricingEnv:
    """
    Create a training environment.
    profile_name=None → MultiDemandEnv (robust training).
    """
    if profile_name is None:
        return MultiDemandEnv(profiles=ALL_PROFILES, seed=seed, **ENV_KWARGS)
    return ElectricityPricingEnv(
        base_demand=ALL_PROFILES[profile_name], seed=seed, **ENV_KWARGS
    )


def train_dqn(profile_name: str | None, label: str) -> DQNAgent:
    print(f"\n{'='*54}")
    print(f"  Training DQN — {label}")
    print(f"{'='*54}")
    env = _make_env(profile_name, seed=0)
    agent = DQNAgent(env, **DQN_KWARGS)
    agent.train(n_episodes=N_TRAIN_EPISODES, print_every=150)
    return agent


def train_ppo(profile_name: str | None, label: str) -> PPOAgent:
    print(f"\n{'='*54}")
    print(f"  Training PPO — {label}")
    print(f"{'='*54}")
    env = _make_env(profile_name, seed=1)
    agent = PPOAgent(env, **PPO_KWARGS)
    agent.train(n_episodes=N_TRAIN_EPISODES, print_every=150)
    return agent


# ---------------------------------------------------------------------------
# Evaluation: full cross-matrix
# ---------------------------------------------------------------------------

def evaluate_all(
    agents: dict,          # {agent_label: agent}
    n_episodes: int = N_EVAL_EPISODES,
) -> dict:
    """
    Evaluate every agent on every profile.

    Returns
    -------
    results[agent_label][profile_name] = metrics_dict
    """
    results = {label: {} for label in agents}

    for profile_name, profile_arr in ALL_PROFILES.items():
        env_eval = ElectricityPricingEnv(
            base_demand=profile_arr, seed=99, **ENV_KWARGS
        )
        print(f"\n  Evaluating on profile: {profile_name}")
        for label, agent in agents.items():
            m = evaluate_agent(agent, env_eval, n_episodes=n_episodes, seed=42)
            results[label][profile_name] = m
            print(f"    {label:30s}  reward={m['total_reward'].mean():7.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _agent_labels(results: dict) -> list[str]:
    return list(results.keys())


def save_heatmaps(results: dict, out_dir: str) -> None:
    """
    One heatmap per metric.
    Rows = agents, Columns = evaluation profiles.
    Cell value = mean over N_EVAL_EPISODES.
    """
    agent_labels = _agent_labels(results)
    n_agents   = len(agent_labels)
    n_profiles = len(PROFILE_NAMES)

    for metric_key, metric_title, note in PLOT_METRICS:
        matrix = np.zeros((n_agents, n_profiles))
        for i, label in enumerate(agent_labels):
            for j, pname in enumerate(PROFILE_NAMES):
                matrix[i, j] = results[label][pname][metric_key].mean()

        fig, ax = plt.subplots(figsize=(2.5 * n_profiles + 1, 0.8 * n_agents + 1.5))

        # colour direction: for overloads / peak / variance, lower = better → reverse cmap
        reverse = "lower" in note
        cmap = "RdYlGn_r" if reverse else "RdYlGn"
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)

        ax.set_xticks(range(n_profiles))
        ax.set_xticklabels(PROFILE_NAMES, fontsize=10)
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels(agent_labels, fontsize=10)
        ax.set_xlabel("Evaluation Profile", fontsize=11)
        ax.set_ylabel("Agent", fontsize=11)
        ax.set_title(f"{metric_title}  ({note})", fontsize=12, pad=10)

        # annotate cells
        for i in range(n_agents):
            for j in range(n_profiles):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black",
                )

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        plt.tight_layout()
        path = os.path.join(out_dir, f"heatmap_{metric_key}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def save_per_profile_barplots(results: dict, out_dir: str) -> None:
    """
    For each evaluation profile, a grouped bar chart comparing all agents
    across the five key metrics.
    """
    agent_labels = _agent_labels(results)
    n_agents = len(agent_labels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_agents))

    for profile_name in PROFILE_NAMES:
        n_metrics = len(PLOT_METRICS)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
        fig.suptitle(f"Profile: {profile_name}", fontsize=13, fontweight="bold")

        x = np.arange(n_agents)
        for ax, (metric_key, metric_title, note) in zip(axes, PLOT_METRICS):
            means = [results[label][profile_name][metric_key].mean() for label in agent_labels]
            stds  = [results[label][profile_name][metric_key].std()  for label in agent_labels]
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
            ax.set_title(f"{metric_title}\n({note})", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(agent_labels, rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        safe_name = profile_name.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"barplot_{safe_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def save_robustness_summary(results: dict, out_dir: str) -> None:
    """
    Text table: for each agent, mean ± std per metric averaged across all
    profiles, plus the worst-case profile (lowest total_reward).

    This is the key robustness metric: a fragile specialist will have a
    very low worst-case, while a robust agent will be more consistent.
    """
    agent_labels = _agent_labels(results)
    lines = []
    lines.append("=" * 72)
    lines.append("ROBUSTNESS EXPERIMENT — SUMMARY TABLE")
    lines.append(f"(mean ± std over {N_EVAL_EPISODES} episodes per profile, then aggregated)")
    lines.append("=" * 72)

    for label in agent_labels:
        lines.append(f"\n  Agent: {label}")
        lines.append("  " + "-" * 60)

        # collect per-profile means
        profile_rewards = {
            pname: results[label][pname]["total_reward"].mean()
            for pname in PROFILE_NAMES
        }
        worst_profile = min(profile_rewards, key=profile_rewards.get)
        best_profile  = max(profile_rewards, key=profile_rewards.get)
        all_rewards   = np.array(list(profile_rewards.values()))

        lines.append(f"  {'Metric':26s} {'Mean':>9} {'Std':>9} {'Best profile':>16} {'Worst profile':>16}")
        lines.append("  " + "-" * 80)

        for metric_key, metric_title, _ in PLOT_METRICS:
            per_profile_means = np.array([
                results[label][pname][metric_key].mean()
                for pname in PROFILE_NAMES
            ])
            lines.append(
                f"  {metric_title:26s} "
                f"{per_profile_means.mean():9.4f} "
                f"{per_profile_means.std():9.4f} "
                f"{best_profile:>16s} "
                f"{worst_profile:>16s}"
            )

        lines.append(
            f"\n  Reward across profiles: "
            + "  ".join(f"{pname}={v:.3f}" for pname, v in profile_rewards.items())
        )
        lines.append(
            f"  Worst-case reward: {all_rewards.min():.4f}  |  "
            f"Range: {all_rewards.max() - all_rewards.min():.4f}  |  "
            f"(smaller range = more robust)"
        )

    path = os.path.join(out_dir, "robustness_metrics.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Train specialist agents (one per profile, for DQN and PPO)
    # -----------------------------------------------------------------------
    specialist_agents: dict = {}

    for profile_name in PROFILE_NAMES:
        label_dqn = f"DQN-specialist-{profile_name}"
        label_ppo = f"PPO-specialist-{profile_name}"

        dqn = train_dqn(profile_name, label_dqn)
        dqn.save(os.path.join(RESULTS_DIR, f"specialist_dqn_{profile_name}.pt"))
        specialist_agents[label_dqn] = dqn

        ppo = train_ppo(profile_name, label_ppo)
        ppo.save(os.path.join(RESULTS_DIR, f"specialist_ppo_{profile_name}.pt"))
        specialist_agents[label_ppo] = ppo

    # -----------------------------------------------------------------------
    # 2. Train robust agents (multi-demand training)
    # -----------------------------------------------------------------------
    robust_dqn = train_dqn(None, "DQN-robust (all profiles)")
    robust_dqn.save(os.path.join(RESULTS_DIR, "robust_dqn.pt"))

    robust_ppo = train_ppo(None, "PPO-robust (all profiles)")
    robust_ppo.save(os.path.join(RESULTS_DIR, "robust_ppo.pt"))

    # -----------------------------------------------------------------------
    # 3. Baselines (no training)
    #    We use a dummy env just to instantiate them; they only need
    #    n_price_levels and p_ref, which are the same for all profiles.
    # -----------------------------------------------------------------------
    dummy_env = ElectricityPricingEnv(**ENV_KWARGS)
    baselines = {
        "Fixed-Price": FixedPriceAgent(dummy_env),
        "Heuristic":   HeuristicAgent(dummy_env),
    }

    # -----------------------------------------------------------------------
    # 4. Assemble all agents for evaluation
    # -----------------------------------------------------------------------
    all_agents = {
        **baselines,
        **specialist_agents,
        "DQN-robust": robust_dqn,
        "PPO-robust": robust_ppo,
    }

    # -----------------------------------------------------------------------
    # 5. Cross-matrix evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 54)
    print("  Cross-matrix Evaluation")
    print("=" * 54)
    results = evaluate_all(all_agents, n_episodes=N_EVAL_EPISODES)

    # -----------------------------------------------------------------------
    # 6. Save all plots and the metrics table
    # -----------------------------------------------------------------------
    print("\nSaving heatmaps ...")
    save_heatmaps(results, RESULTS_DIR)

    print("\nSaving per-profile bar plots ...")
    save_per_profile_barplots(results, RESULTS_DIR)

    print("\nSaving robustness summary ...")
    save_robustness_summary(results, RESULTS_DIR)

    print("\nDone. All results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()