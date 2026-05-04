# RL Electricity Pricing Project

## Overview
This project implements reinforcement learning agents for dynamic electricity pricing in a simulated grid environment. The goal is to optimize pricing strategies to maximize revenue while avoiding grid overloads through demand management.

## Environment
The `ElectricityPricingEnv` is a Markov Decision Process (MDP) simulating 24-hour electricity demand cycles. Key features:
- **State**: Current demand, previous demand, normalized time, previous price
- **Actions**: 5 discrete price levels (0.5 to 2.0)
- **Reward**: Revenue (price × demand) minus quadratic penalty for overloads
- **Stochastic**: Demand includes noise (σ=0.03) and price elasticity (α=0.3)
- **Termination**: Episode ends at 24 hours or on overload (if enabled)

The environment uses realistic demand profiles with morning/evening peaks and is non-deterministic due to consumer noise.

## Agents
### Baselines
- **Fixed Price**: Constant price at reference level (1.0)
- **Heuristic**: Simple rule-based pricing based on demand trends

### RL Agents
- **DQN (Deep Q-Network)**: Value-based agent using experience replay, target networks, and ε-greedy exploration. Learns Q-values for state-action pairs.
- **PPO (Proximal Policy Optimization)**: Policy-based agent with actor-critic architecture, GAE for advantage estimation, and clipped objective for stable updates.

Both agents use neural networks with 64 hidden units and are trained for 600 episodes.

## Key Results
Evaluation over 100 episodes (mean ± std):

| Agent      | Total Reward | Revenue | Penalties | Peak Demand | Overloads | Demand Variance |
|------------|--------------|---------|-----------|-------------|-----------|-----------------|
| Fixed Price| 0.48 ± 0.79 | 10.50 ± 0.79 | 0.013 ± 0.013 | 1.017 ± 0.044 | 1.0 | 0.048 ± 0.004 |
| Heuristic  | 10.23 ± 6.46| 13.33 ± 2.12 | 0.001 ± 0.002 | 0.935 ± 0.031 | 0.31 | 0.033 ± 0.003 |
| DQN        | **20.56 ± 0.33**| **20.56 ± 0.33**| 0.0 | 0.745 ± 0.052 | 0.0 | 0.024 ± 0.003 |
| PPO        | 20.36 ± 0.27| 20.36 ± 0.27| 0.0 | 0.821 ± 0.028 | 0.0 | 0.032 ± 0.002 |

## Findings
- RL agents significantly outperform baselines, achieving ~95% higher rewards
- DQN shows slightly better performance than PPO in this discrete action space
- Both RL agents eliminate overloads and reduce peak demand compared to baselines
- DQN achieves lower demand variance, indicating more stable pricing

## Training and Evaluation
- **Training**: 600 episodes per agent with standard hyperparameters
- **Evaluation**: 100 episodes for statistical significance
- **Setup**: Stochastic environment with simulated data (no real-world datasets used)

## Visualizations
- `training_curves.png`: Shows smoothed reward curves over training episodes for DQN and PPO, demonstrating convergence to optimal policies
- `comparison.png`: Bar charts comparing all agents across key metrics, highlighting RL superiority

## Usage
Run experiments: `python -m experiments.run_experiment`
Evaluate agents: Use `evaluation/evaluate.py`
Models saved in `results/` directory