"""
agents/baselines.py
===================
Simple baseline agents to compare against DQN and PPO.

Agents
------
FixedPriceAgent   : always sets the reference price (p_ref)
HeuristicAgent    : raises price when demand is high, lowers when low
"""

import numpy as np
from envs.electricity_env import ElectricityPricingEnv


class FixedPriceAgent:
    """Always picks the price level closest to p_ref."""

    def __init__(self, env: ElectricityPricingEnv):
        prices = env.price_levels()
        self.action = int(np.argmin(np.abs(prices - env.p_ref)))

    def act(self, obs: np.ndarray) -> int:
        return self.action


class HeuristicAgent:
    """
    Threshold rule:
      demand (normalised) > high_thresh  → highest price
      demand (normalised) < low_thresh   → lowest price
      otherwise                          → middle price

    obs[0] is the normalised current demand (see electricity_env._get_obs).
    """

    def __init__(
        self,
        env: ElectricityPricingEnv,
        high_thresh: float = 0.65,
        low_thresh: float = 0.40,
    ):
        n = env.n_price_levels
        self.high_action = n - 1
        self.low_action  = 0
        self.mid_action  = n // 2
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh

    def act(self, obs: np.ndarray) -> int:
        d_norm = float(obs[0])
        if d_norm > self.high_thresh:
            return self.high_action
        elif d_norm < self.low_thresh:
            return self.low_action
        else:
            return self.mid_action