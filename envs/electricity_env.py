"""
envs/electricity_env.py
=======================
MDP for dynamic electricity pricing.

State:  s_t = (d_t, d_{t-1}, t, p_{t-1})
Action: discrete price level in {0,1,2,3,4} → mapped to [p_min, p_max]
Reward: p_t * d_t  −  λ * max(0, d_t − D_max)²
        + terminal penalty if d_t > D_max (episode ends)

Changes vs original
-------------------
* Accepts an optional `base_demand` parameter so any 24-h demand profile
  can be injected (used by the robustness / multi-demand experiments).
* Default behaviour is identical to the original code.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Default demand base profile (24h) — residential shape
# ---------------------------------------------------------------------------
_DEFAULT_BASE_DEMAND = np.array([
    0.40, 0.35, 0.30, 0.28, 0.28, 0.32,   # 00–05  night valley
    0.50, 0.70, 0.85, 0.80, 0.75, 0.70,   # 06–11  morning ramp + peak
    0.65, 0.60, 0.58, 0.60, 0.65, 0.75,   # 12–17  midday plateau
    0.90, 1.00, 0.95, 0.80, 0.65, 0.50,   # 18–23  evening peak
])


class ElectricityPricingEnv(gym.Env):
    """
    Parameters
    ----------
    alpha          : float       — price elasticity of demand
    sigma          : float       — std of stochastic consumer noise
    lambda_        : float       — penalty weight for exceeding D_max
    D_max          : float       — grid capacity (normalised units)
    p_min          : float       — minimum price level
    p_max          : float       — maximum price level
    p_ref          : float       — reference price for elasticity baseline
    n_price_levels : int         — number of discrete price actions
    base_demand    : np.ndarray | None
                                 — 24-element demand profile; if None the
                                   default residential profile is used
    seed           : int | None
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        alpha: float = 0.3,
        sigma: float = 0.03,
        lambda_: float = 2.0,
        D_max: float = 0.95,
        p_min: float = 0.5,
        p_max: float = 2.0,
        p_ref: float = 1.0,
        n_price_levels: int = 5,
        terminal_on_overload: bool = True,
        terminal_penalty: float = -10.0,
        base_demand: np.ndarray | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        # --- environment parameters ---
        self.alpha = alpha
        self.sigma = sigma
        self.lambda_ = lambda_
        self.D_max = D_max
        self.p_min = p_min
        self.p_max = p_max
        self.p_ref = p_ref
        self.n_price_levels = n_price_levels
        self.terminal_on_overload = terminal_on_overload
        self.terminal_penalty = terminal_penalty

        # demand profile — validated and stored
        if base_demand is not None:
            bd = np.asarray(base_demand, dtype=np.float64)
            if bd.shape != (24,):
                raise ValueError(f"base_demand must have 24 elements, got {bd.shape}")
            self.base_demand = bd
        else:
            self.base_demand = _DEFAULT_BASE_DEMAND.copy()

        # Discrete price grid  e.g. [0.5, 0.875, 1.25, 1.625, 2.0]
        self.price_grid = np.linspace(p_min, p_max, n_price_levels)

        # --- spaces ---
        self.action_space = spaces.Discrete(n_price_levels)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- rng ---
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # internal state
        self._t: int = 0
        self._d_prev: float = 0.0
        self._p_prev: float = p_ref
        self._d_curr: float = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._t = 0
        self._d_curr = float(self.base_demand[0]) + self.np_random.normal(0, self.sigma)
        self._d_curr = np.clip(self._d_curr, 0.0, 1.5)
        self._d_prev = self._d_curr
        self._p_prev = self.p_ref

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        p_t = self.price_grid[action]

        # --- consumer response ---
        d_base = self.base_demand[self._t]
        delta_p = p_t - self.p_ref
        d_t = d_base * (1.0 - self.alpha * delta_p)
        d_t += self.np_random.normal(0.0, self.sigma)
        d_t = float(np.clip(d_t, 0.0, 1.5))

        # --- reward ---
        revenue = p_t * d_t
        overload = max(0.0, d_t - self.D_max)
        penalty = self.lambda_ * (overload ** 2)
        reward = revenue - penalty

        # --- grid failure check ---
        terminated = False
        if self.terminal_on_overload and d_t > self.D_max:
            reward += self.terminal_penalty
            terminated = True

        # --- advance time ---
        self._d_prev = self._d_curr
        self._d_curr = d_t
        self._p_prev = p_t
        self._t += 1

        if self._t >= 24:
            terminated = True

        truncated = False
        info = {
            "hour":    self._t - 1,
            "price":   p_t,
            "demand":  d_t,
            "revenue": revenue,
            "penalty": penalty,
            "overload": overload > 0,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"Hour {self._t:02d} | demand={self._d_curr:.3f} | "
            f"p_prev={self._p_prev:.2f} | D_max={self.D_max:.2f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        d_norm      = np.clip(self._d_curr / 1.5, 0.0, 1.0)
        d_prev_norm = np.clip(self._d_prev / 1.5, 0.0, 1.0)
        t_norm      = self._t / 23.0
        p_norm      = (self._p_prev - self.p_min) / (self.p_max - self.p_min)
        return np.array([d_norm, d_prev_norm, t_norm, p_norm], dtype=np.float32)

    def action_to_price(self, action: int) -> float:
        return float(self.price_grid[action])

    def price_levels(self) -> np.ndarray:
        return self.price_grid.copy()

    def set_base_demand(self, profile: np.ndarray) -> None:
        """Hot-swap the demand profile (used by the multi-demand wrapper)."""
        bd = np.asarray(profile, dtype=np.float64)
        if bd.shape != (24,):
            raise ValueError(f"base_demand must have 24 elements, got {bd.shape}")
        self.base_demand = bd