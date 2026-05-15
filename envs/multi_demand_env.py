"""
envs/multi_demand_env.py
========================
Wrapper around ElectricityPricingEnv that randomises the demand profile
at the start of each episode.

This is the key building block for training *robust* agents:
instead of overfitting to one fixed BASE_DEMAND, the agent sees a
different profile every episode and must learn a policy that generalises.

Usage
-----
    from envs.multi_demand_env import MultiDemandEnv
    from envs.demand_profiles import ALL_PROFILES

    env = MultiDemandEnv(profiles=ALL_PROFILES, **ENV_KWARGS)
    # use exactly like a regular ElectricityPricingEnv

Design
------
* On every reset() call a profile is sampled uniformly (or according to
  optional weights) from the supplied dictionary.
* The underlying env's base_demand is hot-swapped via set_base_demand().
* All spaces and interfaces are identical to the base env, so DQN / PPO
  agents plug in without modification.
"""

import numpy as np
from envs.electricity_env import ElectricityPricingEnv


class MultiDemandEnv(ElectricityPricingEnv):
    """
    ElectricityPricingEnv with per-episode demand profile randomisation.

    Parameters
    ----------
    profiles : dict[str, np.ndarray]
        Mapping of profile name → 24-element demand array.
        Example: envs.demand_profiles.ALL_PROFILES
    weights  : list[float] | None
        Sampling weights (must sum to 1). If None, uniform sampling.
    **kwargs
        Passed verbatim to ElectricityPricingEnv.__init__.
        Do NOT pass base_demand here; profiles controls that instead.
    """

    def __init__(
        self,
        profiles: dict[str, np.ndarray],
        weights: list[float] | None = None,
        **kwargs,
    ):
        if not profiles:
            raise ValueError("profiles dict must not be empty")

        self._profile_names = list(profiles.keys())
        self._profile_arrays = list(profiles.values())

        if weights is not None:
            if len(weights) != len(self._profile_names):
                raise ValueError("len(weights) must equal len(profiles)")
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()          # normalise to be safe
        else:
            n = len(self._profile_names)
            self._weights = np.ones(n) / n       # uniform

        self._active_profile_name: str = self._profile_names[0]

        # initialise with the first profile; reset() will randomise it
        super().__init__(
            base_demand=self._profile_arrays[0],
            **kwargs,
        )

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Sample a new profile, swap it in, then reset the base env."""
        # We use numpy's default rng for profile selection so it is
        # independent of the env's internal gym rng.
        idx = np.random.choice(len(self._profile_names), p=self._weights)
        self._active_profile_name = self._profile_names[idx]
        self.set_base_demand(self._profile_arrays[idx])

        return super().reset(seed=seed, options=options)

    @property
    def active_profile(self) -> str:
        """Name of the demand profile used in the current episode."""
        return self._active_profile_name