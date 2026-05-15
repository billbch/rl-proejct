"""
agents/ppo.py
=============
Proximal Policy Optimization (PPO) agent for the ElectricityPricingEnv.

Implementation includes:
  - Actor-Critic architecture (shared backbone)
  - GAE (Generalised Advantage Estimation) for variance reduction
  - PPO-Clip objective
  - Multiple epochs of minibatch updates per rollout
  - Entropy bonus to encourage exploration

References
----------
Schulman et al. (2017). Proximal Policy Optimization Algorithms.
arXiv:1707.06347.

Schulman et al. (2016). High-Dimensional Continuous Control Using
Generalised Advantage Estimation. ICLR 2016.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from envs.electricity_env import ElectricityPricingEnv


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic network.

    The backbone extracts features from the state.
    The actor head outputs a probability distribution over actions (policy).
    The critic head outputs a scalar state value V(s).

    Input  : state vector of shape (obs_dim,)
    Output : action logits (n_actions,)  +  value scalar (1,)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()

        # shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # actor: policy logits
        self.actor_head = nn.Linear(hidden_dim, n_actions)

        # critic: state value
        self.critic_head = nn.Linear(hidden_dim, 1)

        # initialise output layers with small weights for stability
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(x)
        return value


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one rollout (N steps) of experience for PPO updates.
    Unlike DQN's replay buffer, this is cleared after each policy update.
    """

    def __init__(self):
        self.obs      = []
        self.actions  = []
        self.rewards  = []
        self.values   = []
        self.log_probs= []
        self.dones    = []

    def push(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent with clipped surrogate objective and GAE.

    Parameters
    ----------
    env           : ElectricityPricingEnv
    lr            : learning rate (same for actor and critic)
    gamma         : discount factor
    gae_lambda    : GAE lambda for advantage estimation (0=TD, 1=MC)
    clip_epsilon  : PPO clipping parameter
    n_epochs      : number of update epochs per rollout
    batch_size    : minibatch size within each epoch
    vf_coef       : value function loss coefficient
    ent_coef      : entropy bonus coefficient (encourages exploration)
    max_grad_norm : gradient clipping threshold
    rollout_steps : steps collected before each update (must be >= 1 episode)
    hidden_dim    : hidden layer size
    seed          : random seed
    """

    def __init__(
        self,
        env: ElectricityPricingEnv,
        lr: float = 3e-4,
        gamma: float = 0.97,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 512,
        hidden_dim: int = 64,
        seed: int = 0,
    ):
        self.env          = env
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.vf_coef      = vf_coef
        self.ent_coef     = ent_coef
        self.max_grad_norm= max_grad_norm
        self.rollout_steps= rollout_steps

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim        = env.observation_space.shape[0]
        n_actions      = env.action_space.n

        self.network   = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

    # ------------------------------------------------------------------
    # Inference (evaluation mode — greedy)
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> int:
        """Greedy action for evaluation (no exploration)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(obs_t)
            action = logits.argmax(dim=-1)
        return int(action.item())

    # ------------------------------------------------------------------
    # GAE — Generalised Advantage Estimation
    # ------------------------------------------------------------------

    def _compute_gae(self, last_value: float) -> tuple:
        """
        Compute advantages using GAE and returns (advantages, value targets).

        GAE smoothly interpolates between TD(0) (low variance, high bias)
        and Monte Carlo (high variance, low bias) via gae_lambda.
        """
        rewards   = self.buffer.rewards
        values    = self.buffer.values
        dones     = self.buffer.dones
        T         = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            next_value  = last_value if t == T - 1 else values[t + 1]
            next_done   = dones[t]
            delta       = rewards[t] + self.gamma * next_value * (1.0 - next_done) - values[t]
            gae         = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae

        value_targets = advantages + np.array(values, dtype=np.float32)
        return advantages, value_targets

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update(self, last_value: float) -> dict:
        """Run n_epochs of minibatch PPO updates on the current rollout."""
        advantages, value_targets = self._compute_gae(last_value)

        # normalise advantages (reduces variance, standard practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert buffer to tensors
        obs_t      = torch.tensor(np.array(self.buffer.obs),       dtype=torch.float32, device=self.device)
        actions_t  = torch.tensor(np.array(self.buffer.actions),   dtype=torch.long,    device=self.device)
        old_lp_t   = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        adv_t      = torch.tensor(advantages,                       dtype=torch.float32, device=self.device)
        vt_t       = torch.tensor(value_targets,                    dtype=torch.float32, device=self.device)

        T = len(self.buffer)
        losses = {"policy": [], "value": [], "entropy": [], "total": []}

        for _ in range(self.n_epochs):
            # shuffle indices for minibatch sampling
            indices = np.random.permutation(T)

            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                _, new_lp, entropy, new_value = self.network.get_action_and_value(
                    obs_t[idx], actions_t[idx]
                )

                # PPO clipped surrogate objective
                ratio       = torch.exp(new_lp - old_lp_t[idx])
                surr1       = ratio * adv_t[idx]
                surr2       = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # value function loss
                value_loss  = nn.functional.mse_loss(new_value, vt_t[idx])

                # entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(-entropy_loss.item())
                losses["total"].append(total_loss.item())

        self.buffer.clear()
        return {k: np.mean(v) for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 500, print_every: int = 50) -> dict:
        """
        Train the agent for n_episodes.

        PPO collects rollout_steps transitions, then updates.
        Episodes are tracked for logging even if they span multiple rollouts.

        Returns
        -------
        history : dict with lists per episode:
            episode_rewards, episode_revenues, policy_losses, value_losses
        """
        history = {
            "episode_rewards":  [],
            "episode_revenues": [],
            "policy_losses":    [],
            "value_losses":     [],
        }

        obs, _ = self.env.reset()
        ep_reward = ep_revenue = 0.0
        episodes_done = 0

        # we'll collect steps until we've finished n_episodes
        while episodes_done < n_episodes:

            # --- collect rollout_steps transitions ---
            for _ in range(self.rollout_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, _, value = self.network.get_action_and_value(obs_t)

                action_np = int(action.item())
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.push(
                    obs,
                    action_np,
                    reward,
                    float(value.item()),
                    float(log_prob.item()),
                    float(done),
                )

                ep_reward  += reward
                ep_revenue += info["revenue"]
                obs         = next_obs

                if done:
                    history["episode_rewards"].append(ep_reward)
                    history["episode_revenues"].append(ep_revenue)
                    ep_reward = ep_revenue = 0.0
                    episodes_done += 1
                    obs, _ = self.env.reset()

                    if episodes_done % print_every == 0:
                        recent_r = np.mean(history["episode_rewards"][-print_every:])
                        print(f"  Episode {episodes_done:4d}/{n_episodes} | "
                              f"Avg reward (last {print_every}): {recent_r:7.3f}")

                    if episodes_done >= n_episodes:
                        break

            # --- PPO update ---
            with torch.no_grad():
                obs_t      = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = float(self.network.get_value(obs_t).item())

            update_info = self._update(last_value)
            history["policy_losses"].append(update_info["policy"])
            history["value_losses"].append(update_info["value"])

        return history

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")a