"""
agents/dqn.py
=============
Deep Q-Network (DQN) agent for the ElectricityPricingEnv.

Implementation includes:
  - Experience replay buffer
  - Target network (hard update every C steps)
  - Epsilon-greedy exploration with linear decay
  - Simple MLP Q-network

References
----------
Mnih et al. (2015). Human-level control through deep reinforcement learning.
Nature, 518(7540), 529-533.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from envs.electricity_env import ElectricityPricingEnv


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Simple 3-layer MLP that maps state -> Q-values for each action.

    Input  : state vector of shape (obs_dim,)
    Output : Q-values of shape (n_actions,)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Uniform experience replay buffer.

    Stores transitions (s, a, r, s', done) and samples random minibatches.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN agent with experience replay and target network.

    Parameters
    ----------
    env            : ElectricityPricingEnv
    lr             : learning rate for Adam optimizer
    gamma          : discount factor
    epsilon_start  : initial exploration rate
    epsilon_end    : minimum exploration rate
    epsilon_decay  : linear decay steps from start to end
    buffer_size    : replay buffer capacity
    batch_size     : minibatch size for each update
    target_update  : steps between hard target network updates
    hidden_dim     : hidden layer size for Q-network
    seed           : random seed
    """

    def __init__(
        self,
        env: ElectricityPricingEnv,
        lr: float = 1e-3,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5_000,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update: int = 200,
        hidden_dim: int = 64,
        seed: int = 0,
    ):
        self.env        = env
        self.gamma      = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.n_actions = env.action_space.n
        self.obs_dim   = env.observation_space.shape[0]

        # reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.q_net      = QNetwork(self.obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self.steps_done = 0

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        """Linearly decay epsilon from start to end over epsilon_decay steps."""
        progress = min(self.steps_done / self.epsilon_decay, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray, training: bool = False) -> int:
        """
        Select action.
          training=True  → epsilon-greedy (explore)
          training=False → greedy (evaluate)
        """
        if training and random.random() < self._epsilon():
            return self.env.action_space.sample()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _update(self) -> float | None:
        """Sample a minibatch and perform one gradient step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device)
        rewards_t  = torch.tensor(rewards,  device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)

        # Q(s, a)
        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # target: r + gamma * max_a' Q_target(s', a')  (0 if terminal)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 500, print_every: int = 50) -> dict:
        """
        Train the agent for n_episodes.

        Returns
        -------
        history : dict with lists
            episode_rewards  : total reward per episode
            episode_revenues : total revenue per episode
            epsilons         : epsilon at start of each episode
            losses           : mean loss per episode (None if buffer not ready)
        """
        history = {
            "episode_rewards":  [],
            "episode_revenues": [],
            "epsilons":         [],
            "losses":           [],
        }

        for ep in range(1, n_episodes + 1):
            obs, _ = self.env.reset()
            done = False
            ep_reward = ep_revenue = 0.0
            ep_losses = []

            history["epsilons"].append(self._epsilon())

            while not done:
                action = self.act(obs, training=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, reward, next_obs, float(done))
                obs = next_obs

                loss = self._update()
                if loss is not None:
                    ep_losses.append(loss)

                self.steps_done += 1

                # hard update target network
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                ep_reward  += reward
                ep_revenue += info["revenue"]

            history["episode_rewards"].append(ep_reward)
            history["episode_revenues"].append(ep_revenue)
            history["losses"].append(np.mean(ep_losses) if ep_losses else None)

            if ep % print_every == 0:
                recent_r = np.mean(history["episode_rewards"][-print_every:])
                eps = history["epsilons"][-1]
                print(f"  Episode {ep:4d}/{n_episodes} | "
                      f"Avg reward (last {print_every}): {recent_r:7.3f} | "
                      f"ε: {eps:.3f}")

        return history

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")