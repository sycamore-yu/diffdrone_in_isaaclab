"""Rollout buffer for storing trajectory data.

This buffer stores observations, actions, rewards, dones, and values
for computing returns and advantages in SHAC.
"""

from typing import Optional
import torch


class RolloutBuffer:
    """Buffer for storing rollout trajectories.

    This buffer is designed for short-horizon RL algorithms like SHAC.
    It stores the full trajectory for computing GAE advantages.
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the rollout buffer.

        Args:
            num_envs: Number of parallel environments.
            horizon: Number of steps in the rollout.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            device: Device for tensor storage.
        """
        self.num_envs = num_envs
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Allocate storage
        self.obs = torch.zeros(horizon + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, action_dim, device=device)
        # Buffer for masks corresponding to IsaacLab done tuple components
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.terminations = torch.zeros(horizon, num_envs, device=device)
        self.resets = torch.zeros(horizon, num_envs, device=device)
        
        # Policy output buffers
        self.log_probs = torch.zeros(horizon, num_envs, 1, device=device)
        
        # Rewards
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.losses = torch.zeros(horizon, num_envs, device=device)
        
        # Critic values representing V(obs)
        self.values = torch.zeros(horizon, num_envs, 1, device=device)
        self.next_values = torch.zeros(horizon, num_envs, 1, device=device)

        # Bootstrap value (value at final state)
        self.bootstrap_values = torch.zeros(num_envs, 1, device=device)
        self.actor_loss_graph = None
        self.mean_entropy = torch.tensor(0.0, device=device)

        self.ptr = 0
        self.path_start = 0

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        loss: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        terminated: torch.Tensor = None,
        reset: torch.Tensor = None,
        log_prob: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        next_value: Optional[torch.Tensor] = None,
    ):
        """Add a transition to the buffer.

        Args:
            obs: Current observation [num_envs, obs_dim].
            next_obs: Next observation [num_envs, obs_dim].
            action: Action taken [num_envs, action_dim].
            reward: Reward received [num_envs].
            done: Done flag [num_envs].
            log_prob: Log probability of action [num_envs, 1].
            value: Value estimate [num_envs, 1].
        """
        if self.ptr == 0:
            self.obs[0] = obs

        self.obs[self.ptr + 1] = next_obs
        self.actions[self.ptr] = action
        self.losses[self.ptr] = loss
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done.float()

        if log_prob is not None:
            self.log_probs[self.ptr] = log_prob
            
        if value is not None:
            self.values[self.ptr] = value
            
        if next_value is not None:
            self.next_values[self.ptr] = next_value
            
        if terminated is not None:
            self.terminations[self.ptr] = terminated.flatten().float()
            
        if reset is not None:
            self.resets[self.ptr] = reset.flatten().float()

        self.ptr += 1

    def bootstrap(self, value: torch.Tensor):
        """Set bootstrap value for next iteration.

        Args:
            value: Bootstrap value [num_envs, 1].
        """
        self.bootstrap_values = value

    def reset(self):
        """Reset the buffer for a new rollout."""
        self.ptr = 0
        self.actor_loss_graph = None
        self.mean_entropy = torch.tensor(0.0, device=self.device)

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.ptr >= self.horizon

    def get_statistics(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "mean_reward": self.rewards[:self.ptr].mean().item(),
            "mean_done": self.dones[:self.ptr].mean().item(),
            "mean_value": self.values[:self.ptr].mean().item(),
        }


class PrioritizedRolloutBuffer:
    """Prioritized rollout buffer for importance sampling.

    This buffer stores trajectory priorities for weighted updates.
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the prioritized buffer.

        Args:
            num_envs: Number of parallel environments.
            horizon: Number of steps.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            alpha: Priority exponent.
            device: Device for tensor storage.
        """
        self.base_buffer = RolloutBuffer(
            num_envs, horizon, obs_dim, action_dim, device
        )
        self.alpha = alpha
        self.priorities = torch.zeros(horizon, num_envs, device=device)

    def add(self, obs, next_obs, action, loss, reward, done, log_prob=None, value=None):
        """Add transition with priority."""
        self.base_buffer.add(obs, next_obs, action, loss, reward, done, log_prob=log_prob, value=value)
        # Compute priority from TD error
        if value is not None:
            td_error = abs(reward.mean() - value.mean())
            priority = (td_error + 1e-5) ** self.alpha
            self.priorities[self.base_buffer.ptr - 1] = priority

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample from buffer with importance sampling.

        Args:
            batch_size: Batch size.
            beta: Importance sampling exponent.

        Returns:
            Tuple of (obs, actions, returns, advantages, weights).
        """
        # Compute probabilities
        probs = self.priorities[:self.base_buffer.ptr] ** self.alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = torch.multinomial(probs.flatten(), batch_size)
        indices = indices.reshape(batch_size)

        # Compute importance sampling weights
        weights = (self.num_envs * probs.flatten()[indices]) ** (-beta)
        weights = weights / weights.max()

        # Gather data
        obs = self.base_buffer.obs[:, indices // self.num_envs]
        actions = self.base_buffer.actions[:, indices // self.num_envs]
        rewards = self.base_buffer.rewards[:, indices // self.num_envs]

        return obs, actions, rewards, weights

    def __getattr__(self, name):
        """Forward attribute access to base buffer."""
        return getattr(self.base_buffer, name)
