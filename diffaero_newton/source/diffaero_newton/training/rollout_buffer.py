"""Rollout buffers for PPO and Asymmetric PPO training.

Migrated from DiffAero's buffer.py, simplified to remove tensordict
dependency and work with plain tensors.
"""

from typing import Optional
import torch
from torch import Tensor


class RolloutBufferPPO:
    """Rollout buffer for standard PPO training.

    Stores trajectories for on-policy PPO updates with GAE.
    """

    def __init__(self, l_rollout: int, n_envs: int, obs_dim: int,
                 action_dim: int, device: str = "cuda"):
        kw = {"dtype": torch.float32, "device": device}
        self.obs = torch.zeros((l_rollout, n_envs, obs_dim), **kw)
        self.samples = torch.zeros((l_rollout, n_envs, action_dim), **kw)
        self.logprobs = torch.zeros((l_rollout, n_envs), **kw)
        self.rewards = torch.zeros((l_rollout, n_envs), **kw)
        self.next_dones = torch.zeros((l_rollout, n_envs), **kw)
        self.values = torch.zeros((l_rollout, n_envs), **kw)
        self.next_values = torch.zeros((l_rollout, n_envs), **kw)
        self.step = 0

    def clear(self):
        self.step = 0

    @torch.no_grad()
    def add(self, obs: Tensor, sample: Tensor, logprob: Tensor,
            reward: Tensor, next_done: Tensor, value: Tensor,
            next_value: Tensor):
        self.obs[self.step] = obs
        self.samples[self.step] = sample
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.next_dones[self.step] = next_done.float()
        self.values[self.step] = value
        self.next_values[self.step] = next_value
        self.step += 1


class RolloutBufferAPPO(RolloutBufferPPO):
    """Rollout buffer for Asymmetric PPO with privileged state."""

    def __init__(self, l_rollout: int, n_envs: int, obs_dim: int,
                 state_dim: int, action_dim: int, device: str = "cuda"):
        super().__init__(l_rollout, n_envs, obs_dim, action_dim, device)
        kw = {"dtype": torch.float32, "device": device}
        self.states = torch.zeros((l_rollout, n_envs, state_dim), **kw)

    @torch.no_grad()
    def add(self, obs: Tensor, state: Tensor, sample: Tensor,
            logprob: Tensor, reward: Tensor, next_done: Tensor,
            value: Tensor, next_value: Tensor):
        self.states[self.step] = state
        super().add(obs, sample, logprob, reward, next_done, value, next_value)
