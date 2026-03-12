"""PPO and Asymmetric PPO training algorithms.

Migrated from DiffAero's PPO.py to work natively with IsaacLab DirectRLEnv
and Newton differentiable physics.

Key features:
- PPO: Standard clipped surrogate objective with GAE
- AsymmetricPPO: Critic uses privileged state information while
  actor operates on observations only (teacher-student style)
"""

from typing import Dict, Optional, Tuple
from collections import defaultdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.training.rollout_buffer import RolloutBufferPPO, RolloutBufferAPPO


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class StochasticActorCritic(nn.Module):
    """Combined actor-critic for PPO."""

    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM,
                 hidden_dims: list = None, log_std_init: float = -0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Actor
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            actor_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.actor_backbone = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(in_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        # Critic
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            critic_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns (action, sample, logprob, entropy, value)."""
        h = self.actor_backbone(obs)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std.clamp(-20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            sample = mean
        else:
            sample = dist.rsample()

        action = torch.sigmoid(sample)
        logprob = dist.log_prob(sample).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return action, sample, logprob, entropy, value

    def get_action(self, obs: torch.Tensor, sample: torch.Tensor):
        """Re-evaluate log_prob and entropy for a given sample."""
        h = self.actor_backbone(obs)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std.clamp(-20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = torch.sigmoid(sample)
        logprob = dist.log_prob(sample).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, sample, logprob, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class AsymmetricActorCritic(nn.Module):
    """Actor uses obs, Critic uses privileged state."""

    def __init__(self, obs_dim: int, state_dim: int,
                 action_dim: int = ACTION_DIM,
                 actor_hidden: list = None, critic_hidden: list = None,
                 log_std_init: float = -0.5):
        super().__init__()
        if actor_hidden is None:
            actor_hidden = [256, 256, 128]
        if critic_hidden is None:
            critic_hidden = [256, 256, 128]

        # Actor (uses obs)
        actor_layers = []
        in_dim = obs_dim
        for h in actor_hidden:
            actor_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.actor_backbone = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(in_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        # Critic (uses privileged state)
        critic_layers = []
        in_dim = state_dim
        for h in critic_hidden:
            critic_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def get_action(self, obs: torch.Tensor, sample: torch.Tensor = None,
                   deterministic: bool = False):
        h = self.actor_backbone(obs)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std.clamp(-20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        if sample is None:
            sample = mean if deterministic else dist.rsample()
        action = torch.sigmoid(sample)
        logprob = dist.log_prob(sample).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, sample, logprob, entropy

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class PPO:
    """Proximal Policy Optimization with GAE and clipped surrogate loss."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        n_envs: int = 64,
        l_rollout: int = 16,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        clip_coef: float = 0.2,
        entropy_weight: float = 0.01,
        value_weight: float = 0.5,
        max_grad_norm: float = 1.0,
        n_minibatch: int = 4,
        n_epoch: int = 4,
        clip_value_loss: bool = True,
        norm_adv: bool = True,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.agent = StochasticActorCritic(obs_dim, action_dim).to(self.device)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBufferPPO(l_rollout, n_envs, obs_dim, action_dim, device)

        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_coef = clip_coef
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.max_grad_norm = max_grad_norm
        self.n_minibatch = n_minibatch
        self.n_epoch = n_epoch
        self.clip_value_loss = clip_value_loss
        self.norm_adv = norm_adv
        self.n_envs = n_envs
        self.l_rollout = l_rollout

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(obs, deterministic)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy, "value": value}

    @torch.no_grad()
    def bootstrap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and target values."""
        advantages = torch.zeros_like(self.buffer.rewards)
        lastgaelam = 0
        for t in reversed(range(self.l_rollout)):
            nextnonterminal = 1.0 - self.buffer.next_dones[t]
            nextvalues = self.buffer.next_values[t]
            delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lmbda * nextnonterminal * lastgaelam
        target_values = advantages + self.buffer.values
        return advantages.view(-1), target_values.view(-1)

    def train_epoch(self, advantages: torch.Tensor, target_values: torch.Tensor) -> Tuple[Dict, Dict]:
        """One epoch of minibatch PPO updates."""
        T, N = self.l_rollout, self.n_envs
        obs = self.buffer.obs.flatten(0, 1)
        samples = self.buffer.samples.flatten(0, 1)
        logprobs = self.buffer.logprobs.flatten(0, 1)
        values = self.buffer.values.flatten(0, 1)

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_indices = torch.randperm(T * N, device=self.device)
        mb_size = T * N // self.n_minibatch
        losses = defaultdict(list)
        grad_norms = defaultdict(list)

        for start in range(0, T * N, mb_size):
            mb = batch_indices[start:start + mb_size]

            _, _, newlogprob, entropy = self.agent.get_action(obs[mb], samples[mb])
            logratio = newlogprob - logprobs[mb]
            ratio = logratio.exp()

            # Clipped surrogate loss
            adv_mb = advantages[mb]
            pg_loss = torch.max(
                -adv_mb * ratio,
                -adv_mb * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            ).mean()

            entropy_loss = -entropy.mean()

            # Value loss
            newvalue = self.agent.get_value(obs[mb])
            if self.clip_value_loss:
                v_unclipped = (newvalue - target_values[mb]) ** 2
                v_clipped = values[mb] + torch.clamp(
                    newvalue - values[mb], -self.clip_coef, self.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - target_values[mb]) ** 2).mean()
            else:
                v_loss = F.mse_loss(newvalue, target_values[mb])

            loss = pg_loss + self.value_weight * v_loss + self.entropy_weight * entropy_loss
            self.optim.zero_grad()
            loss.backward()

            actor_gn = sum(p.grad.data.norm().item() ** 2 for p in self.agent.actor_backbone.parameters() if p.grad is not None) ** 0.5
            critic_gn = sum(p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters() if p.grad is not None) ** 0.5
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optim.step()

            losses["actor_loss"].append(pg_loss.item())
            losses["entropy_loss"].append(entropy_loss.item())
            losses["critic_loss"].append(v_loss.item())
            grad_norms["actor_grad_norm"].append(actor_gn)
            grad_norms["critic_grad_norm"].append(critic_gn)

        return (
            {k: sum(v) / len(v) for k, v in losses.items()},
            {k: sum(v) / len(v) for k, v in grad_norms.items()},
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.agent.state_dict(), os.path.join(path, "ppo_agent.pt"))

    def load(self, path: str):
        self.agent.load_state_dict(
            torch.load(os.path.join(path, "ppo_agent.pt"), map_location=self.device))


# ---------------------------------------------------------------------------
# AsymmetricPPO
# ---------------------------------------------------------------------------

class AsymmetricPPO(PPO):
    """PPO variant where the critic uses privileged state information."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int = ACTION_DIM,
        n_envs: int = 64,
        l_rollout: int = 16,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        clip_coef: float = 0.2,
        entropy_weight: float = 0.01,
        value_weight: float = 0.5,
        max_grad_norm: float = 1.0,
        n_minibatch: int = 4,
        n_epoch: int = 4,
        clip_value_loss: bool = True,
        norm_adv: bool = True,
        device: str = "cuda",
    ):
        # Skip PPO.__init__ to use different agent type
        self.device = torch.device(device)
        self.agent = AsymmetricActorCritic(obs_dim, state_dim, action_dim).to(self.device)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBufferAPPO(l_rollout, n_envs, obs_dim, state_dim, action_dim, device)

        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_coef = clip_coef
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.max_grad_norm = max_grad_norm
        self.n_minibatch = n_minibatch
        self.n_epoch = n_epoch
        self.clip_value_loss = clip_value_loss
        self.norm_adv = norm_adv
        self.n_envs = n_envs
        self.l_rollout = l_rollout

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        action, sample, logprob, entropy = self.agent.get_action(obs, deterministic=deterministic)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy}

    def train_epoch(self, advantages: torch.Tensor, target_values: torch.Tensor):
        T, N = self.l_rollout, self.n_envs
        obs = self.buffer.obs.flatten(0, 1)
        states = self.buffer.states.flatten(0, 1)
        samples = self.buffer.samples.flatten(0, 1)
        logprobs = self.buffer.logprobs.flatten(0, 1)
        values = self.buffer.values.flatten(0, 1)

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_indices = torch.randperm(T * N, device=self.device)
        mb_size = T * N // self.n_minibatch
        losses = defaultdict(list)
        grad_norms = defaultdict(list)

        for start in range(0, T * N, mb_size):
            mb = batch_indices[start:start + mb_size]

            _, _, newlogprob, entropy = self.agent.get_action(obs[mb], samples[mb])
            logratio = newlogprob - logprobs[mb]
            ratio = logratio.exp()
            adv_mb = advantages[mb]
            pg_loss = torch.max(
                -adv_mb * ratio,
                -adv_mb * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            ).mean()
            entropy_loss = -entropy.mean()

            # Critic uses privileged state
            newvalue = self.agent.get_value(states[mb])
            if self.clip_value_loss:
                v_unclipped = (newvalue - target_values[mb]) ** 2
                v_clipped = values[mb] + torch.clamp(
                    newvalue - values[mb], -self.clip_coef, self.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - target_values[mb]) ** 2).mean()
            else:
                v_loss = F.mse_loss(newvalue, target_values[mb])

            loss = pg_loss + self.value_weight * v_loss + self.entropy_weight * entropy_loss
            self.optim.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optim.step()

            losses["actor_loss"].append(pg_loss.item())
            losses["entropy_loss"].append(entropy_loss.item())
            losses["critic_loss"].append(v_loss.item())

        return (
            {k: sum(v) / len(v) for k, v in losses.items()},
            {k: sum(v) / len(v) for k, v in grad_norms.items()} if grad_norms else {},
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.agent.state_dict(), os.path.join(path, "appo_agent.pt"))

    def load(self, path: str):
        self.agent.load_state_dict(
            torch.load(os.path.join(path, "appo_agent.pt"), map_location=self.device))
