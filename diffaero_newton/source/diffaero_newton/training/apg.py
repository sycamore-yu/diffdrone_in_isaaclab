"""APG (Analytic Policy Gradient) and APG_stochastic training algorithms.

Migrated from DiffAero's APG.py to work natively with IsaacLab DirectRLEnv
and Newton differentiable physics.

Key differences from SHAC:
- APG: Pure deterministic actor, no critic. Loss flows directly through
  the differentiable environment.
- APG_stochastic: Stochastic actor with entropy regularisation.
  Still critic-free; the loss graph is maintained end-to-end.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from diffaero_newton.common.constants import ACTION_DIM


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.ELU()))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def _compute_grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    return sum(
        parameter.grad.data.norm().item() ** 2
        for parameter in parameters
        if parameter.grad is not None
    ) ** 0.5


class DeterministicActor(nn.Module):
    """Simple deterministic MLP actor."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (256, 256, 128)
        self.net = _build_mlp(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            output_activation=nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class StochasticActor(nn.Module):
    """Gaussian stochastic actor with learnable log-std."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dims: Sequence[int] | None = None,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (256, 256, 128)
        self.backbone = _build_mlp(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
        )
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the action, log-probability, and entropy."""

        mean = self.mean_head(self.backbone(obs))
        if deterministic:
            zeros = torch.zeros(obs.shape[0], 1, device=obs.device)
            return torch.sigmoid(mean), zeros, zeros

        std = self.log_std.clamp(-20, 2).exp()
        dist = Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return torch.sigmoid(sample), log_prob, entropy


class APG:
    """Deterministic analytic policy gradient over a differentiable environment."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        lr: float = 3e-4,
        max_grad_norm: float | None = 1.0,
        l_rollout: int = 16,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.actor = DeterministicActor(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.l_rollout = l_rollout
        self.actor_loss = torch.zeros(1, device=self.device)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def record_loss(self, loss: torch.Tensor) -> None:
        """Accumulate per-step loss into the rolling actor loss."""

        self.actor_loss = self.actor_loss + loss.mean()

    def update_actor(self) -> dict[str, float]:
        """Back-propagate through the accumulated loss graph and step once."""

        self.actor_loss = self.actor_loss / self.l_rollout
        self.optimizer.zero_grad()
        self.actor_loss.backward()
        grad_norm = _compute_grad_norm(list(self.actor.parameters()))
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.max_grad_norm,
            )
        self.optimizer.step()
        loss_value = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": loss_value, "actor_grad_norm": grad_norm}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "apg_actor.pt"))

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "apg_actor.pt"), map_location=self.device),
        )


class APGStochastic(APG):
    """APG with a stochastic Gaussian actor and entropy regularisation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        lr: float = 3e-4,
        max_grad_norm: float | None = 1.0,
        l_rollout: int = 16,
        entropy_weight: float = 0.01,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.actor = StochasticActor(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.l_rollout = l_rollout
        self.entropy_weight = entropy_weight
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor(obs, deterministic=deterministic)

    def record_loss(self, loss: torch.Tensor, entropy: torch.Tensor) -> None:
        self.actor_loss = self.actor_loss + loss.mean()
        self.entropy_loss = self.entropy_loss - entropy.mean()

    def update_actor(self) -> dict[str, float]:
        actor_loss = self.actor_loss / self.l_rollout
        entropy_loss = self.entropy_loss / self.l_rollout
        total_loss = actor_loss + self.entropy_weight * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.max_grad_norm,
            ).item()
        else:
            grad_norm = _compute_grad_norm(list(self.actor.parameters()))
        self.optimizer.step()

        metrics = {
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "actor_grad_norm": grad_norm,
        }
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "apg_sto_actor.pt"))

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "apg_sto_actor.pt"), map_location=self.device),
        )
