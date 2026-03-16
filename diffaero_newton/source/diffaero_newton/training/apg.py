"""APG (Analytic Policy Gradient) and APG_stochastic training algorithms.

Migrated from DiffAero's APG.py to work natively with IsaacLab DirectRLEnv
and Newton differentiable physics.

Key differences from SHAC:
- APG: Pure deterministic actor, no critic. Loss flows directly through
  the differentiable environment.
- APG_stochastic: Stochastic actor with entropy regularisation.
  Still critic-free; the loss graph is maintained end-to-end.
"""

from typing import Dict, Optional, Tuple
import os
import torch
import torch.nn as nn
from torch.distributions import Normal

from diffaero_newton.common.constants import ACTION_DIM


# ---------------------------------------------------------------------------
# Actor networks
# ---------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    """Simple deterministic MLP actor."""

    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Sigmoid())  # actions ∈ [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class StochasticActor(nn.Module):
    """Gaussian stochastic actor with learnable log-std."""

    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM,
                 hidden_dims: list = None, log_std_init: float = -0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns (action, log_prob, entropy)."""
        h = self.backbone(obs)
        mean = self.mean_head(h)
        if deterministic:
            return torch.sigmoid(mean), torch.zeros(obs.shape[0], 1, device=obs.device), torch.zeros(obs.shape[0], 1, device=obs.device)
        std = self.log_std.clamp(-20, 2).exp()
        dist = Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return torch.sigmoid(sample), log_prob, entropy


# ---------------------------------------------------------------------------
# APG (deterministic)
# ---------------------------------------------------------------------------

class APG:
    """Analytic Policy Gradient – deterministic actor trained through the
    differentiable simulation graph.

    Compatible with any env that follows the IsaacLab DirectRLEnv contract:
        obs, state, loss_terms, reward, extras = env.step(action)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        lr: float = 3e-4,
        max_grad_norm: Optional[float] = 1.0,
        l_rollout: int = 16,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.actor = DeterministicActor(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.l_rollout = l_rollout

        # Accumulated loss over rolling horizon
        self.actor_loss = torch.zeros(1, device=self.device)

    # ---- inference ---------------------------------------------------------
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    # ---- training ----------------------------------------------------------
    def record_loss(self, loss: torch.Tensor):
        """Accumulate per-step loss into the rolling actor loss."""
        self.actor_loss = self.actor_loss + loss.mean()

    def update_actor(self) -> Dict[str, float]:
        """Back-propagate through the accumulated loss graph and step."""
        self.actor_loss = self.actor_loss / self.l_rollout
        self.optimizer.zero_grad()
        self.actor_loss.backward()
        grad_norm = sum(
            p.grad.data.norm().item() ** 2
            for p in self.actor.parameters() if p.grad is not None
        ) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        loss_val = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": loss_val, "actor_grad_norm": grad_norm}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "apg_actor.pt"))

    def load(self, path: str):
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "apg_actor.pt"), map_location=self.device))


# ---------------------------------------------------------------------------
# APG_stochastic
# ---------------------------------------------------------------------------

class APGStochastic(APG):
    """APG with a stochastic (Gaussian) actor and entropy regularisation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        lr: float = 3e-4,
        max_grad_norm: Optional[float] = 1.0,
        l_rollout: int = 16,
        entropy_weight: float = 0.01,
        device: str = "cuda",
    ):
        # Skip APG.__init__ to replace actor
        self.device = torch.device(device)
        self.actor = StochasticActor(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.l_rollout = l_rollout
        self.entropy_weight = entropy_weight

        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        action, log_prob, entropy = self.actor(obs, deterministic=deterministic)
        return action, log_prob, entropy

    def record_loss(self, loss: torch.Tensor, entropy: torch.Tensor):
        self.actor_loss = self.actor_loss + loss.mean()
        self.entropy_loss = self.entropy_loss - entropy.mean()

    def update_actor(self) -> Dict[str, float]:
        actor_loss = self.actor_loss / self.l_rollout
        entropy_loss = self.entropy_loss / self.l_rollout
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = 0.0
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.max_grad_norm).item()
        else:
            grad_norm = sum(
                p.grad.data.norm().item() ** 2
                for p in self.actor.parameters() if p.grad is not None
            ) ** 0.5
        self.optimizer.step()
        result = {
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "actor_grad_norm": grad_norm,
        }
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return result

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "apg_sto_actor.pt"))

    def load(self, path: str):
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "apg_sto_actor.pt"), map_location=self.device))
