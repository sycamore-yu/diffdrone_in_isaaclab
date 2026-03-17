"""Project-local DirectRLEnv shim for Newton-only headless tasks.

This module is intentionally narrow: it does not try to import or mirror the full
IsaacLab environment stack. The classes here exist because the project's Newton-only
tasks currently rely on a lightweight vectorized RL hook contract that can run without
the full IsaacLab/Kit environment layer.

It should not be treated as a drop-in replacement for IsaacLab's ``DirectRLEnv``.
Numerical kernels, rollout logic, and task costs remain in the project's dynamics and
environment implementations rather than in this shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "DirectRLEnv",
    "DirectRLEnvCfg",
    "InteractiveSceneCfg",
    "SimulationCfg",
    "NewtonCfg",
    "FeatherstoneSolverCfg",
    "configclass",
]


class DirectRLEnvCfg:
    """Stand-in for the small DirectRLEnvCfg subset used by Newton-only tasks."""


@dataclass
class InteractiveSceneCfg:
    """Minimal scene configuration used by the Newton-only env shim."""

    num_envs: int = 1
    env_spacing: float = 1.0
    replicate_physics: bool = True
    clone_in_fabric: bool = True


@dataclass
class FeatherstoneSolverCfg:
    """Placeholder Featherstone solver config."""


@dataclass
class NewtonCfg:
    """Placeholder Newton manager config."""

    solver_cfg: Any | None = None


@dataclass
class SimulationCfg:
    """Minimal simulation config used by the Newton-only env shim."""

    dt: float = 1.0 / 120.0
    render_interval: int = 4
    newton_cfg: Any | None = None


def configclass(cls: type) -> type:
    """Fallback decorator matching IsaacLab's configclass behavior."""

    return dataclass(eq=False)(cls)


class DirectRLEnv:
    """Minimal RL hook surface used by this project's Newton-only environments."""

    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.render_mode = render_mode
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = kwargs.get("device", getattr(cfg, "device", default_device))
        self.num_envs = cfg.num_envs
        self.decimation = cfg.decimation
        self.physics_dt = cfg.sim.dt
        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = getattr(
            cfg,
            "max_episode_length",
            int(cfg.episode_length_s / max(self.step_dt, 1.0e-6)),
        )
        self.episode_length_max = self.max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.single_observation_space = cfg.observation_space
        self.single_action_space = cfg.action_space
        self.observation_space = cfg.observation_space
        self.action_space = cfg.action_space
        self._setup_scene()

    def _setup_scene(self):
        """Mirror the IsaacLab hook with a no-op default."""

    def _reset_idx(self, env_ids):
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() > 0:
            self.episode_length_buf[env_ids_tensor] = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all environments and return the current observations."""

        if seed is not None:
            torch.manual_seed(seed)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._get_observations(), {}

    def step(self, action: torch.Tensor):
        """Run one vectorized environment step using the shim's gym-like test contract.

        Project trainers may wrap or reinterpret these values. This shim intentionally
        does not define the higher-level training tuple semantics used elsewhere in the
        repository.
        """

        self._pre_physics_step(action)
        for _ in range(self.decimation):
            self._apply_action()

        self.episode_length_buf += 1
        obs = self._get_observations()
        reward = self._get_rewards()
        terminated, truncated = self._get_dones()
        reset_mask = terminated | truncated
        if reset_mask.any():
            self._reset_idx(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
            obs = self._get_observations()
        return obs, reward, terminated, truncated, {}

    def close(self):
        """Mirror IsaacLab's close hook."""
