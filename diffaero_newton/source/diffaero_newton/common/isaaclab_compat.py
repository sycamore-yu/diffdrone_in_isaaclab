"""Minimal IsaacLab compatibility layer for tests and headless execution.

This module prefers the real IsaacLab classes when they can be imported safely.
If IsaacLab bootstrapping fails (for example due to IsaacSim TLS/runtime issues),
it falls back to lightweight stand-ins that implement only the contracts used by
``diffaero_newton``. It also preserves the public ``launch_app`` helper used by
the unified training entrypoint.
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
    "launch_app",
]

try:
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
    from isaaclab.sim._impl.solvers_cfg import FeatherstoneSolverCfg
    from isaaclab.utils import configclass
except BaseException:
    class DirectRLEnvCfg:
        """Lightweight stand-in for IsaacLab's DirectRLEnvCfg."""

    @dataclass
    class InteractiveSceneCfg:
        """Minimal scene configuration used for config compatibility."""

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
        """Minimal simulation config used by the fallback environment base."""

        dt: float = 1.0 / 120.0
        render_interval: int = 4
        newton_cfg: Any | None = None

    def configclass(cls: type) -> type:
        """Fallback decorator matching IsaacLab's configclass behavior."""

        return dataclass(eq=False)(cls)

    class DirectRLEnv:
        """Small subset of IsaacLab's DirectRLEnv API used by this project."""

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

        def reset(self):
            """Reset all environments and return observations."""
            env_ids = list(range(self.num_envs))
            self._reset_idx(env_ids)
            if hasattr(self, "_get_observations"):
                return self._get_observations(), {}
            return None, {}

        def step(self, action):
            """Approximate IsaacLab's DirectRLEnv step contract for tests."""
            if hasattr(self, "_pre_physics_step"):
                self._pre_physics_step(action)
            if hasattr(self, "_apply_action"):
                self._apply_action()

            self.episode_length_buf += 1

            obs = self._get_observations() if hasattr(self, "_get_observations") else None
            reward = self._get_rewards() if hasattr(self, "_get_rewards") else torch.zeros(self.num_envs, device=self.device)
            if hasattr(self, "_get_dones"):
                terminated, truncated = self._get_dones()
            else:
                terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            reset_mask = terminated | truncated
            if reset_mask.any():
                self._reset_idx(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
                obs = self._get_observations() if hasattr(self, "_get_observations") else obs

            return obs, reward, terminated, truncated, {}

        def close(self):
            """Mirror the real environment API."""


class _NullApp:
    """Headless app stub used when IsaacLab's AppLauncher is unavailable."""

    def close(self):
        """Mirror the real app handle API."""


def launch_app():
    """Bootstrap IsaacLab's AppLauncher when available.

    The training entrypoint imports this helper unconditionally, so keep the
    symbol exported even when tests run without the full IsaacLab stack.
    """

    try:
        from isaaclab.app import AppLauncher
    except BaseException:
        return _NullApp()

    app_launcher = AppLauncher(headless=True)
    return app_launcher.app
