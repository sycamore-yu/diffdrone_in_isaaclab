"""Minimal IsaacLab compatibility layer for tests and headless execution.

This module prefers the real IsaacLab classes when they can be imported safely.
If IsaacLab bootstrapping fails (for example due to IsaacSim TLS/runtime issues),
it falls back to lightweight stand-ins that implement only the contracts used by
``diffaero_newton``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

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
