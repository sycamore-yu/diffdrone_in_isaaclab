"""Obstacle avoidance environment with multi-modal sensor observations."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from diffaero_newton.configs.obstacle_env_cfg import ObstacleAvoidanceEnvCfg
from diffaero_newton.configs.sensor_cfg import SensorCfg
from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.envs.sensors import create_sensor


class ObstacleAvoidanceEnv(DroneEnv):
    """Obstacle avoidance environment with configurable observation modes.

    Wraps DroneEnv and adds a sensor that produces obstacle observation tensors.
    Supported sensors: 'relpos', 'camera', 'lidar'.
    """

    cfg: ObstacleAvoidanceEnvCfg

    def __init__(
        self,
        cfg: ObstacleAvoidanceEnvCfg,
        render_mode: str | None = None,
        sensor_cfg: Optional[SensorCfg] = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        if sensor_cfg is None:
            sensor_cfg = cfg.sensor_cfg

        self.sensor = create_sensor(sensor_cfg, self.num_envs, self.device)
        self.sensor_cfg = sensor_cfg
        self.sensor_obs_dim = self.sensor.H * self.sensor.W

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Augment base observations with sensor data."""
        obs_dict = super()._get_observations()
        base_state = obs_dict["policy"]

        # Compute sensor observations via dynamics state API
        drone_state = self.drone.get_state()
        pos = drone_state["position"]
        # Keep gradients if already tracked, otherwise enable them
        if not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
        quat_xyzw = drone_state["orientation"].roll(-1, dims=-1)

        sensor_obs = self.sensor(self.obstacle_manager, pos, quat_xyzw)

        # Flatten sensor output to 1D per environment
        sensor_flat = sensor_obs.reshape(self.num_envs, -1)

        # Expose split state/perception keys for world-model paths while keeping
        # the existing policy tensor contract for non-world trainers.
        obs_dict["state"] = base_state
        obs_dict["perception"] = sensor_obs
        obs_dict["policy"] = torch.cat([base_state, sensor_flat], dim=-1)

        return obs_dict
