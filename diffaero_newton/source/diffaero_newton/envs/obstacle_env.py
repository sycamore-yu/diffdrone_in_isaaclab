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

        # Compute sensor observations via dynamics state API
        drone_state = self.drone.get_state()
        pos = drone_state["position"]
        # Keep gradients if already tracked, otherwise enable them
        if not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
        # Use identity quaternion (xyzw) for sensor frame = body frame
        quat_xyzw = torch.zeros(self.num_envs, 4, device=self.device)
        quat_xyzw[:, 3] = 1.0  # w=1 for identity

        sensor_obs = self.sensor(self.obstacle_manager, pos, quat_xyzw)

        # Flatten sensor output to 1D per environment
        sensor_flat = sensor_obs.reshape(self.num_envs, -1)

        # Add sensor data to observations
        obs_dict["policy"] = torch.cat([obs_dict["policy"], sensor_flat], dim=-1)

        return obs_dict
