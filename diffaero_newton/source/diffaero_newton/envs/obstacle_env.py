"""Obstacle avoidance environment with multi-modal sensor observations.

Extends DroneEnv to inject sensor-based observations (Camera/LiDAR/RelPos)
into the observation dict, enabling differentiable obstacle avoidance training.
"""

from typing import Any, Dict, Optional
import torch

from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.envs.sensors import create_sensor, CameraSensor, LiDARSensor, RelativePositionSensor
from diffaero_newton.configs.sensor_cfg import SensorCfg, RelposSensorCfg


class ObstacleAvoidanceEnv(DroneEnv):
    """Obstacle avoidance environment with configurable observation modes.

    Wraps DroneEnv and adds a sensor that produces obstacle observation tensors.
    Supported sensors: 'relpos', 'camera', 'lidar'.
    """

    def __init__(self, cfg, render_mode=None, sensor_cfg: Optional[SensorCfg] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Default to relpos if not provided
        if sensor_cfg is None:
            sensor_cfg = RelposSensorCfg(n_obstacles=self.obstacle_manager.cfg.num_obstacles)

        self.sensor = create_sensor(sensor_cfg, self.num_envs, self.device)
        self.sensor_cfg = sensor_cfg

        # Pre-compute sensor observation dimensions
        self.sensor_obs_dim = self.sensor.H * self.sensor.W

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Augment base observations with sensor data."""
        obs_dict = super()._get_observations()

        # Compute sensor observations
        pos = self.drone.p.detach().requires_grad_(True) if not self.drone.p.requires_grad else self.drone.p
        # Use identity quaternion (xyzw) for sensor frame = body frame
        quat_xyzw = torch.zeros(self.num_envs, 4, device=self.device)
        quat_xyzw[:, 3] = 1.0  # w=1 for identity

        sensor_obs = self.sensor(self.obstacle_manager, pos, quat_xyzw)

        # Flatten sensor output to 1D per environment
        sensor_flat = sensor_obs.reshape(self.num_envs, -1)

        # Add sensor data to observations
        obs_dict["policy"] = torch.cat([obs_dict["policy"], sensor_flat], dim=-1)

        return obs_dict
