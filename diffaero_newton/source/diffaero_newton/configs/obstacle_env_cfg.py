"""Obstacle-avoidance environment configuration with explicit sensor contract."""

from __future__ import annotations

from dataclasses import field

import numpy as np
from gymnasium.spaces import Box

from diffaero_newton.common.direct_rl_shim import configclass
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.configs.sensor_cfg import RelposSensorCfg, SensorCfg, sensor_observation_dim


@configclass
class ObstacleAvoidanceEnvCfg(DroneEnvCfg):
    """Configuration for obstacle avoidance with explicit sensor observations."""

    sensor_cfg: SensorCfg = field(default_factory=lambda: RelposSensorCfg(n_obstacles=5))

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if isinstance(self.sensor_cfg, RelposSensorCfg):
            self.sensor_cfg.n_obstacles = self.num_obstacles
        self.num_states = 21
        self.num_observations = 21 + sensor_observation_dim(self.sensor_cfg)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
        )
