"""Racing environment configuration."""

from __future__ import annotations

from dataclasses import field

import numpy as np
from gymnasium.spaces import Box

from diffaero_newton.common.direct_rl_shim import configclass
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg


@configclass
class RacingEnvCfg(DroneEnvCfg):
    """Configuration for the racing environment with gate tracking."""

    num_observations: int = 10
    observation_space: Box = field(default_factory=lambda: Box(low=-np.inf, high=np.inf, shape=(10,)))

    gate_radius: float = 3.0
    gate_height: float = 2.0
    gate_size: float = 3.0

    vel_loss_weight: float = 1.0
    jerk_loss_weight: float = 0.005
    progress_loss_weight: float = 1.0
    pos_loss_weight: float = 0.25
    collision_loss_weight: float = 5.0
    oob_loss_weight: float = 5.0

    reward_constant: float = 1.0
    reward_pass: float = 10.0
    reward_collision: float = 5.0
    reward_oob: float = 5.0
    reward_progress: float = 1.0

    xy_bound: float = 5.0
    z_bound: float = 7.0

    use_vel_track: bool = False
    min_target_vel: float = 1.0
    max_target_vel: float = 4.0
