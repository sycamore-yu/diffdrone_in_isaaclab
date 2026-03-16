"""Racing environment configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RacingEnvCfg:
    """Configuration for the racing environment with gate tracking."""

    # Gate layout
    gate_radius: float = 3.0       # radius of figure-8 pattern
    gate_height: float = 2.0       # default altitude of gates
    gate_size: float = 3.0         # gate opening diameter for pass check

    # Loss weights (PointMass dynamics)
    vel_loss_weight: float = 1.0
    jerk_loss_weight: float = 0.005
    progress_loss_weight: float = 1.0
    pos_loss_weight: float = 0.0

    # Reward weights
    reward_constant: float = 1.0
    reward_pass: float = 10.0
    reward_collision: float = 5.0
    reward_oob: float = 5.0
    reward_progress: float = 1.0

    # Bounds
    xy_bound: float = 5.0
    z_bound: float = 7.0

    # Velocity tracking
    use_vel_track: bool = False
    min_target_vel: float = 1.0
    max_target_vel: float = 4.0
