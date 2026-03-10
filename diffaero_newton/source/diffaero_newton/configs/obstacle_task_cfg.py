"""Obstacle task configuration."""

from dataclasses import dataclass, field
from typing import Tuple, List

from diffaero_newton.common.constants import (
    MAX_OBSTACLES,
    COLLISION_RADIUS,
    OBSTACLE_COLLISION_RADIUS,
    RISK_DISTANCE_WARNING,
    RISK_DISTANCE_CRITICAL,
)


@dataclass
class ObstacleTaskCfg:
    """Configuration for the obstacle avoidance task.

    Attributes:
        num_obstacles: Number of obstacles in the environment.
        obstacle_bounds: Bounds for obstacle positions (min, max).
        obstacle_radius: Radius of spherical obstacles.
        collision_radius: Drone collision radius.
        risk_weights: Weights for different risk terms.
        spawn_strategy: How to spawn obstacles ('random', 'grid', 'fixed').
    """

    # Obstacle configuration
    num_obstacles: int = 5
    obstacle_bounds: Tuple[float, float, float, float, float, float] = (
        -5.0, -5.0, 0.0, 5.0, 5.0, 8.0  # x_min, y_min, z_min, x_max, y_max, z_max
    )
    obstacle_radius: float = OBSTACLE_COLLISION_RADIUS

    # Collision parameters
    collision_radius: float = COLLISION_RADIUS
    risk_distance_warning: float = RISK_DISTANCE_WARNING
    risk_distance_critical: float = RISK_DISTANCE_CRITICAL

    # Spawn configuration
    spawn_strategy: str = "random"  # 'random', 'grid', 'fixed'

    # Task rewards
    goal_reward_scale: float = 10.0
    collision_penalty: float = 100.0
    time_penalty: float = 0.1

    # Distance thresholds
    goal_threshold: float = 0.3  # Distance to consider goal reached
    max_distance: float = 20.0  # Max distance for normalization


@dataclass
class ObstacleConfig:
    """Configuration for a single obstacle."""

    position: Tuple[float, float, float]
    radius: float = OBSTACLE_COLLISION_RADIUS
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_static: bool = True


@dataclass
class RiskWeights:
    """Weights for different risk terms in the loss."""

    collision: float = 1.0
    proximity: float = 0.5
    heading: float = 0.1
    velocity: float = 0.1
