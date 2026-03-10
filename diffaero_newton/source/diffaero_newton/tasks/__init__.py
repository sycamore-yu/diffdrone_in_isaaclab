"""Task layer for obstacle avoidance."""

from diffaero_newton.tasks.obstacle_manager import ObstacleManager
from diffaero_newton.tasks.reward_terms import compute_risk_loss, compute_rewards
from diffaero_newton.tasks.observations import (
    build_state_observation,
    build_goal_observation,
    build_obstacle_observation,
)
