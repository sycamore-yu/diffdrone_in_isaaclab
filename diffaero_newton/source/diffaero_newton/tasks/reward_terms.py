"""Reward and risk term computation for obstacle avoidance task.

This module provides differentiable loss functions for training
and detached reward functions for RL.
"""

from typing import Optional, Tuple
import torch

from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg, RiskWeights
from diffaero_newton.common.constants import (
    COLLISION_RADIUS,
    RISK_DISTANCE_WARNING,
    RISK_DISTANCE_CRITICAL,
    GOAL_REWARD_SCALE,
    COLLISION_PENALTY,
    TIME_PENALTY,
)


def compute_risk_loss(
    states: torch.Tensor,
    obstacles: torch.Tensor,
    risk_weights: Optional[RiskWeights] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute differentiable risk loss for training.

    This loss flows gradients through the trajectory, enabling
    optimization of action sequences that minimize collision risk.

    Args:
        states: State trajectory [horizon, num_envs, 13] or [num_envs, 13]
        obstacles: Obstacle positions [num_envs, num_obstacles, 4] = [pos, radius].
        risk_weights: Weights for different risk components.

    Returns:
        Tuple of (per_env_risk_loss [num_envs], loss_components_dict).
    """
    if risk_weights is None:
        risk_weights = RiskWeights()

    # Handle both [horizon, num_envs, dim] and [num_envs, dim]
    has_horizon = states.dim() == 3
    if not has_horizon:
        states = states.unsqueeze(0)

    num_envs = states.shape[1]
    device = states.device

    # Extract positions from states
    positions = states[:, :, :3]  # [horizon, num_envs, 3]

    # Extract obstacle info
    obs_positions = obstacles[:, :, :3]  # [num_envs, num_obs, 3]
    obs_radii = obstacles[:, :, 3]  # [num_envs, num_obs]

    distances = _compute_pairwise_distances(positions, obs_positions)  # [horizon, num_envs, num_obs]
    combined_radii = obs_radii.unsqueeze(0) + COLLISION_RADIUS
    clearance = distances - combined_radii

    # Continuous penetration depth penalty keeps gradients informative near collision.
    collision_loss = torch.relu(-clearance).sum(dim=2)  # [horizon, num_envs]

    # Smooth warning-zone penalty outside hard contact.
    warning_distance = max(float(RISK_DISTANCE_WARNING), 1.0e-6)
    proximity_loss = torch.exp(-clearance / warning_distance).mean(dim=2)  # [horizon, num_envs]

    total_risk = (
        risk_weights.collision * collision_loss
        + risk_weights.proximity * proximity_loss
    ).mean(dim=0)

    loss_components = {
        "collision": collision_loss.mean().detach().item(),
        "proximity": proximity_loss.mean().detach().item(),
        "total": total_risk.mean().detach().item(),
    }

    return total_risk, loss_components


def _compute_pairwise_distances(
    positions: torch.Tensor,
    obstacle_positions: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise distances between positions and obstacles.

    Args:
        positions: Positions [horizon, num_envs, 3].
        obstacle_positions: Obstacle positions [num_envs, num_obstacles, 3].

    Returns:
        Distances [horizon, num_envs, num_obstacles].
    """
    horizon, num_envs, _ = positions.shape
    num_obs = obstacle_positions.shape[1]

    # Reshape for broadcasting
    positions_expanded = positions.unsqueeze(2).expand(horizon, num_envs, num_obs, 3)
    obs_expanded = obstacle_positions.unsqueeze(0).expand(horizon, num_envs, num_obs, 3)

    # Compute distances
    distances = torch.norm(positions_expanded - obs_expanded, dim=-1)

    return distances


def compute_rewards(
    states: torch.Tensor,
    goal_position: Optional[torch.Tensor] = None,
    obstacles: Optional[torch.Tensor] = None,
    prev_states: Optional[torch.Tensor] = None,
    time_penalty: float = TIME_PENALTY,
    goal_scale: float = GOAL_REWARD_SCALE,
    collision_penalty: float = COLLISION_PENALTY,
) -> Tuple[torch.Tensor, dict]:
    """Compute detached rewards for RL training.

    Args:
        states: Current states [num_envs, 13].
        goal_position: Goal positions [num_envs, 3].
        obstacles: Obstacle data [num_envs, num_obstacles, 4].
        prev_states: Previous states for computing velocity [num_envs, 13].
        time_penalty: Penalty per timestep.
        goal_scale: Scaling for goal reward.
        collision_penalty: Penalty for collision.

    Returns:
        Tuple of (rewards [num_envs], reward_components_dict).
    """
    device = states.device
    num_envs = states.shape[0]

    # Initialize components
    goal_reward = torch.zeros(num_envs, device=device)
    collision_reward = torch.zeros(num_envs, device=device)
    velocity_penalty = torch.zeros(num_envs, device=device)
    smoothness_penalty = torch.zeros(num_envs, device=device)

    # Goal reward
    if goal_position is not None:
        positions = states[:, :3]
        goal_dist = torch.norm(goal_position - positions, dim=1)
        # Exponential reward based on distance
        goal_reward = torch.exp(-goal_dist / 0.5)

    # Collision penalty
    if obstacles is not None:
        positions = states[:, :3]
        obs_positions = obstacles[:, :, :3]
        obs_radii = obstacles[:, :, 3]

        distances = _compute_pairwise_distances(
            positions.unsqueeze(0),
            obs_positions
        ).squeeze(0)  # [num_envs, num_obstacles]

        combined_radii = obs_radii + COLLISION_RADIUS
        in_collision = distances < combined_radii
        collision_reward = -collision_penalty * in_collision.any(dim=1).float()

    # Velocity penalty
    velocities = states[:, 7:10]
    velocity_penalty = -0.01 * torch.sum(velocities ** 2, dim=1)

    # Action smoothness (if previous states provided)
    if prev_states is not None:
        current_vel = states[:, 7:10]
        prev_vel = prev_states[:, 7:10]
        acc = current_vel - prev_vel
        smoothness_penalty = -0.001 * torch.sum(acc ** 2, dim=1)

    # Total reward
    reward = (
        goal_scale * goal_reward
        + collision_reward
        + velocity_penalty
        + smoothness_penalty
        + time_penalty
    )

    reward_components = {
        "goal": goal_reward.mean().item(),
        "collision": collision_reward.mean().item(),
        "velocity": velocity_penalty.mean().item(),
        "smoothness": smoothness_penalty.mean().item(),
        "time": time_penalty,
        "total": reward.mean().item(),
    }

    return reward, reward_components


def compute_goal_progress(
    states: torch.Tensor,
    goal_position: torch.Tensor,
    prev_goal_distances: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute goal progress reward and new distances.

    Args:
        states: Current states [num_envs, 13].
        goal_position: Goal positions [num_envs, 3].
        prev_goal_distances: Previous distances to goal [num_envs].

    Returns:
        Tuple of (progress_reward, current_distances).
    """
    positions = states[:, :3]
    current_distances = torch.norm(goal_position - positions, dim=1)

    progress_reward = torch.zeros_like(current_distances)
    if prev_goal_distances is not None:
        # Reward for getting closer
        progress = prev_goal_distances - current_distances
        progress_reward = progress.clamp(min=0.0)

    return progress_reward, current_distances


def compute_orientation_reward(
    states: torch.Tensor,
    target_up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute reward for maintaining upright orientation.

    Args:
        states: States [num_envs, 13] (includes quaternion in indices 3:7).
        target_up: Target up direction [num_envs, 3]. Defaults to (0, 0, 1).

    Returns:
        Orientation reward [num_envs].
    """
    if target_up is None:
        target_up = torch.tensor([0.0, 0.0, 1.0], device=states.device)
        target_up = target_up.unsqueeze(0).expand(states.shape[0], 3)

    # Extract quaternion (w, x, y, z)
    quat = states[:, 3:7]  # [num_envs, 4]

    # Convert quaternion to rotation matrix and extract up vector
    # Simplified: just use w component as indicator of upright
    # Full implementation would use proper quaternion rotation
    up_reward = quat[:, 0]  # w component

    return up_reward
