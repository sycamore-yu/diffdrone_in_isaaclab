"""Observation builders for the obstacle avoidance task.

This module provides functions to build different observation types
for the policy and critic networks.
"""

from typing import Optional, Tuple
import torch

from diffaero_newton.common.constants import (
    STATE_DIM,
    OBS_DIM_BASE,
    OBS_DIM_GOAL,
    OBS_DIM_OBSTACLE,
    MAX_OBSTACLES,
)


def build_state_observation(
    states: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Build base state observation.

    Args:
        states: Full state tensor [num_envs, 13] = [pos(3), quat(4), vel(3), omega(3)].
        normalize: Whether to normalize velocities.

    Returns:
        Observation tensor [num_envs, 13].
    """
    obs = states.clone()

    if normalize:
        # Normalize velocities to reasonable range
        obs[:, 7:10] = obs[:, 7:10] / 10.0  # velocity
        obs[:, 10:13] = obs[:, 10:13] / 10.0  # omega

    return obs


def build_goal_observation(
    states: torch.Tensor,
    goal_position: torch.Tensor,
    max_distance: float = 20.0,
) -> torch.Tensor:
    """Build goal-relative observation.

    Args:
        states: State tensor [num_envs, 13].
        goal_position: Goal positions [num_envs, 3].
        max_distance: Maximum distance for normalization.

    Returns:
        Goal-relative observation [num_envs, 3].
    """
    positions = states[:, :3]
    goal_rel = goal_position - positions

    # Normalize by max distance
    goal_rel = goal_rel / max_distance

    return goal_rel


def build_obstacle_observation(
    states: torch.Tensor,
    obstacles: torch.Tensor,
    max_obstacles: int = MAX_OBSTACLES,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build obstacle-aware observation.

    Returns the nearest obstacles in a fixed-size format.

    Args:
        states: State tensor [num_envs, 13].
        obstacles: Obstacle data [num_envs, num_obstacles, 4] = [pos(3), radius].
        max_obstacles: Maximum number of obstacles to include.
        normalize: Whether to normalize distances.

    Returns:
        Tuple of (obstacle_obs, mask).
        - obstacle_obs: [num_envs, max_obstacles * 4] = [pos(3), radius] per obstacle
        - mask: [num_envs, max_obstacles] indicating valid obstacles
    """
    num_envs = states.shape[0]
    device = states.device

    positions = states[:, :3].unsqueeze(1)  # [num_envs, 1, 3]
    obs_positions = obstacles[:, :, :3]  # [num_envs, num_obstacles, 3]
    obs_radii = obstacles[:, :, 3]  # [num_envs, num_obstacles]

    # Compute distances
    distances = torch.norm(positions - obs_positions, dim=2)  # [num_envs, num_obstacles]

    # Get nearest obstacles
    num_available = min(obstacles.shape[1], max_obstacles)
    nearest_indices = distances.argsort(dim=1)[:, :num_available]  # [num_envs, num_obs]

    # Gather obstacle data
    batch_indices = torch.arange(num_envs, device=device).unsqueeze(1).expand(-1, num_available)
    nearest_obs = obs_positions[batch_indices, nearest_indices]  # [num_envs, num_obs, 3]
    nearest_radii = obs_radii[batch_indices, nearest_indices]  # [num_envs, num_obs]

    # Compute relative positions
    rel_positions = nearest_obs - positions.squeeze(1).unsqueeze(1).expand(-1, num_available, -1)

    if normalize:
        rel_positions = rel_positions / 10.0  # Normalize by typical distance
        nearest_radii = nearest_radii / 2.0  # Normalize by typical radius

    # Concatenate position and radius
    obstacle_obs = torch.cat([rel_positions, nearest_radii.unsqueeze(2)], dim=2)  # [num_envs, num_obs, 4]

    # Flatten
    obstacle_obs = obstacle_obs.reshape(num_envs, num_available * 4)

    # Create mask (1 for valid, 0 for padded)
    mask = torch.ones(num_envs, num_available, device=device)

    return obstacle_obs, mask


def build_full_observation(
    states: torch.Tensor,
    goal_position: torch.Tensor,
    obstacles: Optional[torch.Tensor] = None,
    prev_action: Optional[torch.Tensor] = None,
    max_obstacles: int = 5,
) -> dict:
    """Build the full observation dictionary.

    Args:
        states: State tensor [num_envs, 13].
        goal_position: Goal positions [num_envs, 3].
        obstacles: Obstacle data [num_envs, num_obstacles, 4].
        prev_action: Previous action [num_envs, 4].
        max_obstacles: Maximum obstacles to include.

    Returns:
        Dictionary with observation tensors.
    """
    # Base state observation
    state_obs = build_state_observation(states)

    # Goal observation
    goal_obs = build_goal_observation(states, goal_position)

    # Obstacle observation
    if obstacles is not None:
        obstacle_obs, obstacle_mask = build_obstacle_observation(
            states, obstacles, max_obstacles=max_obstacles
        )
    else:
        obstacle_obs = torch.zeros(states.shape[0], max_obstacles * 4, device=states.device)
        obstacle_mask = torch.zeros(states.shape[0], max_obstacles, device=states.device)

    # Previous action
    if prev_action is None:
        prev_action = torch.zeros(states.shape[0], 4, device=states.device)

    # Concatenate all
    policy_obs = torch.cat([
        state_obs,
        goal_obs,
        prev_action,
    ], dim=1)

    return {
        "policy": policy_obs,
        "obstacles": obstacle_obs,
        "obstacle_mask": obstacle_mask,
    }


def build_critic_observation(
    states: torch.Tensor,
    goal_position: torch.Tensor,
    obstacles: Optional[torch.Tensor] = None,
    max_obstacles: int = 5,
) -> torch.Tensor:
    """Build critic (state) observation for asymmetric actor-critic.

    The critic gets more information than the policy.

    Args:
        states: State tensor [num_envs, 13].
        goal_position: Goal positions [num_envs, 3].
        obstacles: Obstacle data [num_envs, num_obstacles, 4].
        max_obstacles: Maximum obstacles to include.

    Returns:
        Full state observation for critic.
    """
    # Build full observation
    full_obs = build_full_observation(
        states, goal_position, obstacles, max_obstacles=max_obstacles
    )

    # Concatenate all for critic
    critic_obs = torch.cat([
        full_obs["policy"],
        full_obs["obstacles"],
    ], dim=1)

    return critic_obs


def normalize_observation(
    obs: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize observation using running statistics.

    Args:
        obs: Observation tensor.
        mean: Mean for normalization. If None, compute from obs.
        std: Std for normalization. If None, compute from obs.

    Returns:
        Normalized observation.
    """
    if mean is None:
        mean = obs.mean(dim=0, keepdim=True)
    if std is None:
        std = obs.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-6)

    return (obs - mean) / std
