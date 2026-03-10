"""Rollout functions for differentiable quadrotor dynamics.

This module provides functions for computing single-step and multi-step
trajectories through the differentiable physics model.
"""

from typing import Optional

import torch

from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig
from diffaero_newton.common.constants import DEFAULT_DT, DEFAULT_ROLLOUT_HORIZON


def rollout_onestep(
    drone: Drone,
    state: torch.Tensor,
    control: torch.Tensor,
    dt: Optional[float] = None,
) -> torch.Tensor:
    """Compute a single-step state transition.

    Args:
        drone: The drone dynamics instance.
        state: Current state [num_envs, 13] = [pos(3), quat(4), vel(3), omega(3)].
        control: Control input [num_envs, 4] normalized thrusts [0, 1].
        dt: Timestep. If None, use drone config default.

    Returns:
        Next state [num_envs, 13].
    """
    dt = dt or drone.config.dt

    # Reset drone to given state
    positions = state[:, :3]
    drone.reset_states(positions)

    # Apply control
    drone.apply_control(control)

    # Integrate one step
    drone.integrate(dt)

    # Return new state
    return drone.get_flat_state()


def rollout_horizon(
    env_cfg: DroneConfig,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    dt: Optional[float] = None,
) -> torch.Tensor:
    """Compute a multi-step differentiable rollout.

    This function runs the drone dynamics forward for a horizon of steps,
    applying the given sequence of actions. The gradient flows through
    the entire trajectory, enabling optimization of action sequences.

    Args:
        env_cfg: Environment configuration (num_envs, dt, requires_grad).
        initial_states: Initial states [num_envs, 13].
        actions: Action sequence [num_envs, horizon, 4] normalized thrusts.
        dt: Timestep. If None, use config default.

    Returns:
        Final states [num_envs, 13] after rolling out the horizon.
    """
    num_envs = env_cfg.num_envs
    horizon = actions.shape[1]
    dt = dt or env_cfg.dt

    # Create drone with gradient tracking if needed
    drone = Drone(env_cfg)

    # Initialize state
    positions = initial_states[:, :3]
    drone.reset_states(positions)

    # Rollout through the horizon
    final_state = initial_states

    for t in range(horizon):
        # Get action for this step
        control = actions[:, t, :]  # [num_envs, 4]

        # Apply control and integrate
        drone.apply_control(control)
        drone.integrate(dt)

        # Get current state
        final_state = drone.get_flat_state()

    return final_state


def compute_rollout_loss(
    final_states: torch.Tensor,
    target_positions: torch.Tensor,
    collision_states: Optional[torch.Tensor] = None,
    collision_margin: float = 0.5,
) -> torch.Tensor:
    """Compute a simple rollout loss for optimization.

    This is a differentiable loss that can be used to optimize
    action sequences via gradient descent.

    Args:
        final_states: Final states from rollout [num_envs, 13].
        target_positions: Target positions [num_envs, 3].
        collision_states: States along trajectory for collision checking.
        collision_margin: Margin for collision penalty.

    Returns:
        Scalar loss tensor.
    """
    # Position cost
    final_pos = final_states[:, :3]
    position_loss = torch.sum((final_pos - target_positions) ** 2, dim=1).mean()

    # Collision penalty (if states provided)
    collision_loss = torch.tensor(0.0)
    if collision_states is not None:
        # Simple ground collision check
        min_altitude = collision_states[:, :, 2].min()
        if min_altitude < collision_margin:
            collision_loss = torch.relu(collision_margin - min_altitude) * 10.0

    return position_loss + collision_loss


class RolloutBuffer:
    """Buffer for storing rollout trajectories.

    This is used to store state-action trajectories for analysis
    or for computing returns/advantages in RL algorithms.
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the rollout buffer.

        Args:
            num_envs: Number of parallel environments.
            horizon: Number of steps in the rollout.
            device: Device for tensor storage.
        """
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device

        # Allocate storage
        self.states = torch.zeros(horizon + 1, num_envs, 13, device=device)
        self.actions = torch.zeros(horizon, num_envs, 4, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)

        self.ptr = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add a transition to the buffer.

        Args:
            state: Current state [num_envs, 13].
            action: Action taken [num_envs, 4].
            reward: Reward received [num_envs].
            done: Done flag [num_envs].
        """
        if self.ptr == 0:
            self.states[0] = state

        self.states[self.ptr + 1] = state  # Store next state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1

    def reset(self):
        """Reset the buffer for a new rollout."""
        self.ptr = 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.ptr >= self.horizon
