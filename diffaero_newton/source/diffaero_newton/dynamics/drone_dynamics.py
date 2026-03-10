"""Differentiable quadrotor dynamics using simple physics model.

This module provides a simplified drone dynamics model using basic torch operations.
"""

from dataclasses import dataclass
from typing import Optional
import torch

from diffaero_newton.common.constants import (
    GRAVITY,
    QUADROTOR_MASS,
    IXX,
    IYY,
    IZZ,
    ARM_LENGTH,
    COLLISION_RADIUS,
    DEFAULT_DT,
)


@dataclass
class DroneConfig:
    """Configuration for the drone dynamics."""

    num_envs: int = 1
    dt: float = DEFAULT_DT
    requires_grad: bool = False
    arm_length: float = ARM_LENGTH
    mass: float = QUADROTOR_MASS
    inertia: tuple[float, float, float] = (IXX, IYY, IZZ)


class Drone:
    """Differentiable quadrotor dynamics using simple physics."""

    def __init__(self, config: DroneConfig, device: str = "cpu"):
        """Initialize the drone dynamics."""
        self.config = config
        self.num_envs = config.num_envs
        self.device = device

        # State: [pos(3), quat(4), vel(3), omega(3)] = 13
        self._state = torch.zeros(self.num_envs, 13, device=device, requires_grad=config.requires_grad)
        self._state[:, 2] = 1.0  # z=1 hover
        self._state[:, 6] = 1.0  # w quaternion

        self._last_thrust = torch.zeros(self.num_envs, 4, device=device)

    @property
    def state(self) -> torch.Tensor:
        return self._state

    def reset_states(self, positions: Optional[torch.Tensor] = None, env_ids: Optional[torch.Tensor] = None):
        """Reset the drone to initial states."""
        if positions is None:
            positions = torch.zeros(self.num_envs, 3, device=self.device)
            positions[:, 2] = 1.0

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._state[env_ids, :3] = positions[env_ids]
        self._state[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._state[env_ids, 7:10] = 0.0
        self._state[env_ids, 10:13] = 0.0

    def apply_control(self, thrust_normalized: torch.Tensor):
        """Apply control inputs (normalized thrust per motor)."""
        self._last_thrust = thrust_normalized.to(self.device)

    def integrate(self, dt: Optional[float] = None):
        """Step the simulation forward."""
        dt = dt or self.config.dt

        pos = self._state[:, :3]
        vel = self._state[:, 7:10]

        # Total thrust
        total_thrust = self._last_thrust.sum(dim=1) * 20.0
        thrust_force = torch.zeros_like(vel)
        thrust_force[:, 2] = total_thrust

        # Gravity
        gravity_force = torch.zeros_like(vel)
        gravity_force[:, 2] = -GRAVITY * self.config.mass

        # Total force and acceleration
        total_force = thrust_force + gravity_force
        acc = total_force / self.config.mass

        # Update
        vel_new = vel + acc * dt
        pos_new = pos + vel_new * dt

        # Update state
        self._state[:, :3] = pos_new
        self._state[:, 7:10] = vel_new
        self._state[:, 10:13] = self._state[:, 10:13] * 0.95  # damping

    def get_state(self) -> dict[str, torch.Tensor]:
        """Get the current state as dict of tensors."""
        return {
            "position": self._state[:, :3].clone(),
            "orientation": self._state[:, 3:7].clone(),
            "velocity": self._state[:, 7:10].clone(),
            "omega": self._state[:, 10:13].clone(),
        }

    def get_flat_state(self) -> torch.Tensor:
        """Get the current state as flat tensor [num_envs, 13]."""
        return self._state.clone()


def create_drone(
    num_envs: int = 1,
    dt: float = DEFAULT_DT,
    requires_grad: bool = False,
    device: str = "cpu",
) -> Drone:
    """Create a differentiable drone dynamics instance."""
    config = DroneConfig(num_envs=num_envs, dt=dt, requires_grad=requires_grad)
    return Drone(config, device=device)
