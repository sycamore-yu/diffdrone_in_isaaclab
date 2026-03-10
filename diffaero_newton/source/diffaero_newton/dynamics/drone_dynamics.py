"""Differentiable quadrotor dynamics using full quadrotor physics model.

This module provides a differentiable drone dynamics model with:
- Full quaternion-based attitude dynamics
- Angular velocity integration with inertia
- Control allocation (motor inputs -> thrust + torque)
- Euler/RK4 integration options
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
    solver_type: str = "euler"  # "euler" or "rk4"
    n_substeps: int = 1


class Drone:
    """Differentiable quadrotor dynamics with full physics."""

    def __init__(self, config: DroneConfig, device: str = "cpu"):
        """Initialize the drone dynamics."""
        self.config = config
        self.num_envs = config.num_envs
        self.device = device

        # State: [pos(3), quat(4), vel(3), omega(3)] = 13
        self._state = torch.zeros(
            self.num_envs, 13, device=device, requires_grad=config.requires_grad
        )
        self._state[:, 2] = 1.0  # z=1 hover
        # Initialize quaternion to identity [w=1, x=0, y=0, z=0]
        self._state[:, 3] = 1.0  # w component
        self._state[:, 4:7] = 0.0  # x, y, z components

        # Control input (4 motor thrusts)
        self._last_thrust = torch.zeros(self.num_envs, 4, device=device)

        # Physical constants
        self.mass = config.mass
        self.arm_length = config.arm_length
        self.J = torch.tensor(
            [config.inertia[0], config.inertia[1], config.inertia[2]],
            device=device
        )
        self.J_inv = torch.tensor(
            [1.0/config.inertia[0], 1.0/config.inertia[1], 1.0/config.inertia[2]],
            device=device
        )

        # Torque constant (simplified - proportional to thrust)
        self.ct = 0.01  # Torque coefficient

        # Control allocation matrix: maps 4 motors to [roll_torque, pitch_torque, yaw_torque, thrust]
        # Standard quadrotor X configuration
        self._tau_thrust_matrix = self._build_tau_thrust_matrix()

    def _build_tau_thrust_matrix(self) -> torch.Tensor:
        """Build control allocation matrix.
        
        Maps motor thrusts to [tau_x, tau_y, tau_z, total_thrust]
        """
        d = self.arm_length / (2**0.5)  # arm length / sqrt(2)
        
        # X-configuration:
        # Motor layout (front-right, front-left, rear-left, rear-right)
        # tau_x:  d*(T1 + T4 - T2 - T3) / sqrt(2)
        # tau_y:  d*(-T1 - T4 + T2 + T3) / sqrt(2)
        # tau_z:  (T1 + T3 - T2 - T4) * ct
        # thrust: T1 + T2 + T3 + T4
        
        matrix = torch.zeros(4, 4, device=self.device)
        matrix[0, 0] = d   # T1
        matrix[0, 1] = -d  # T2
        matrix[0, 2] = -d # T3
        matrix[0, 3] = d   # T4
        
        matrix[1, 0] = -d  # T1
        matrix[1, 1] = d   # T2
        matrix[1, 2] = d   # T3
        matrix[1, 3] = -d  # T4
        
        matrix[2, 0] = self.ct  # T1 (yaw)
        matrix[2, 1] = -self.ct # T2
        matrix[2, 2] = self.ct  # T3
        matrix[2, 3] = -self.ct # T4
        
        matrix[3, :] = 1.0  # All contribute to thrust
        
        return matrix

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
            reset_positions = positions
        else:
            reset_positions = positions

        self._state[env_ids, :3] = reset_positions
        # Reset quaternion to identity (w=1, x=y=z=0)
        self._state[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # Reset velocities
        self._state[env_ids, 7:10] = 0.0
        self._state[env_ids, 10:13] = 0.0

    def apply_control(self, thrust_normalized: torch.Tensor):
        """Apply control inputs (normalized thrust per motor [0,1])."""
        # Scale normalized thrust [0,1] to actual thrust
        max_thrust = 20.0  # Maximum thrust per motor
        self._last_thrust = thrust_normalized * max_thrust

    def integrate(self, dt: Optional[float] = None):
        """Step the simulation forward."""
        dt = dt or self.config.dt

        # Apply substeps for stability
        substeps = self.config.n_substeps
        sub_dt = dt / substeps

        for _ in range(substeps):
            if self.config.solver_type == "rk4":
                self._rk4_step(sub_dt)
            else:
                self._euler_step(sub_dt)

    def _euler_step(self, dt: float):
        """Euler integration step."""
        # Unpack state
        pos = self._state[:, :3]
        quat = self._state[:, 3:7]  # [w, x, y, z]
        vel = self._state[:, 7:10]
        omega = self._state[:, 10:13]

        # Compute control outputs: [num_envs, 4] @ [4, 4].T -> [num_envs, 4]
        # Each row of _tau_thrust_matrix applies to corresponding motor config
        # Matrix layout: rows are [roll_torque, pitch_torque, yaw_torque, thrust] contributions
        thrust_vec = self._last_thrust  # [num_envs, 4]
        
        # Build proper control matrix for batched matmul
        # Each motor contributes to all 4 outputs
        tau_x = thrust_vec[:, 0] * self._tau_thrust_matrix[0, 0] + thrust_vec[:, 1] * self._tau_thrust_matrix[0, 1] + thrust_vec[:, 2] * self._tau_thrust_matrix[0, 2] + thrust_vec[:, 3] * self._tau_thrust_matrix[0, 3]
        tau_y = thrust_vec[:, 0] * self._tau_thrust_matrix[1, 0] + thrust_vec[:, 1] * self._tau_thrust_matrix[1, 1] + thrust_vec[:, 2] * self._tau_thrust_matrix[1, 2] + thrust_vec[:, 3] * self._tau_thrust_matrix[1, 3]
        tau_z = thrust_vec[:, 0] * self._tau_thrust_matrix[2, 0] + thrust_vec[:, 1] * self._tau_thrust_matrix[2, 1] + thrust_vec[:, 2] * self._tau_thrust_matrix[2, 2] + thrust_vec[:, 3] * self._tau_thrust_matrix[2, 3]
        total_thrust = thrust_vec[:, 0] * self._tau_thrust_matrix[3, 0] + thrust_vec[:, 1] * self._tau_thrust_matrix[3, 1] + thrust_vec[:, 2] * self._tau_thrust_matrix[3, 2] + thrust_vec[:, 3] * self._tau_thrust_matrix[3, 3]
        
        tau = torch.stack([tau_x, tau_y, tau_z], dim=1)  # [num_envs, 3]
        thrust = total_thrust  # [num_envs]

        # Compute acceleration
        # Thrust direction in world frame (quaternion rotation of z-axis)
        thrust_dir = self._quat_rotate(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        thrust_acc = thrust_dir * (thrust / self.mass).unsqueeze(-1)
        
        # Gravity
        gravity_acc = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device)
        
        # Total linear acceleration
        acc = thrust_acc + gravity_acc

        # Angular acceleration (tau = J * alpha)
        omega_cross_J_omega = torch.cross(omega, (self.J * omega), dim=-1)
        alpha = self.J_inv * (tau - omega_cross_J_omega)

        # Update state
        vel_new = vel + acc * dt
        pos_new = pos + vel_new * dt
        
        omega_new = omega + alpha * dt
        quat_new = self._quat_integrate(quat, omega_new, dt)
        
        # Normalize quaternion
        quat_new = quat_new / torch.norm(quat_new, dim=-1, keepdim=True)

        # Store
        self._state[:, :3] = pos_new
        self._state[:, 3:7] = quat_new
        self._state[:, 7:10] = vel_new
        self._state[:, 10:13] = omega_new

    def _rk4_step(self, dt: float):
        """Runge-Kutta 4 integration step."""
        # RK4 for quaternion integration
        quat = self._state[:, 3:7]
        omega = self._state[:, 10:13]
        
        # Compute k1
        k1_quat = self._quat_derivative(quat, omega)
        k1_vel = self._compute_acceleration(quat, self._last_thrust)
        k1_omega = self._compute_angular_acceleration(quat, omega, self._last_thrust)
        
        # k2
        quat2 = quat + 0.5 * dt * k1_quat
        quat2 = quat2 / torch.norm(quat2, dim=-1, keepdim=True)
        omega2 = omega + 0.5 * dt * k1_omega
        k2_quat = self._quat_derivative(quat2, omega2)
        k2_vel = self._compute_acceleration(quat2, self._last_thrust)
        k2_omega = self._compute_angular_acceleration(quat2, omega2, self._last_thrust)
        
        # k3
        quat3 = quat + 0.5 * dt * k2_quat
        quat3 = quat3 / torch.norm(quat3, dim=-1, keepdim=True)
        omega3 = omega + 0.5 * dt * k2_omega
        k3_quat = self._quat_derivative(quat3, omega3)
        k3_vel = self._compute_acceleration(quat3, self._last_thrust)
        k3_omega = self._compute_angular_acceleration(quat3, omega3, self._last_thrust)
        
        # k4
        quat4 = quat + dt * k3_quat
        quat4 = quat4 / torch.norm(quat4, dim=-1, keepdim=True)
        omega4 = omega + dt * k3_omega
        k4_quat = self._quat_derivative(quat4, omega4)
        k4_vel = self._compute_acceleration(quat4, self._last_thrust)
        k4_omega = self._compute_angular_acceleration(quat4, omega4, self._last_thrust)
        
        # Update state
        pos = self._state[:, :3]
        vel = self._state[:, 7:10]
        
        # Integrate velocity to get position (not from acceleration)
        avg_vel = (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel) / 6.0
        new_vel = vel + avg_vel * dt
        new_pos = pos + new_vel * dt
        
        new_quat = quat + (dt / 6.0) * (k1_quat + 2*k2_quat + 2*k3_quat + k4_quat)
        new_quat = new_quat / torch.norm(new_quat, dim=-1, keepdim=True)
        
        new_omega = omega + (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        
        self._state[:, :3] = new_pos
        self._state[:, 3:7] = new_quat
        self._state[:, 7:10] = new_vel
        self._state[:, 10:13] = new_omega

    def _quat_derivative(self, quat: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Compute quaternion time derivative."""
        # q_dot = 0.5 * q * omega_quat
        # omega_quat = [0, omega_x, omega_y, omega_z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
        
        q_dot = torch.zeros_like(quat)
        q_dot[:, 0] = 0.5 * (-x*wx - y*wy - z*wz)
        q_dot[:, 1] = 0.5 * (w*wx + y*wz - z*wy)
        q_dot[:, 2] = 0.5 * (w*wy - x*wz + z*wx)
        q_dot[:, 3] = 0.5 * (w*wz + x*wy - y*wx)
        
        return q_dot

    def _quat_integrate(self, quat: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """Integrate quaternion given angular velocity."""
        return quat + dt * self._quat_derivative(quat, omega)

    def _quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector by quaternion."""
        # q * v * q^-1
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Convert vec to quaternion form
        vx, vy, vz = vec[0], vec[1], vec[2]
        
        # Rotation formula
        tx = (y*vz - z*vy) * 2
        ty = (z*vx - x*vz) * 2
        tz = (x*vy - y*vx) * 2
        
        result = torch.zeros_like(vec).unsqueeze(0).repeat(quat.shape[0], 1) if vec.dim() == 1 else torch.zeros(quat.shape[0], 3, device=self.device)
        
        if vec.dim() == 1:
            result[:, 0] = vx + w*tx + (y*tz - z*ty)
            result[:, 1] = vy + w*ty + (z*tx - x*tz)
            result[:, 2] = vz + w*tz + (x*ty - y*tx)
        else:
            # Batch vector
            tx = (y*vz - z*vy) * 2
            ty = (z*vx - x*vz) * 2
            tz = (x*vy - y*vx) * 2
            result[:, 0] = vx + w*tx + (y*tz - z*ty)
            result[:, 1] = vy + w*ty + (z*tx - x*tz)
            result[:, 2] = vz + w*tz + (x*ty - y*tx)
        
        return result

    def _compute_acceleration(self, quat: torch.Tensor, thrust: torch.Tensor) -> torch.Tensor:
        """Compute linear acceleration."""
        # Total thrust
        total_thrust = thrust.sum(dim=-1)
        
        # Thrust direction (world z-axis rotated by quaternion)
        thrust_dir = self._quat_rotate(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        
        # Thrust acceleration
        thrust_acc = thrust_dir * (total_thrust / self.mass).unsqueeze(-1)
        
        # Gravity
        gravity = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device)
        
        return thrust_acc + gravity

    def _compute_angular_acceleration(self, quat: torch.Tensor, omega: torch.Tensor, thrust: torch.Tensor) -> torch.Tensor:
        """Compute angular acceleration."""
        # Compute torques from control allocation
        controls = self._tau_thrust_matrix @ thrust.T  # [4, num_envs]
        tau = controls[:3].T  # [num_envs, 3]
        
        # Coriolis effect: omega_cross_J_omega
        J_omega = self.J * omega
        omega_cross_J_omega = torch.cross(omega, J_omega, dim=-1)
        
        # Angular acceleration: J_inv * (tau - omega x J*omega)
        alpha = self.J_inv * (tau - omega_cross_J_omega)
        
        return alpha

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
