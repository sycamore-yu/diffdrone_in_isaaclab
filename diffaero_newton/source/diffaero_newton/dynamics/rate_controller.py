"""Body-rate control helpers for quadrotor dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RateControllerConfig:
    """Configuration for the DiffAero-style body-rate controller."""

    k_angvel: tuple[float, float, float] = (6.0, 6.0, 2.5)
    min_body_rates: tuple[float, float, float] = (-3.14, -3.14, -3.14)
    max_body_rates: tuple[float, float, float] = (3.14, 3.14, 3.14)
    min_normed_thrust: float = 0.0
    max_normed_thrust: float = 5.0
    thrust_ratio: float = 1.0
    torque_ratio: float = 1.0
    mass: float = 1.0
    gravity: float = 9.81
    compensate_gravity: bool = False
    gyro_cross_limit: float = 100.0


def quaternion_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return rotation matrices from `wxyz` quaternions."""

    quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quat.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


class RateController:
    """DiffAero-style body-rate controller operating on batched tensors."""

    def __init__(
        self,
        inertia: torch.Tensor,
        cfg: RateControllerConfig | None = None,
        *,
        device: torch.device,
    ) -> None:
        self.cfg = cfg or RateControllerConfig()
        self.device = device
        self.inertia = inertia.to(device=device, dtype=torch.float32)
        self.k_angvel = torch.tensor(self.cfg.k_angvel, device=device, dtype=torch.float32)
        self.min_body_rates = torch.tensor(self.cfg.min_body_rates, device=device, dtype=torch.float32)
        self.max_body_rates = torch.tensor(self.cfg.max_body_rates, device=device, dtype=torch.float32)
        self.body_rate_range = self.max_body_rates - self.min_body_rates

    def compute_torque(
        self,
        orientation_wxyz: torch.Tensor,
        angular_velocity_world: torch.Tensor,
        desired_angvel_body: torch.Tensor,
    ) -> torch.Tensor:
        """Return body torques from desired body rates."""

        rotmat_b2w = quaternion_to_matrix(orientation_wxyz)
        rotmat_w2b = rotmat_b2w.transpose(-1, -2)
        actual_angvel_body = torch.bmm(rotmat_w2b, angular_velocity_world.unsqueeze(-1)).squeeze(-1)
        angvel_err = desired_angvel_body - actual_angvel_body

        inertia_omega = torch.bmm(self.inertia, actual_angvel_body.unsqueeze(-1)).squeeze(-1)
        gyro_cross = torch.cross(actual_angvel_body, inertia_omega, dim=-1)
        gyro_scale = (gyro_cross.norm(dim=-1, keepdim=True) / self.cfg.gyro_cross_limit).clamp_min(1.0)
        gyro_cross = gyro_cross / gyro_scale.detach()

        angacc = self.cfg.torque_ratio * self.k_angvel * angvel_err
        return torch.bmm(self.inertia, angacc.unsqueeze(-1)).squeeze(-1) + gyro_cross

    def __call__(
        self,
        orientation_wxyz: torch.Tensor,
        angular_velocity_world: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map normalized `[collective, body_rate_xyz]` commands to body-frame wrench."""

        collective = self.cfg.min_normed_thrust + action[:, 0].clamp(0.0, 1.0) * (
            self.cfg.max_normed_thrust - self.cfg.min_normed_thrust
        )
        desired_angvel_body = self.min_body_rates + action[:, 1:].clamp(0.0, 1.0) * self.body_rate_range
        torque = self.compute_torque(orientation_wxyz, angular_velocity_world, desired_angvel_body)
        thrust = collective * self.cfg.thrust_ratio
        if self.cfg.compensate_gravity:
            thrust = thrust + 1.0
        thrust = thrust * self.cfg.gravity * self.cfg.mass

        body_force = torch.zeros(action.shape[0], 3, device=action.device, dtype=action.dtype)
        body_force[:, 2] = thrust
        return body_force, torque
