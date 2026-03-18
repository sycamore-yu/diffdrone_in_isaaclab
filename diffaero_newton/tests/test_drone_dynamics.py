"""Regression tests for quadrotor control and differentiable dynamics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from diffaero_newton.dynamics.drone_dynamics import (
    Drone,
    DroneConfig,
    compute_linear_drag_force,
    motor_thrust_to_body_wrench,
)
from diffaero_newton.dynamics.rate_controller import RateController, RateControllerConfig, quaternion_to_matrix


def _quat_rotate(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    rotmat = quaternion_to_matrix(quat_wxyz)
    return torch.bmm(rotmat, vec.unsqueeze(-1)).squeeze(-1)


def _quat_rotate_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    rotmat = quaternion_to_matrix(quat_wxyz)
    return torch.bmm(rotmat.transpose(-1, -2), vec.unsqueeze(-1)).squeeze(-1)


def _reference_rate_torque(
    orientation_wxyz: torch.Tensor,
    angular_velocity_world: torch.Tensor,
    desired_angvel_body: torch.Tensor,
    inertia: torch.Tensor,
    k_angvel: torch.Tensor,
    *,
    torque_ratio: float,
    gyro_cross_limit: float,
) -> torch.Tensor:
    rotmat_b2w = quaternion_to_matrix(orientation_wxyz)
    actual_angvel_body = torch.bmm(rotmat_b2w.transpose(-1, -2), angular_velocity_world.unsqueeze(-1)).squeeze(-1)
    angvel_err = desired_angvel_body - actual_angvel_body
    inertia_omega = torch.bmm(inertia, actual_angvel_body.unsqueeze(-1)).squeeze(-1)
    gyro_cross = torch.cross(actual_angvel_body, inertia_omega, dim=-1)
    gyro_cross = gyro_cross / (gyro_cross.norm(dim=-1, keepdim=True) / gyro_cross_limit).clamp_min(1.0)
    angacc = torque_ratio * k_angvel * angvel_err
    return torch.bmm(inertia, angacc.unsqueeze(-1)).squeeze(-1) + gyro_cross


@pytest.mark.cpu_smoke
def test_quaternion_to_matrix_matches_identity_and_half_turn() -> None:
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotmats = quaternion_to_matrix(quats)

    assert torch.allclose(rotmats[0], torch.eye(3), atol=1.0e-6)
    assert torch.allclose(rotmats[1], torch.diag(torch.tensor([-1.0, -1.0, 1.0])), atol=1.0e-6)


@pytest.mark.cpu_smoke
def test_rate_controller_matches_reference_torque_equation() -> None:
    inertia = torch.diag(torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32)).unsqueeze(0)
    min_body_rates = torch.tensor([-3.0, -2.0, -1.0], dtype=torch.float32)
    max_body_rates = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    controller = RateController(
        inertia,
        RateControllerConfig(
            k_angvel=(5.0, 4.0, 3.0),
            min_body_rates=tuple(min_body_rates.tolist()),
            max_body_rates=tuple(max_body_rates.tolist()),
            max_normed_thrust=5.0,
            thrust_ratio=0.8,
            torque_ratio=0.6,
            mass=1.0,
            gravity=9.81,
            gyro_cross_limit=100.0,
        ),
        device=torch.device("cpu"),
    )

    orientation = torch.tensor([[0.9238795, 0.0, 0.3826834, 0.0]], dtype=torch.float32)
    omega_world = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    action = torch.tensor([[0.6, 0.75, 0.25, 0.5]], dtype=torch.float32)
    desired_angvel_body = min_body_rates.unsqueeze(0) + action[:, 1:] * (max_body_rates - min_body_rates).unsqueeze(0)

    body_force, torque = controller(orientation, omega_world, action)
    reference_torque = _reference_rate_torque(
        orientation,
        omega_world,
        desired_angvel_body,
        inertia,
        torch.tensor([5.0, 4.0, 3.0]),
        torque_ratio=0.6,
        gyro_cross_limit=100.0,
    )

    assert body_force[0, 2].item() == pytest.approx(0.6 * 5.0 * 0.8 * 9.81)
    assert torch.allclose(torque, reference_torque, atol=1.0e-6)


@pytest.mark.cpu_smoke
def test_linear_drag_matches_diffaero_body_frame_formula() -> None:
    orientation = torch.tensor([[0.9238795, 0.0, 0.3826834, 0.0]], dtype=torch.float32)
    velocity_world = torch.tensor([[3.0, -1.5, 2.0]], dtype=torch.float32)

    drag_force = compute_linear_drag_force(
        orientation,
        velocity_world,
        drag_coeff_xy=0.15,
        drag_coeff_z=0.3,
    )

    vel_body = _quat_rotate_inverse(orientation, velocity_world)
    expected_drag = -_quat_rotate(
        orientation,
        torch.tensor([0.15, 0.15, 0.3]).unsqueeze(0) * vel_body,
    )

    assert torch.allclose(drag_force, expected_drag, atol=1.0e-6)


@pytest.mark.cpu_smoke
def test_motor_thrust_allocation_matches_expected_body_wrench() -> None:
    thrusts = torch.tensor([[8.0, 6.0, 4.0, 10.0]], dtype=torch.float32)
    body_force, body_torque = motor_thrust_to_body_wrench(thrusts, arm_length=0.2, torque_coeff=0.01)

    arm = 0.2 / (2.0 ** 0.5)
    expected_force = torch.tensor([[0.0, 0.0, 28.0]], dtype=torch.float32)
    expected_torque = torch.tensor(
        [[arm * 8.0, arm * -8.0, -0.04]],
        dtype=torch.float32,
    )

    assert torch.allclose(body_force, expected_force, atol=1.0e-6)
    assert torch.allclose(body_torque, expected_torque, atol=1.0e-6)


@pytest.mark.gpu_smoke
def test_drone_motor_thrust_gradients_flow() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    drone = Drone(
        DroneConfig(num_envs=2, dt=0.01, requires_grad=True, n_substeps=2),
        device=device,
    )
    actions = torch.full((2, 4), 0.55, device=device, requires_grad=True)
    drone.apply_control(actions)
    drone.integrate()

    state = drone.get_flat_state()
    loss = state[:, 2].sum() + 0.1 * state[:, 10:13].pow(2).sum()
    loss.backward()

    assert actions.grad is not None
    assert torch.any(actions.grad.abs() > 0.0)


@pytest.mark.gpu_smoke
def test_drone_body_rate_control_changes_orientation_and_backprops() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    drone = Drone(
        DroneConfig(
            num_envs=1,
            dt=0.01,
            requires_grad=True,
            control_mode="body_rate",
            max_body_rates=(3.14, 3.14, 3.14),
            drag_coeff_xy=0.05,
            drag_coeff_z=0.1,
            n_substeps=2,
        ),
        device=device,
    )
    actions = torch.tensor([[0.7, 1.0, 0.0, 0.75]], device=device, requires_grad=True)
    initial_orientation = drone.get_state()["orientation"].detach().clone()

    drone.apply_control(actions)
    for _ in range(10):
        drone.integrate()

    final_orientation = drone.get_state()["orientation"]
    loss = final_orientation[:, 1:].pow(2).sum() + drone.get_state()["omega"].pow(2).sum()
    loss.backward()

    assert not torch.allclose(initial_orientation, final_orientation.detach(), atol=1.0e-4)
    assert actions.grad is not None
    assert torch.any(actions.grad.abs() > 0.0)


@pytest.mark.cpu_smoke
def test_drone_body_rate_config_applies_diffaero_scaling_semantics() -> None:
    drone = Drone(
        DroneConfig(
            num_envs=1,
            dt=0.01,
            control_mode="body_rate",
            mass=1.0,
            gravity=9.81,
            k_angvel=(5.0, 4.0, 3.0),
            min_body_rates=(-3.0, -2.0, -1.0),
            max_body_rates=(3.0, 2.0, 1.0),
            min_normed_thrust=0.0,
            max_normed_thrust=5.0,
            thrust_ratio=0.8,
            torque_ratio=0.6,
        ),
        device="cpu",
    )
    state = drone.get_flat_state().clone()
    action = torch.tensor([[0.6, 0.75, 0.25, 0.5]], dtype=torch.float32)

    body_force, body_torque = drone._resolve_body_wrench(state, action, "body_rate")
    expected_torque = _reference_rate_torque(
        state[:, 3:7],
        state[:, 10:13],
        torch.tensor([[1.5, -1.0, 0.0]], dtype=torch.float32),
        torch.diag(torch.tensor([0.01, 0.01, 0.02], dtype=torch.float32)).unsqueeze(0),
        torch.tensor([5.0, 4.0, 3.0], dtype=torch.float32),
        torque_ratio=0.6,
        gyro_cross_limit=100.0,
    )

    assert body_force[0, 2].item() == pytest.approx(0.6 * 5.0 * 0.8 * 9.81)
    assert torch.allclose(body_torque, expected_torque, atol=1.0e-6)


@pytest.mark.cpu_smoke
def test_drag_coefficients_reduce_translational_speed() -> None:
    cfg_base = DroneConfig(num_envs=1, dt=0.02, max_thrust=0.0)
    no_drag = Drone(cfg_base, device="cpu")
    with_drag = Drone(
        DroneConfig(num_envs=1, dt=0.02, max_thrust=0.0, drag_coeff_xy=0.5, drag_coeff_z=1.0),
        device="cpu",
    )

    state = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.5, 0.0, 0.0, 0.0]])
    no_drag.set_state(state)
    with_drag.set_state(state)
    zero_action = torch.zeros(1, 4)

    no_drag.apply_control(zero_action)
    with_drag.apply_control(zero_action)
    no_drag.integrate()
    with_drag.integrate()

    no_drag_vel = no_drag.get_state()["velocity"][0]
    with_drag_vel = with_drag.get_state()["velocity"][0]

    assert with_drag_vel[0].abs() < no_drag_vel[0].abs()
    assert with_drag_vel[2] < no_drag_vel[2]


if __name__ == "__main__":
    pytest.main([str(Path(__file__)), "-v"])
