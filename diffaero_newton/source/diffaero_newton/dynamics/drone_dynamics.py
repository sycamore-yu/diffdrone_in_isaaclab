"""Newton-backed quadrotor dynamics with optional differentiable propagation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import newton
import newton.solvers
import torch
import warp as wp

from diffaero_newton.common.constants import ARM_LENGTH, DEFAULT_DT, IXX, IYY, IZZ, QUADROTOR_MASS
from diffaero_newton.dynamics.rate_controller import RateController, RateControllerConfig, quaternion_to_matrix

wp.init()


@wp.kernel
def compute_quadrotor_wrenches(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_force_b: wp.array(dtype=wp.float32, ndim=2),
    body_torque_b: wp.array(dtype=wp.float32, ndim=2),
    drag_coeff_xy: float,
    drag_coeff_z: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    tf = body_q[tid]
    rot = wp.transform_get_rotation(tf)
    vel_w = wp.spatial_bottom(body_qd[tid])
    vel_b = wp.quat_rotate_inv(rot, vel_w)

    force_b = wp.vec3(
        body_force_b[tid, 0] - drag_coeff_xy * vel_b[0],
        body_force_b[tid, 1] - drag_coeff_xy * vel_b[1],
        body_force_b[tid, 2] - drag_coeff_z * vel_b[2],
    )
    torque_b = wp.vec3(
        body_torque_b[tid, 0],
        body_torque_b[tid, 1],
        body_torque_b[tid, 2],
    )

    force_w = wp.transform_vector(tf, force_b)
    torque_w = wp.transform_vector(tf, torque_b)
    wp.atomic_add(body_f, tid, wp.spatial_vector(torque_w, force_w))


@wp.kernel
def read_state_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    out_state: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()
    tf = body_q[tid]
    qd = body_qd[tid]
    pos = wp.transform_get_translation(tf)
    quat = wp.transform_get_rotation(tf)
    omega = wp.spatial_top(qd)
    vel = wp.spatial_bottom(qd)

    out_state[tid, 0] = pos[0]
    out_state[tid, 1] = pos[1]
    out_state[tid, 2] = pos[2]
    out_state[tid, 3] = quat[3]
    out_state[tid, 4] = quat[0]
    out_state[tid, 5] = quat[1]
    out_state[tid, 6] = quat[2]
    out_state[tid, 7] = vel[0]
    out_state[tid, 8] = vel[1]
    out_state[tid, 9] = vel[2]
    out_state[tid, 10] = omega[0]
    out_state[tid, 11] = omega[1]
    out_state[tid, 12] = omega[2]


@wp.kernel
def write_state_kernel(
    in_state: wp.array(dtype=wp.float32, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    pos = wp.vec3(in_state[tid, 0], in_state[tid, 1], in_state[tid, 2])
    quat = wp.quat(in_state[tid, 4], in_state[tid, 5], in_state[tid, 6], in_state[tid, 3])
    vel = wp.vec3(in_state[tid, 7], in_state[tid, 8], in_state[tid, 9])
    omega = wp.vec3(in_state[tid, 10], in_state[tid, 11], in_state[tid, 12])

    body_q[tid] = wp.transform(pos, quat)
    body_qd[tid] = wp.spatial_vector(omega, vel)


class _NewtonStepFn(torch.autograd.Function):
    """Torch autograd wrapper around a single Newton integration step."""

    @staticmethod
    def forward(
        ctx,
        drone: "Drone",
        state: torch.Tensor,
        body_force_b: torch.Tensor,
        body_torque_b: torch.Tensor,
        dt: float,
    ):
        requires_grad = bool(
            state.requires_grad
            or body_force_b.requires_grad
            or body_torque_b.requires_grad
            or drone.config.requires_grad
        )

        state_in = drone.model.state(requires_grad=requires_grad)
        state_out = drone.model.state(requires_grad=requires_grad)
        state_wp = wp.from_torch(state.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
        force_wp = wp.from_torch(body_force_b.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
        torque_wp = wp.from_torch(body_torque_b.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
        out_state_wp = wp.zeros((drone.num_envs, 13), dtype=wp.float32, device=drone.wp_device, requires_grad=requires_grad)

        def _forward_step():
            wp.launch(
                write_state_kernel,
                dim=drone.num_envs,
                inputs=(state_wp, state_in.body_q, state_in.body_qd),
                device=drone.wp_device,
            )
            state_in.clear_forces()
            wp.launch(
                compute_quadrotor_wrenches,
                dim=drone.num_envs,
                inputs=(
                    state_in.body_q,
                    state_in.body_qd,
                    force_wp,
                    torque_wp,
                    drone.drag_coeff_xy,
                    drone.drag_coeff_z,
                    state_in.body_f,
                ),
                device=drone.wp_device,
            )
            drone.solver.step(state_in, state_out, None, None, dt)
            wp.launch(
                read_state_kernel,
                dim=drone.num_envs,
                inputs=(state_out.body_q, state_out.body_qd, out_state_wp),
                device=drone.wp_device,
            )

        tape = wp.Tape() if requires_grad else None
        if tape is None:
            _forward_step()
        else:
            with tape:
                _forward_step()

        out_state = wp.to_torch(out_state_wp)

        ctx.tape = tape
        ctx.state_wp = state_wp
        ctx.force_wp = force_wp
        ctx.torque_wp = torque_wp
        ctx.out_state_wp = out_state_wp
        ctx.propagate_state_grad = state.requires_grad
        ctx.propagate_force_grad = body_force_b.requires_grad
        ctx.propagate_torque_grad = body_torque_b.requires_grad
        return out_state

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_state = None
        grad_force = None
        grad_torque = None

        if ctx.tape is not None:
            grad_wp = wp.from_torch(grad_output.contiguous(), dtype=wp.float32, requires_grad=False)
            ctx.tape.backward(grads={ctx.out_state_wp: grad_wp})
            if ctx.propagate_state_grad:
                grad_state = wp.to_torch(ctx.state_wp.grad).clone()
            if ctx.propagate_force_grad:
                grad_force = wp.to_torch(ctx.force_wp.grad).clone()
            if ctx.propagate_torque_grad:
                grad_torque = wp.to_torch(ctx.torque_wp.grad).clone()
            ctx.tape.zero()

        return None, grad_state, grad_force, grad_torque, None

@dataclass
class DroneConfig:
    """Configuration for the drone dynamics."""
    
    num_envs: int = 1
    dt: float = DEFAULT_DT
    requires_grad: bool = False
    arm_length: float = ARM_LENGTH
    mass: float = QUADROTOR_MASS
    inertia: tuple[float, float, float] = (IXX, IYY, IZZ)
    control_mode: str = "motor_thrust"
    torque_coeff: float = 0.01
    max_thrust: float = 20.0
    drag_coeff_xy: float = 0.0
    drag_coeff_z: float = 0.0
    k_angvel: tuple[float, float, float] = (6.0, 6.0, 2.5)
    max_body_rates: tuple[float, float, float] = (3.14, 3.14, 3.14)
    solver_type: str = "semi_implicit"
    n_substeps: int = 1
    # Control mode: "motor_thrust" (default) or "rate_controller"
    control_mode: str = "motor_thrust"
    # Rate controller gains
    K_angvel: tuple[float, float, float] = (1.0, 1.0, 0.5)
    torque_ratio: float = 1.0
    thrust_ratio: float = 1.0


VALID_CONTROL_MODES = {"motor_thrust", "body_rate"}


def motor_thrust_to_body_wrench(
    thrusts: torch.Tensor,
    *,
    arm_length: float,
    torque_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert per-rotor thrusts to body-frame force and torque."""

    d = arm_length / math.sqrt(2.0)
    body_force = torch.zeros(thrusts.shape[0], 3, device=thrusts.device, dtype=thrusts.dtype)
    body_force[:, 2] = thrusts.sum(dim=-1)
    body_torque = torch.stack(
        (
            d * (thrusts[:, 0] + thrusts[:, 3] - thrusts[:, 1] - thrusts[:, 2]),
            d * (-thrusts[:, 0] - thrusts[:, 3] + thrusts[:, 1] + thrusts[:, 2]),
            torque_coeff * (thrusts[:, 0] + thrusts[:, 2] - thrusts[:, 1] - thrusts[:, 3]),
        ),
        dim=-1,
    )
    return body_force, body_torque


def compute_linear_drag_force(
    orientation_wxyz: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    *,
    drag_coeff_xy: float,
    drag_coeff_z: float,
) -> torch.Tensor:
    """Return world-frame drag force opposing the current linear velocity."""

    rotmat_b2w = quaternion_to_matrix(orientation_wxyz)
    vel_b = torch.bmm(rotmat_b2w.transpose(-1, -2), linear_velocity_world.unsqueeze(-1)).squeeze(-1)
    drag_diag = torch.tensor(
        [drag_coeff_xy, drag_coeff_xy, drag_coeff_z],
        device=linear_velocity_world.device,
        dtype=linear_velocity_world.dtype,
    )
    drag_b = -drag_diag * vel_b
    return torch.bmm(rotmat_b2w, drag_b.unsqueeze(-1)).squeeze(-1)


class Drone:
    """Differentiable quadrotor dynamics backed by Newton."""

    def __init__(self, config: DroneConfig, device: str = "cpu"):
        self.config = config
        if config.control_mode not in VALID_CONTROL_MODES:
            raise ValueError(f"Unsupported quadrotor control mode: {config.control_mode}")
        self.num_envs = config.num_envs
        self.device = torch.device(device)
        self.wp_device = wp.get_device(str(self.device))

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        for env_id in range(self.num_envs):
            builder.add_body(
                mass=config.mass,
                I_m=wp.mat33(
                    config.inertia[0], 0.0, 0.0,
                    0.0, config.inertia[1], 0.0,
                    0.0, 0.0, config.inertia[2],
                ),
                key=f"drone_{env_id}",
            )

        self.model = builder.finalize(requires_grad=config.requires_grad, device=self.wp_device)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.arm_length = config.arm_length
        self.torque_coeff = config.torque_coeff
        self.max_thrust = config.max_thrust
        self.drag_coeff_xy = config.drag_coeff_xy
        self.drag_coeff_z = config.drag_coeff_z
        self.inertia = torch.diag(
            torch.tensor(config.inertia, device=self.device, dtype=torch.float32)
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.rate_controller = RateController(
            self.inertia,
            RateControllerConfig(
                k_angvel=config.k_angvel,
                max_body_rates=config.max_body_rates,
            ),
            device=self.device,
        )

        self._state_tensor = torch.zeros(self.num_envs, 13, dtype=torch.float32, device=self.device)
        self._control_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._control_mode = config.control_mode
        self.reset_states()

    @property
    def state(self) -> torch.Tensor:
        return self._state_tensor

    def set_state(self, new_state_tensor: torch.Tensor):
        self._state_tensor = new_state_tensor.to(self.device, dtype=torch.float32)

    def reset_states(self, positions: Optional[torch.Tensor] = None, env_ids: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.zeros(self.num_envs, 3, device=self.device)
            positions[:, 2] = 1.0
        else:
            positions = positions.to(self.device)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        new_state = self._state_tensor.detach().clone()
        new_state[env_ids, :3] = positions
        new_state[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        new_state[env_ids, 7:10] = 0.0
        new_state[env_ids, 10:13] = 0.0
        self._state_tensor = new_state

    def _resolve_body_wrench(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        control_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if control_mode == "motor_thrust":
            thrusts = control * self.max_thrust
            return motor_thrust_to_body_wrench(
                thrusts,
                arm_length=self.arm_length,
                torque_coeff=self.torque_coeff,
            )
        if control_mode == "body_rate":
            return self.rate_controller(
                state[:, 3:7],
                state[:, 10:13],
                control,
                max_collective_thrust=self.max_thrust * 4.0,
            )
        raise ValueError(f"Unsupported quadrotor control mode: {control_mode}")

    def apply_control(self, control: torch.Tensor, control_mode: Optional[str] = None):
        self._control_tensor = control.to(self.device, dtype=torch.float32)
        self._control_mode = control_mode or self.config.control_mode

    def integrate(self, dt: Optional[float] = None):
        dt = dt or self.config.dt
        sub_dt = dt / self.config.n_substeps

        state = self._state_tensor
        for _ in range(self.config.n_substeps):
            body_force_b, body_torque_b = self._resolve_body_wrench(state, self._control_tensor, self._control_mode)
            state = _NewtonStepFn.apply(self, state, body_force_b, body_torque_b, float(sub_dt))
        self._state_tensor = state

    def get_state(self) -> dict[str, torch.Tensor]:
        st = self._state_tensor
        return {
            "position": st[:, :3],
            "orientation": st[:, 3:7],
            "velocity": st[:, 7:10],
            "omega": st[:, 10:13],
        }

    def get_flat_state(self) -> torch.Tensor:
        return self._state_tensor

    def detach_graph(self):
        self._state_tensor = self._state_tensor.detach()
        self._control_tensor = self._control_tensor.detach()


def create_drone(
    num_envs: int = 1,
    dt: float = DEFAULT_DT,
    requires_grad: bool = False,
    device: str = "cpu",
) -> Drone:
    """Create a differentiable drone dynamics instance."""

    config = DroneConfig(num_envs=num_envs, dt=dt, requires_grad=requires_grad)
    return Drone(config, device=device)
