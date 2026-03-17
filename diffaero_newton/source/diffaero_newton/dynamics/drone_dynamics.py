"""Newton-backed quadrotor dynamics with optional differentiable propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import newton
import newton.solvers
import torch
import warp as wp

from diffaero_newton.common.constants import ARM_LENGTH, DEFAULT_DT, IXX, IYY, IZZ, QUADROTOR_MASS

wp.init()


@wp.kernel
def compute_quadrotor_wrenches(
    body_q: wp.array(dtype=wp.transform),
    controls: wp.array(dtype=wp.float32, ndim=2),
    arm_length: float,
    torque_coeff: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    t1 = controls[tid, 0]
    t2 = controls[tid, 1]
    t3 = controls[tid, 2]
    t4 = controls[tid, 3]

    d = arm_length / 1.41421356
    force_b = wp.vec3(0.0, 0.0, t1 + t2 + t3 + t4)
    torque_b = wp.vec3(
        d * (t1 + t4 - t2 - t3),
        d * (-t1 - t4 + t2 + t3),
        torque_coeff * (t1 + t3 - t2 - t4),
    )

    tf = body_q[tid]
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
    def forward(ctx, drone: "Drone", state: torch.Tensor, controls: torch.Tensor, dt: float):
        requires_grad = bool(state.requires_grad or controls.requires_grad or drone.config.requires_grad)

        state_in = drone.model.state(requires_grad=requires_grad)
        state_out = drone.model.state(requires_grad=requires_grad)
        state_wp = wp.from_torch(state.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
        controls_wp = wp.from_torch(controls.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
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
                inputs=(state_in.body_q, controls_wp, drone.arm_length, drone.ct, state_in.body_f),
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
        ctx.controls_wp = controls_wp
        ctx.out_state_wp = out_state_wp
        ctx.propagate_state_grad = state.requires_grad
        ctx.propagate_control_grad = controls.requires_grad
        return out_state

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_state = None
        grad_controls = None

        if ctx.tape is not None:
            grad_wp = wp.from_torch(grad_output.contiguous(), dtype=wp.float32, requires_grad=False)
            ctx.tape.backward(grads={ctx.out_state_wp: grad_wp})
            if ctx.propagate_state_grad:
                grad_state = wp.to_torch(ctx.state_wp.grad).clone()
            if ctx.propagate_control_grad:
                grad_controls = wp.to_torch(ctx.controls_wp.grad).clone()
            ctx.tape.zero()

        return None, grad_state, grad_controls, None

@dataclass
class DroneConfig:
    """Configuration for the drone dynamics."""
    
    num_envs: int = 1
    dt: float = DEFAULT_DT
    requires_grad: bool = False
    arm_length: float = ARM_LENGTH
    mass: float = QUADROTOR_MASS
    inertia: tuple[float, float, float] = (IXX, IYY, IZZ)
    solver_type: str = "semi_implicit"
    n_substeps: int = 1
    # Control mode: "motor_thrust" (default) or "rate_controller"
    control_mode: str = "motor_thrust"
    # Rate controller gains
    K_angvel: tuple[float, float, float] = (1.0, 1.0, 0.5)
    torque_ratio: float = 1.0
    thrust_ratio: float = 1.0


class Drone:
    """Differentiable quadrotor dynamics backed by Newton."""

    def __init__(self, config: DroneConfig, device: str = "cpu"):
        self.config = config
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
        self.ct = 0.01
        self.max_thrust = 20.0

        self._state_tensor = torch.zeros(self.num_envs, 13, dtype=torch.float32, device=self.device)
        self._control_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.reset_states()

    @property
    def state(self) -> torch.Tensor:
        return self._state_tensor

    def set_state(self, new_state_tensor: torch.Tensor):
        self._state_tensor = new_state_tensor.to(self.device)

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

    def apply_control(self, thrust_normalized: torch.Tensor):
        self._control_tensor = thrust_normalized.to(self.device, dtype=torch.float32) * self.max_thrust

    def integrate(self, dt: Optional[float] = None):
        dt = dt or self.config.dt
        sub_dt = dt / self.config.n_substeps

        state = self._state_tensor
        controls = self._control_tensor
        for _ in range(self.config.n_substeps):
            state = _NewtonStepFn.apply(self, state, controls, float(sub_dt))
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
