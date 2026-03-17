"""Point-mass dynamics with continuous and discrete differentiable variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import newton
import newton.solvers
import torch
import warp as wp

from diffaero_newton.common.constants import DEFAULT_DT, GRAVITY

wp.init()


@wp.kernel
def compute_pointmass_wrenches(
    body_qd: wp.array(dtype=wp.spatial_vector),
    controls: wp.array(dtype=wp.float32, ndim=2),
    gravity: wp.vec3,
    drag_coeff: float,
    mass: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    qd = body_qd[tid]
    vel = wp.spatial_bottom(qd)

    force = wp.vec3(
        controls[tid, 0] - drag_coeff * vel[0] + mass * gravity[0],
        controls[tid, 1] - drag_coeff * vel[1] + mass * gravity[1],
        controls[tid, 2] - drag_coeff * vel[2] + mass * gravity[2],
    )
    wp.atomic_add(body_f, tid, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), force))


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


class _ContinuousPointMassStepFn(torch.autograd.Function):
    """Autograd wrapper around one Newton point-mass step."""

    @staticmethod
    def forward(
        ctx,
        model: newton.Model,
        solver: newton.solvers.SolverSemiImplicit,
        wp_device,
        gravity: wp.vec3,
        drag_coeff: float,
        mass: float,
        state: torch.Tensor,
        controls: torch.Tensor,
        dt: float,
    ):
        requires_grad = bool(state.requires_grad or controls.requires_grad)
        state_in = model.state(requires_grad=requires_grad)
        state_out = model.state(requires_grad=requires_grad)
        state_wp = wp.from_torch(state.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
        controls_wp = wp.from_torch(controls.contiguous(), dtype=wp.float32, requires_grad=requires_grad)
        out_state_wp = wp.zeros(state.shape, dtype=wp.float32, device=wp_device, requires_grad=requires_grad)

        def _forward_step():
            wp.launch(
                write_state_kernel,
                dim=state.shape[0],
                inputs=(state_wp, state_in.body_q, state_in.body_qd),
                device=wp_device,
            )
            state_in.clear_forces()
            wp.launch(
                compute_pointmass_wrenches,
                dim=state.shape[0],
                inputs=(state_in.body_qd, controls_wp, gravity, drag_coeff, mass, state_in.body_f),
                device=wp_device,
            )
            solver.step(state_in, state_out, None, None, dt)
            wp.launch(
                read_state_kernel,
                dim=state.shape[0],
                inputs=(state_out.body_q, state_out.body_qd, out_state_wp),
                device=wp_device,
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

        return None, None, None, None, None, None, grad_state, grad_controls, None


@dataclass
class PointMassConfig:
    """Backward-compatible config for the continuous point-mass model."""

    num_envs: int = 1
    dt: float = DEFAULT_DT
    requires_grad: bool = False
    mass: float = 1.0
    drag_coeff: float = 0.1
    solver_type: str = "semi_implicit"
    n_substeps: int = 1
    # Action frame: 'world' or 'local'
    action_frame: str = "world"
    # Yaw alignment options
    align_yaw_with_target_direction: bool = False
    align_yaw_with_vel_ema: bool = False
    # Control delay factor
    control_delay_factor: float = 1.0


@dataclass
class ContinuousPointMassConfig(PointMassConfig):
    """Config for continuous point-mass dynamics."""


@dataclass
class DiscretePointMassConfig(PointMassConfig):
    """Config for discrete point-mass dynamics."""


class _PointMassBase:
    """Shared interface for point-mass models."""

    def __init__(self, config: PointMassConfig, device: str = "cpu"):
        self.config = config
        self.num_envs = config.num_envs
        self.device = torch.device(device)
        self.gravity = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float32, device=self.device)
        self._state_tensor = torch.zeros(self.num_envs, 13, dtype=torch.float32, device=self.device)
        self._control_tensor = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.reset_states()

    @property
    def state(self) -> torch.Tensor:
        return self._state_tensor

    def set_state(self, new_state_tensor: torch.Tensor):
        self._state_tensor = new_state_tensor.to(self.device, dtype=torch.float32)

    def reset_states(self, positions: Optional[torch.Tensor] = None, env_ids: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            positions[:, 2] = 1.0
        else:
            positions = positions.to(self.device, dtype=torch.float32)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        new_state = self._state_tensor.detach().clone()
        new_state[env_ids, :3] = positions
        new_state[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        new_state[env_ids, 7:10] = 0.0
        new_state[env_ids, 10:13] = 0.0
        self._state_tensor = new_state

    def apply_control(self, thrust_vector: torch.Tensor):
        self._control_tensor = thrust_vector[..., :3].to(self.device, dtype=torch.float32)

    def integrate(self, dt: Optional[float] = None):
        raise NotImplementedError

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


class ContinuousPointMass(_PointMassBase):
    """Newton-backed continuous point-mass model with autograd support."""

    def __init__(self, config: PointMassConfig, device: str = "cpu"):
        super().__init__(config, device=device)
        self.wp_device = wp.get_device(str(self.device))
        self.gravity_wp = wp.vec3(0.0, 0.0, GRAVITY)

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        for env_id in range(self.num_envs):
            builder.add_body(
                mass=config.mass,
                I_m=wp.mat33(
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ),
                key=f"pm_{env_id}",
            )

        self.model = builder.finalize(requires_grad=config.requires_grad, device=self.wp_device)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

    def integrate(self, dt: Optional[float] = None):
        dt = dt or self.config.dt
        sub_dt = dt / self.config.n_substeps

        state = self._state_tensor
        controls = self._control_tensor
        for _ in range(self.config.n_substeps):
            state = _ContinuousPointMassStepFn.apply(
                self.model,
                self.solver,
                self.wp_device,
                self.gravity_wp,
                float(self.config.drag_coeff),
                float(self.config.mass),
                state,
                controls,
                float(sub_dt),
            )
        self._state_tensor = state


class DiscretePointMass(_PointMassBase):
    """Torch-native discrete point-mass update with local/world frame support."""

    def integrate(self, dt: Optional[float] = None):
        dt = float(dt or self.config.dt)
        pos = self._state_tensor[:, :3]
        vel = self._state_tensor[:, 7:10]
        quat = self._state_tensor[:, 3:7]
        inv_mass = 1.0 / float(self.config.mass)

        # Transform control to world frame if using local frame
        action_frame = getattr(self.config, 'action_frame', 'world')
        if action_frame == "local":
            R = self._quat_to_rotation_matrix(quat)
            control_world = torch.bmm(R, self._control_tensor.unsqueeze(-1)).squeeze(-1)
        else:
            control_world = self._control_tensor

        # Apply drag
        drag_force = -self.config.drag_coeff * vel

        # Compute acceleration
        acc = control_world * inv_mass + self.gravity + drag_force * inv_mass

        # Semi-implicit Euler integration
        next_vel = vel + dt * acc
        next_pos = pos + dt * next_vel

        next_state = self._state_tensor.clone()
        next_state[:, :3] = next_pos
        next_state[:, 3:7] = quat
        next_state[:, 7:10] = next_vel
        next_state[:, 10:13] = 0.0
        self._state_tensor = next_state

    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w,x,y,z] to rotation matrix."""
        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]
        norm = torch.sqrt(qw*qw + qx*qx + qy*qy + qz*qz + 1e-8)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        R = torch.zeros(quat.shape[0], 3, 3, device=quat.device)
        R[:, 0, 0] = (1 - 2*(qy*qy + qz*qz)).reshape(-1)
        R[:, 0, 1] = (2*(qx*qy - qw*qz)).reshape(-1)
        R[:, 0, 2] = (2*(qx*qz + qw*qy)).reshape(-1)
        R[:, 1, 0] = (2*(qx*qy + qw*qz)).reshape(-1)
        R[:, 1, 1] = (1 - 2*(qx*qx + qz*qz)).reshape(-1)
        R[:, 1, 2] = (2*(qy*qz - qw*qx)).reshape(-1)
        R[:, 2, 0] = (2*(qx*qz - qw*qy)).reshape(-1)
        R[:, 2, 1] = (2*(qy*qz + qw*qx)).reshape(-1)
        R[:, 2, 2] = (1 - 2*(qx*qx + qy*qy)).reshape(-1)
        return R

# Backward-compatible alias
PointMass = ContinuousPointMass


def create_pointmass(
    num_envs: int = 1,
    dt: float = DEFAULT_DT,
    requires_grad: bool = False,
    device: str = "cpu",
) -> PointMass:
    """Create the backward-compatible continuous point-mass model."""
    config = PointMassConfig(num_envs=num_envs, dt=dt, requires_grad=requires_grad)
    return PointMass(config, device=device)
