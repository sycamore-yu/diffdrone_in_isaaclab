"""Differentiable pointmass dynamics using Newton physics engine.

Replaces the previous PyTorch-only RK4/Euler approach with a true Newton
integration lifecycle.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import warp as wp
import newton
import newton.solvers

from diffaero_newton.common.constants import (
    GRAVITY,
    DEFAULT_DT,
)

# Initialize Warp once
wp.init()

@wp.kernel
def compute_pointmass_wrenches(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    controls: wp.array(dtype=float, ndim=2),  # [num_envs, 3] representing thrust vector
    gravity: wp.vec3,
    drag_coeff: float,
    mass: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    # controls are thrust vector [Tx, Ty, Tz]
    tx, ty, tz = controls[tid, 0], controls[tid, 1], controls[tid, 2]
    
    # Drag force: F_drag = -D * v
    qd = body_qd[tid]
    v = wp.spatial_bottom(qd)
    v_x, v_y, v_z = v[0], v[1], v[2]
    
    f_drag_x = -drag_coeff * v_x
    f_drag_y = -drag_coeff * v_y
    f_drag_z = -drag_coeff * v_z
    
    # Gravity force
    f_grav_x = mass * gravity[0]
    f_grav_y = mass * gravity[1]
    f_grav_z = mass * gravity[2]

    total_force_x = tx + f_drag_x + f_grav_x
    total_force_y = ty + f_drag_y + f_grav_y
    total_force_z = tz + f_drag_z + f_grav_z
    
    # Newton wrench is spatial_vector(torque, force)
    # Pointmass has no torque
    torque = wp.vec3(0.0, 0.0, 0.0)
    force = wp.vec3(total_force_x, total_force_y, total_force_z)
    
    wp.atomic_add(body_f, tid, wp.spatial_vector(torque, force))


@wp.kernel
def read_state_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    out_state: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    tf = body_q[tid]
    qd = body_qd[tid]
    
    pos = wp.transform_get_translation(tf)
    quat = wp.transform_get_rotation(tf)
    
    # qd = spatial_vector(w, v) meaning angular then linear
    w = wp.spatial_top(qd)      # w_x, w_y, w_z
    v = wp.spatial_bottom(qd)   # v_x, v_y, v_z
    
    # state: [pos(3), quat(4: w, x, y, z), vel(3), omega(3)]
    out_state[tid, 0] = pos[0]
    out_state[tid, 1] = pos[1]
    out_state[tid, 2] = pos[2]
    
    # PyTorch version expects [w, x, y, z]
    out_state[tid, 3] = quat[3] # w
    out_state[tid, 4] = quat[0] # x
    out_state[tid, 5] = quat[1] # y
    out_state[tid, 6] = quat[2] # z
    
    out_state[tid, 7] = v[0]
    out_state[tid, 8] = v[1]
    out_state[tid, 9] = v[2]
    
    out_state[tid, 10] = w[0]
    out_state[tid, 11] = w[1]
    out_state[tid, 12] = w[2]


@wp.kernel
def write_state_kernel(
    in_state: wp.array(dtype=float, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    
    px = in_state[tid, 0]
    py = in_state[tid, 1]
    pz = in_state[tid, 2]
    
    # Torch uses [w, x, y, z] -> Warp uses [x, y, z, w]
    qw = in_state[tid, 3]
    qx = in_state[tid, 4]
    qy = in_state[tid, 5]
    qz = in_state[tid, 6]
    
    vx = in_state[tid, 7]
    vy = in_state[tid, 8]
    vz = in_state[tid, 9]
    
    wx = in_state[tid, 10]
    wy = in_state[tid, 11]
    wz = in_state[tid, 12]
    
    pos = wp.vec3(px, py, pz)
    quat = wp.quat(qx, qy, qz, qw)
    
    body_q[tid] = wp.transform(pos, quat)
    
    # body_qd is (angular, linear)
    w = wp.vec3(wx, wy, wz)
    v = wp.vec3(vx, vy, vz)
    body_qd[tid] = wp.spatial_vector(w, v)


@dataclass
class PointMassConfig:
    """Configuration for the pointmass dynamics."""
    num_envs: int = 1
    dt: float = DEFAULT_DT
    requires_grad: bool = False
    mass: float = 1.0
    drag_coeff: float = 0.1
    solver_type: str = "semi_implicit"
    n_substeps: int = 1


class PointMass:
    """Differentiable pointmass dynamics backed by Newton."""

    def __init__(self, config: PointMassConfig, device: str = "cpu"):
        self.config = config
        self.num_envs = config.num_envs
        self.device = device
        
        self.wp_device = wp.get_device(device)

        # Initialize the Newton ModelBuilder
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        
        # We model each environment as a separate independent body
        for i in range(self.num_envs):
            builder.add_body(
                mass=config.mass,
                I_m=wp.mat33(
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ), # Point mass doesn't technically rotate but needs inertia matrix for Newton
                key=f"pm_{i}"
            )
        
        self.model = builder.finalize(requires_grad=config.requires_grad, device=self.wp_device)
        
        self.state_curr = self.model.state()
        self.state_next = self.model.state()
        
        self._wp_controls = wp.zeros((self.num_envs, 3), dtype=float, device=self.wp_device, requires_grad=config.requires_grad)

        # PyTorch representation of the state [num_envs, 13] for unified interface
        self._state_tensor = torch.zeros(self.num_envs, 13, device=device, requires_grad=config.requires_grad)
        self.reset_states()
        
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.gravity = wp.vec3(0.0, 0.0, GRAVITY)

    @property
    def state(self) -> torch.Tensor:
        """Returns the PyTorch tensor representing state: [pos, quat, vel, omega]."""
        wp.launch(
            read_state_kernel,
            dim=self.num_envs,
            inputs=(self.state_curr.body_q, self.state_curr.body_qd, wp.from_torch(self._state_tensor)),
            device=self.wp_device
        )
        return self._state_tensor

    def set_state(self, new_state_tensor: torch.Tensor):
        self._state_tensor = new_state_tensor
        wp.launch(
            write_state_kernel,
            dim=self.num_envs,
            inputs=(wp.from_torch(self._state_tensor), self.state_curr.body_q, self.state_curr.body_qd),
            device=self.wp_device
        )

    def reset_states(self, positions: Optional[torch.Tensor] = None, env_ids: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.zeros(self.num_envs, 3, device=self.device)
            positions[:, 2] = 1.0

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            reset_positions = positions
        else:
            reset_positions = positions

        new_state = self._state_tensor.clone()
        new_state[env_ids, :3] = reset_positions
        # Reset quaternion to identity (w=1, x=y=z=0)
        new_state[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # Reset velocities
        new_state[env_ids, 7:10] = 0.0
        new_state[env_ids, 10:13] = 0.0
        
        self.set_state(new_state)

    def apply_control(self, thrust_vector: torch.Tensor):
        """Apply control inputs. Slices first 3 dims if a 4D action is provided."""
        if thrust_vector.shape[1] > 3:
            thrust_vector = thrust_vector[:, :3]
        # Copy to warp
        wp.copy(self._wp_controls, wp.from_torch(thrust_vector))

    def integrate(self, dt: Optional[float] = None):
        """Step the simulation forward using Newton."""
        dt = dt or self.config.dt
        sub_dt = dt / self.config.n_substeps

        for _ in range(self.config.n_substeps):
            self.state_curr.clear_forces()
            
            wp.launch(
                compute_pointmass_wrenches,
                dim=self.num_envs,
                inputs=(
                    self.state_curr.body_q,
                    self.state_curr.body_qd,
                    self._wp_controls,
                    self.gravity,
                    self.config.drag_coeff,
                    self.config.mass,
                    self.state_curr.body_f,
                ),
                device=self.wp_device
            )
            
            self.solver.step(
                self.state_curr,
                self.state_next,
                None,
                None,
                sub_dt,
            )
            
            self.state_curr, self.state_next = self.state_next, self.state_curr

        # Read back Newton state to pyTorch
        new_state_tensor = torch.zeros_like(self._state_tensor)
        wp.launch(
            read_state_kernel,
            dim=self.num_envs,
            inputs=(self.state_curr.body_q, self.state_curr.body_qd, wp.from_torch(new_state_tensor)),
            device=self.wp_device
        )
        self._state_tensor = new_state_tensor

    def get_state(self) -> dict[str, torch.Tensor]:
        st = self.state
        return {
            "position": st[:, :3],
            "orientation": st[:, 3:7],
            "velocity": st[:, 7:10],
            "omega": st[:, 10:13],
        }

    def get_flat_state(self) -> torch.Tensor:
        """Get the current state as flat tensor [num_envs, 13]."""
        return self.state

    def detach_graph(self):
        """Detach runtime tensors between rollout iterations."""
        self._state_tensor = self._state_tensor.detach()
        self.set_state(self._state_tensor)


def create_pointmass(
    num_envs: int = 1,
    dt: float = DEFAULT_DT,
    requires_grad: bool = False,
    device: str = "cpu",
) -> PointMass:
    """Create a differentiable pointmass dynamics instance."""
    config = PointMassConfig(num_envs=num_envs, dt=dt, requires_grad=requires_grad)
    return PointMass(config, device=device)
