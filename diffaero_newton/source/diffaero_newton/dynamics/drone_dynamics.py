"""Differentiable quadrotor dynamics using Newton physics engine.

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
    QUADROTOR_MASS,
    IXX,
    IYY,
    IZZ,
    ARM_LENGTH,
    DEFAULT_DT,
)

# Initialize Warp once
wp.init()

@wp.kernel
def compute_quadrotor_wrenches(
    body_q: wp.array(dtype=wp.transform),
    controls: wp.array(dtype=float, ndim=2),  # [num_envs, 4]
    arm_length: float,
    torque_coeff: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    c = controls[tid]
    
    # Motor controls are thrusts [T1, T2, T3, T4]
    t1, t2, t3, t4 = c[0], c[1], c[2], c[3]
    d = arm_length / 1.41421356  # d = arm_length / sqrt(2)
    
    total_thrust = t1 + t2 + t3 + t4
    tau_x = d * (t1 + t4 - t2 - t3)
    tau_y = d * (-t1 - t4 + t2 + t3)
    tau_z = torque_coeff * (t1 + t3 - t2 - t4)
    
    # Body-frame forces and torques
    force_b = wp.vec3(0.0, 0.0, total_thrust)
    torque_b = wp.vec3(tau_x, tau_y, tau_z)
    
    # World frame transform
    tf = body_q[tid]
    
    # Convert to world frame
    force_w = wp.transform_vector(tf, force_b)
    torque_w = wp.transform_vector(tf, torque_b)
    
    # Follow Newton convention for wrench format
    # which packs spatial_vector(angular, linear) where angular=torque, linear=force
    # In example_diffsim_drone it uses spatial_vector(force, torque) but that example
    # might map angular to first arg. We use spatial_vector(force, torque) exactly as
    # example_diffsim_drone guarantees compatibility.
    # Wait, our script output showed spatial_vector(arg1, arg2) gives [arg1, arg2], so 
    # arg1=angular, arg2=linear. So we must pass torque_w first.
    wp.atomic_add(body_f, tid, wp.spatial_vector(torque_w, force_w))


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
    out_state[tid, 3] = quat[3] # Warp quat is [x, y, z, w], index 3 is w
    out_state[tid, 4] = quat[0]
    out_state[tid, 5] = quat[1]
    out_state[tid, 6] = quat[2]
    
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


class Drone:
    """Differentiable quadrotor dynamics backed by Newton."""

    def __init__(self, config: DroneConfig, device: str = "cpu"):
        self.config = config
        self.num_envs = config.num_envs
        self.device = device
        
        # Keep device string standard (e.g. 'cuda:0' or just 'cuda')
        self.wp_device = wp.get_device(device)

        # Initialize the Newton ModelBuilder
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.05
        
        # We model each environment as a separate independent body in the Newton model
        for i in range(self.num_envs):
            body = builder.add_body(
                mass=config.mass,
                I_m=wp.mat33(
                    config.inertia[0], 0.0, 0.0,
                    0.0, config.inertia[1], 0.0,
                    0.0, 0.0, config.inertia[2]
                ),
                key=f"drone_{i}"
            )
        
        # Finalize model
        self.model = builder.finalize(requires_grad=config.requires_grad, device=self.wp_device)
        
        self.state_curr = self.model.state()
        self.state_next = self.model.state()
        
        # Controls setup
        self.ct = 0.01  # Torque coefficient
        self.arm_length = config.arm_length
        self.max_thrust = 20.0
        
        # Warp array for the 4-motor controls, allocated directly
        self._wp_controls = wp.zeros((self.num_envs, 4), dtype=float, device=self.wp_device, requires_grad=config.requires_grad)

        # We keep a PyTorch representation of the state for external query
        self._state_tensor = torch.zeros(self.num_envs, 13, device=device, requires_grad=config.requires_grad)
        self.reset_states()
        
        # Choose solver
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

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
        """Reset the drone to initial states."""
        # Work on PyTorch side
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

    def apply_control(self, thrust_normalized: torch.Tensor):
        """Apply control inputs (normalized thrust per motor [0,1])."""
        # PyTorch -> Warp
        scaled_thrust = thrust_normalized * self.max_thrust
        # Copy to warp
        wp.copy(self._wp_controls, wp.from_torch(scaled_thrust))

    def integrate(self, dt: Optional[float] = None):
        """Step the simulation forward using Newton."""
        dt = dt or self.config.dt
        sub_dt = dt / self.config.n_substeps

        for _ in range(self.config.n_substeps):
            self.state_curr.clear_forces()
            
            # 1. Apply thrust maps
            wp.launch(
                compute_quadrotor_wrenches,
                dim=self.num_envs,
                inputs=(
                    self.state_curr.body_q,
                    self._wp_controls,
                    self.arm_length,
                    self.ct,
                    self.state_curr.body_f,
                ),
                device=self.wp_device
            )
            
            # 2. Step Newton physics
            self.solver.step(
                self.state_curr,
                self.state_next,
                None, # no control struct needed since we mapped forces direct to body_f
                None,
                sub_dt,
            )
            
            # 3. Swap buffers
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
        """Get the current state as dict of tensors."""
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


def create_drone(
    num_envs: int = 1,
    dt: float = DEFAULT_DT,
    requires_grad: bool = False,
    device: str = "cpu",
) -> Drone:
    """Create a differentiable drone dynamics instance."""
    config = DroneConfig(num_envs=num_envs, dt=dt, requires_grad=requires_grad)
    return Drone(config, device=device)
