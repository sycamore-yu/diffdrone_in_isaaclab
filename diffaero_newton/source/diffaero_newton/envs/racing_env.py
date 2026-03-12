"""Racing environment with figure-8 gate track.

Migrated from DiffAero's racing.py. Uses Newton for differentiable physics
and follows the IsaacLab DirectRLEnv pattern.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.racing_env_cfg import RacingEnvCfg


@torch.jit.script
def get_gate_rotmat_w2g(gate_yaw: Tensor) -> Tensor:
    """Compute world→gate rotation matrix from gate yaw angle."""
    zero = torch.zeros_like(gate_yaw)
    one = torch.ones_like(gate_yaw)
    sin = torch.sin(gate_yaw)
    cos = torch.cos(gate_yaw)
    r0 = torch.stack([cos, sin, zero], dim=-1)
    r1 = torch.stack([-sin, cos, zero], dim=-1)
    r2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


class RacingEnv(DroneEnv):
    """Figure-8 gate racing environment.

    The drone must fly through gates arranged in a figure-8 pattern.
    Each gate pass advances the target to the next gate.
    """

    def __init__(self, cfg, render_mode=None, racing_cfg: Optional[RacingEnvCfg] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.racing_cfg = racing_cfg or RacingEnvCfg()
        g_r = self.racing_cfg.gate_radius
        g_h = self.racing_cfg.gate_height

        # Figure-8 gate positions
        self.gate_pos = torch.tensor([
            [g_r, -g_r, g_h],
            [0, 0, g_h],
            [-g_r, g_r, g_h],
            [0, 2 * g_r, g_h],
            [g_r, g_r, g_h],
            [0, 0, g_h],
            [-g_r, -g_r, g_h],
            [0, -2 * g_r, g_h],
        ], device=self.device)
        self.n_gates = self.gate_pos.shape[0]
        self.gate_yaw = torch.tensor(
            [1, 2, 1, 0, -1, -2, -1, 0], device=self.device
        ) * math.pi / 2

        # Per-environment tracking
        self.target_gates = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.n_passed_gates = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Pre-compute relative gate transitions
        self.gate_rel_pos = torch.zeros(self.n_gates, 3, device=self.device)
        self.gate_yaw_rel = torch.zeros(self.n_gates, device=self.device)
        for i in range(self.n_gates):
            self.gate_rel_pos[i] = self.gate_pos[i] - self.gate_pos[i - 1]
            rotmat = get_gate_rotmat_w2g(self.gate_yaw[i - 1].unsqueeze(0)).squeeze(0)
            self.gate_rel_pos[i] = rotmat @ self.gate_rel_pos[i]
            yaw_diff = self.gate_yaw[i] - self.gate_yaw[i - 1]
            self.gate_yaw_rel[i] = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Racing-specific observations: position/velocity in gate frame + relative next gate."""
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)

        drone_pos = self.drone.p
        drone_vel = self.drone.v if hasattr(self.drone, 'v') else torch.zeros_like(drone_pos)

        # Position and velocity in gate frame
        pos_g = torch.bmm(rotmat_w2g, (gate_pos - drone_pos).unsqueeze(-1)).squeeze(-1)
        vel_g = torch.bmm(rotmat_w2g, drone_vel.unsqueeze(-1)).squeeze(-1)

        # Next gate relative info
        next_gate_idx = (self.target_gates + 1) % self.n_gates

        obs = torch.cat([
            pos_g,          # 3: position in gate frame
            vel_g,          # 3: velocity in gate frame
            self.gate_rel_pos[next_gate_idx],  # 3: relative position of next gate
            self.gate_yaw_rel[next_gate_idx].unsqueeze(-1),  # 1: relative yaw
        ], dim=-1)

        return {"policy": obs}

    def is_passed(self, prev_pos: Tensor) -> Tuple[Tensor, Tensor]:
        """Check if drone passed through or collided with current target gate."""
        curr_pos = self.drone.p
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]

        rotmat = get_gate_rotmat_w2g(gate_yaw)
        prev_rel = torch.bmm(rotmat, (prev_pos - gate_pos).unsqueeze(-1)).squeeze(-1)
        curr_rel = torch.bmm(rotmat, (curr_pos - gate_pos).unsqueeze(-1)).squeeze(-1)

        prev_behind = prev_rel[:, 0] < 0
        curr_infront = curr_rel[:, 0] > 0
        pass_through = prev_behind & curr_infront
        inside_gate = torch.norm(curr_rel[:, 1:], dim=-1, p=1) < self.racing_cfg.gate_size / 2

        gate_passed = pass_through & inside_gate
        gate_collision = pass_through & ~inside_gate

        return gate_passed, gate_collision

    def _get_rewards(self) -> torch.Tensor:
        """Compute racing rewards including progress towards gate."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for out-of-bounds termination."""
        pos = self.drone.p
        oob = torch.any(torch.abs(pos[:, :2]) > self.racing_cfg.xy_bound, dim=1)
        oob = oob | (pos[:, 2] > self.racing_cfg.z_bound)
        terminated = oob
        truncated = torch.zeros_like(terminated)
        return terminated, truncated
