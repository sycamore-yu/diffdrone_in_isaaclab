"""Racing environment with figure-8 gate track."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from diffaero_newton.configs.racing_env_cfg import RacingEnvCfg
from diffaero_newton.envs.drone_env import DroneEnv


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

    cfg: RacingEnvCfg

    def __init__(self, cfg: RacingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.racing_cfg = cfg
        g_r = cfg.gate_radius
        g_h = cfg.gate_height

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

        self.target_gates = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.n_passed_gates = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_gate_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_oob = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_reward = torch.zeros(self.num_envs, device=self.device)
        self.last_loss = torch.zeros(self.num_envs, device=self.device)

        self.gate_rel_pos = torch.zeros(self.n_gates, 3, device=self.device)
        self.gate_yaw_rel = torch.zeros(self.n_gates, device=self.device)
        for i in range(self.n_gates):
            self.gate_rel_pos[i] = self.gate_pos[i] - self.gate_pos[i - 1]
            rotmat = get_gate_rotmat_w2g(self.gate_yaw[i - 1].unsqueeze(0)).squeeze(0)
            self.gate_rel_pos[i] = rotmat @ self.gate_rel_pos[i]
            yaw_diff = self.gate_yaw[i] - self.gate_yaw[i - 1]
            self.gate_yaw_rel[i] = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Racing-specific observations in the current target-gate frame."""
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)

        drone_state = self.drone.get_state()
        drone_pos = drone_state["position"]
        drone_vel = drone_state.get("velocity", torch.zeros_like(drone_pos))

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
        """Check whether the current target gate was crossed cleanly or hit."""
        curr_pos = self.drone.get_state()["position"]
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]

        rotmat = get_gate_rotmat_w2g(gate_yaw)
        prev_rel = torch.bmm(rotmat, (prev_pos - gate_pos).unsqueeze(-1)).squeeze(-1)
        curr_rel = torch.bmm(rotmat, (curr_pos - gate_pos).unsqueeze(-1)).squeeze(-1)

        prev_behind = prev_rel[:, 0] < 0
        curr_infront = curr_rel[:, 0] > 0
        pass_through = prev_behind & curr_infront
        inside_gate = torch.linalg.vector_norm(curr_rel[:, 1:], dim=-1) <= self.racing_cfg.gate_size / 2

        gate_passed = pass_through & inside_gate
        gate_collision = pass_through & ~inside_gate

        return gate_passed, gate_collision

    def _get_rewards(self) -> torch.Tensor:
        return self.last_reward

    def _get_loss(self) -> torch.Tensor:
        return self.last_loss

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self.drone.get_state()["position"]
        oob = torch.any(torch.abs(pos[:, :2]) > self.racing_cfg.xy_bound, dim=1)
        oob = oob | (pos[:, 2] > self.racing_cfg.z_bound) | (pos[:, 2] < self.cfg.termination_height)
        terminated = oob | self.last_gate_collision
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        obs, extras = super().reset(seed=seed, options=options)
        self.target_gates.zero_()
        self.n_passed_gates.zero_()
        self.last_gate_passed.zero_()
        self.last_gate_collision.zero_()
        self.last_oob.zero_()
        self.last_reward.zero_()
        self.last_loss.zero_()
        return self._get_observations(), extras

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.target_gates[env_ids_tensor] = 0
        self.n_passed_gates[env_ids_tensor] = 0
        self.last_gate_passed[env_ids_tensor] = False
        self.last_gate_collision[env_ids_tensor] = False
        self.last_oob[env_ids_tensor] = False
        self.last_reward[env_ids_tensor] = 0.0
        self.last_loss[env_ids_tensor] = 0.0

    def _current_gate_frame(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)
        drone_state = self.drone.get_state()
        rel_pos = torch.bmm(rotmat_w2g, (gate_pos - drone_state["position"]).unsqueeze(-1)).squeeze(-1)
        vel_g = torch.bmm(rotmat_w2g, drone_state["velocity"].unsqueeze(-1)).squeeze(-1)
        return gate_pos, gate_yaw, rel_pos, vel_g

    def _compute_step_terms(self, prev_gate_dist: Tensor) -> Tuple[Tensor, Tensor]:
        _, _, rel_pos, vel_g = self._current_gate_frame()
        gate_dist = torch.linalg.vector_norm(rel_pos, dim=-1)
        lateral_dist = torch.linalg.vector_norm(rel_pos[:, 1:], dim=-1)
        progress = (prev_gate_dist - gate_dist).clamp(min=-2.0, max=2.0)
        jerk = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)
        vel_loss = torch.sum(vel_g ** 2, dim=-1)

        pass_bonus = self.last_gate_passed.float() * self.racing_cfg.reward_pass
        collision_penalty = self.last_gate_collision.float() * self.racing_cfg.reward_collision
        oob_penalty = self.last_oob.float() * self.racing_cfg.reward_oob

        reward = (
            self.racing_cfg.reward_constant
            + self.racing_cfg.reward_progress * progress
            + pass_bonus
            - collision_penalty
            - oob_penalty
        )
        loss = (
            self.racing_cfg.progress_loss_weight * gate_dist
            + self.racing_cfg.pos_loss_weight * lateral_dist
            + self.racing_cfg.vel_loss_weight * vel_loss
            + self.racing_cfg.jerk_loss_weight * jerk
            + self.last_gate_collision.float() * self.racing_cfg.collision_loss_weight
            + self.last_oob.float() * self.racing_cfg.oob_loss_weight
            - self.last_gate_passed.float() * self.racing_cfg.reward_pass
        )
        return reward.detach(), loss

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict[str, Any]]:
        prev_pos = self.drone.get_state()["position"].clone()
        _, _, prev_rel_pos, _ = self._current_gate_frame()
        prev_gate_dist = torch.linalg.vector_norm(prev_rel_pos, dim=-1)

        self._pre_physics_step(action)
        for _ in range(self.decimation):
            self._apply_action()

        self.episode_length_buf += 1

        gate_passed, gate_collision = self.is_passed(prev_pos)
        self.last_gate_passed = gate_passed
        self.last_gate_collision = gate_collision

        pos = self.drone.get_state()["position"]
        self.last_oob = torch.any(torch.abs(pos[:, :2]) > self.racing_cfg.xy_bound, dim=1)
        self.last_oob = self.last_oob | (pos[:, 2] > self.racing_cfg.z_bound) | (pos[:, 2] < self.cfg.termination_height)

        self.last_reward, self.last_loss = self._compute_step_terms(prev_gate_dist)
        reward = self.last_reward.clone()
        loss = self.last_loss.clone()

        if gate_passed.any():
            self.n_passed_gates = self.n_passed_gates + gate_passed.long()
            self.target_gates = (self.target_gates + gate_passed.long()) % self.n_gates

        obs_before_reset = self._get_observations()
        terminated, truncated = self._get_dones()
        reset_mask = terminated | truncated

        returned_obs = obs_before_reset
        state_before_reset = self.drone.get_flat_state()
        if reset_mask.any():
            self._reset_idx(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
            returned_obs = self._get_observations()

        extras = {
            "episode": {
                "r": reward.mean().item(),
                "l": self.episode_length_buf.float().mean().item(),
            },
            "racing": {
                "gate_passed": int(gate_passed.sum().item()),
                "gate_collisions": int(gate_collision.sum().item()),
                "target_gate_mean": self.target_gates.float().mean().item(),
                "passed_gate_mean": self.n_passed_gates.float().mean().item(),
            },
            "reset": reset_mask,
            "terminated": terminated,
            "truncated": truncated,
            "next_obs_before_reset": obs_before_reset["policy"].clone(),
            "next_state_before_reset": state_before_reset.clone(),
        }
        return returned_obs, self.drone.get_flat_state(), loss, reward, extras
