"""Drone environment with Newton-backed dynamics and a lightweight RL contract."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import math

import torch

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.common.direct_rl_shim import DirectRLEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg
from diffaero_newton.dynamics.registry import create_dynamics
from diffaero_newton.tasks.obstacle_manager import ObstacleManager
from diffaero_newton.tasks.reward_terms import compute_risk_loss


class DroneEnv(DirectRLEnv):
    """Obstacle-avoidance environment using Newton dynamics for propagation."""

    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.dynamics.num_envs = self.num_envs
        self.cfg.dynamics.dt = self.physics_dt
        self.cfg.dynamics.requires_grad = self.cfg.differentiable_dynamics
        self.drone = create_dynamics(self.cfg.dynamics, device=self.device)
        self.drone.reset_states()
        self.actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.goal_position = torch.zeros(self.num_envs, 3, device=self.device)

        obstacle_cfg = ObstacleTaskCfg(num_obstacles=self.cfg.num_obstacles)
        self.obstacle_manager = ObstacleManager(
            num_envs=self.num_envs,
            cfg=obstacle_cfg,
            device=self.device,
        )

        self.collision_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.nearest_obstacle_dist = torch.zeros(self.num_envs, device=self.device)
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """No-op scene hook kept for IsaacLab compatibility."""
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if seed is not None:
            torch.manual_seed(seed)

        self.drone.reset_states()
        self.episode_length_buf.zero_()
        self.actions.zero_()
        self.prev_actions.zero_()
        self._sample_goals_for_envs()
        self.obstacle_manager.reset()

        obs = self._get_observations()
        return obs, {}

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions
        self.actions = actions.to(self.device)

    def _apply_action(self):
        self.drone.apply_control(self.actions)
        self.drone.integrate(self.physics_dt)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        state = self.drone.get_state()
        state_flat = torch.cat(
            [
                state["position"],
                state["velocity"],
                state["orientation"],
                state["omega"],
            ],
            dim=1,
        )

        goal_rel = self.goal_position - state["position"]
        nearest_dist = self.obstacle_manager.compute_nearest_distances(state["position"])

        obs = torch.cat(
            [
                state_flat,
                goal_rel,
                self.prev_actions,
                nearest_dist.unsqueeze(1),
            ],
            dim=1,
        )

        self.nearest_obstacle_dist = nearest_dist
        self.goal_dist = torch.norm(goal_rel, dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos = state["position"]
        goal_dist = torch.norm(self.goal_position - pos, dim=1)
        tracking_reward = torch.exp(-goal_dist / 0.5)

        nearest_dist = self.obstacle_manager.compute_nearest_distances(pos)
        obstacle_penalty = -0.1 * torch.exp(-nearest_dist / 0.5)

        vel_penalty = -0.01 * torch.sum(state["velocity"] ** 2, dim=1)
        up_reward = state["orientation"][:, 0]
        action_penalty = -0.001 * torch.sum((self.prev_actions - self.actions) ** 2, dim=1)

        reward = tracking_reward + obstacle_penalty + vel_penalty + up_reward * 0.1 + action_penalty - 0.01
        return reward.detach()

    def _get_loss(self) -> torch.Tensor:
        state = self.drone.get_state()
        flat_state = self.drone.get_flat_state()

        goal_dist = torch.norm(self.goal_position - state["position"], dim=1)
        risk_loss, _ = compute_risk_loss(flat_state, self.obstacle_manager.obstacles)
        vel_loss = 0.01 * torch.sum(state["velocity"] ** 2, dim=1)
        up_loss = 0.05 * (1.0 - state["orientation"][:, 0].clamp(-1.0, 1.0))
        action_rate_loss = 0.01 * torch.sum((self.prev_actions - self.actions) ** 2, dim=1)

        return goal_dist + risk_loss + vel_loss + up_loss + action_rate_loss

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.drone.get_state()
        pos = state["position"]

        ground_collision = pos[:, 2] < self.cfg.termination_height
        obstacle_collision = self.obstacle_manager.check_collisions(pos)
        terminated = ground_collision | obstacle_collision
        truncated = self.episode_length_buf >= self.max_episode_length

        self.collision_count = obstacle_collision
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        positions = torch.zeros(len(env_ids), 3, device=self.device)
        positions[:, 2] = 1.0
        self.drone.reset_states(positions, env_ids=env_ids_tensor)
        self.prev_actions[env_ids_tensor] = 0.0
        self.actions[env_ids_tensor] = 0.0
        self._sample_goals_for_envs(env_ids_tensor)
        self.obstacle_manager.reset(env_ids_tensor)

    def _sample_goals_for_envs(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        count = len(env_ids)
        radius = 3.0
        angles = torch.rand(count, 2, device=self.device) * 2.0 * math.pi
        z_vals = torch.rand(count, device=self.device) * 2.0 + 1.0

        self.goal_position[env_ids, 0] = radius * torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
        self.goal_position[env_ids, 1] = radius * torch.sin(angles[:, 0]) * torch.sin(angles[:, 1])
        self.goal_position[env_ids, 2] = z_vals

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._pre_physics_step(action)
        for _ in range(self.decimation):
            self._apply_action()

        self.episode_length_buf += 1

        obs_before_reset = self._get_observations()
        loss = self._get_loss()
        reward = self._get_rewards()
        terminated, truncated = self._get_dones()
        reset_mask = terminated | truncated

        returned_obs = obs_before_reset
        state_before_reset = self.drone.get_flat_state()
        if reset_mask.any():
            self._reset_idx(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
            returned_obs = self._get_observations()

        extras = {
            "episode": {
                "r": reward.sum().item() / self.num_envs,
                "l": self.episode_length_buf.float().mean().item(),
            },
            "obstacles": {
                "nearest_dist": self.nearest_obstacle_dist.mean().item(),
                "goal_dist": self.goal_dist.mean().item(),
                "collisions": self.collision_count.sum().item(),
            },
            "reset": reset_mask,
            "terminated": terminated,
            "truncated": truncated,
            "next_obs_before_reset": obs_before_reset["policy"].clone(),
            "next_state_before_reset": state_before_reset.clone(),
        }

        return returned_obs, self.drone.get_flat_state(), loss, reward, extras

    def detach_graph(self):
        self.prev_actions = self.prev_actions.detach()
        self.actions = self.actions.detach()
        self.goal_position = self.goal_position.detach()
        self.drone.detach_graph()


def create_env(cfg: Optional[DroneEnvCfg] = None, **kwargs) -> DroneEnv:
    """Create a drone environment."""

    return DroneEnv(cfg=cfg or DroneEnvCfg(), **kwargs)
