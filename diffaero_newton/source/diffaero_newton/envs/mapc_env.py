from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.common.direct_rl_shim import DirectRLEnv
from diffaero_newton.configs.dynamics_cfg import is_pointmass_model_type
from diffaero_newton.configs.mapc_env_cfg import MAPCEnvCfg
from diffaero_newton.dynamics.registry import create_dynamics


EnvStepResult = tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]
RESET_MARGIN = 0.5
MIN_POINTMASS_RESET_HEIGHT = 1.0
GRID_POINTS_PER_AXIS = 5


class MAPCEnv(DirectRLEnv):
    """Multi-agent position-control environment."""

    cfg: MAPCEnvCfg

    def __init__(self, cfg: MAPCEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.n_agents = self.cfg.n_agents
        self.cfg.dynamics.num_envs = self.num_envs * self.n_agents
        self.cfg.dynamics.dt = self.physics_dt
        self.drone = create_dynamics(self.cfg.dynamics, device=self.device)
        self.drone.reset_states()

        self.target_pos_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos_rel = torch.zeros(self.num_envs, self.n_agents, 3, device=self.device)
        self.target_vel = torch.zeros(self.num_envs, self.n_agents, 3, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.n_agents, ACTION_DIM, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.internal_min_distance = torch.ones(self.num_envs, device=self.device)

        self._sample_targets()

    @property
    def target_pos(self) -> torch.Tensor:
        return self.target_pos_base.unsqueeze(1) + self.target_pos_rel

    def _setup_scene(self) -> None:
        """Use an empty Newton-only scene."""

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.view(self.num_envs, self.n_agents, ACTION_DIM).clone()

    def _apply_action(self) -> None:
        flat_actions = self.actions.view(self.num_envs * self.n_agents, ACTION_DIM)
        self.drone.apply_control(flat_actions)
        self.drone.integrate(self.physics_dt)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        vel = state["velocity"].view(self.num_envs, self.n_agents, 3)
        quat = state["orientation"].view(self.num_envs, self.n_agents, 4)

        target_vel_all = self.target_vel.reshape(self.num_envs, self.n_agents * 3).unsqueeze(1)
        target_vel_all = target_vel_all.expand(-1, self.n_agents, -1)

        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        rel_pos_all_others = torch.stack(
            [
                torch.cat((rel_pos[:, agent_id, :agent_id, :], rel_pos[:, agent_id, agent_id + 1 :, :]), dim=-2)
                for agent_id in range(self.n_agents)
            ],
            dim=1,
        ).reshape(self.num_envs, self.n_agents, -1)
        rel_vel_all_others = torch.stack(
            [
                torch.cat((rel_vel[:, agent_id, :agent_id, :], rel_vel[:, agent_id, agent_id + 1 :, :]), dim=-2)
                for agent_id in range(self.n_agents)
            ],
            dim=1,
        ).reshape(self.num_envs, self.n_agents, -1)
        related_pos = (self.target_pos.unsqueeze(1) - pos.unsqueeze(2)).reshape(
            self.num_envs,
            self.n_agents,
            -1,
        )

        obs = torch.cat(
            (
                target_vel_all,
                quat,
                vel,
                rel_pos_all_others,
                rel_vel_all_others,
                related_pos,
            ),
            dim=-1,
        )
        return {"policy": obs.view(self.num_envs, -1)}

    def _collision_metrics(self) -> torch.Tensor:
        pos = self.drone.get_state()["position"].view(self.num_envs, self.n_agents, 3)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.norm(rel_pos, dim=-1)
        mask = torch.eye(self.n_agents, device=self.device).bool().unsqueeze(0).expand(
            self.num_envs,
            -1,
            -1,
        )
        dist_masked = dist.clone()
        dist_masked[mask] = float("inf")
        self.internal_min_distance = dist_masked.min(dim=-1)[0].min(dim=-1)[0]
        return self.internal_min_distance < self.cfg.collision_distance

    def _get_rewards(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        vel = state["velocity"].view(self.num_envs, self.n_agents, 3)
        pos_dist = torch.norm(self.target_pos - pos, dim=-1)
        vel_dist = torch.norm(vel - self.target_vel, dim=-1)
        jerk_penalty = torch.norm(self.actions - self.prev_actions, dim=-1)
        collision = self._collision_metrics().float()

        weights = self.cfg.reward_weights
        reward = (
            weights.constant
            - weights.vel * vel_dist
            - weights.jerk * jerk_penalty
            - weights.pos * (1 - torch.exp(-pos_dist))
            - weights.collision * collision.unsqueeze(1)
        )
        return reward.sum(dim=-1).detach()

    def _get_loss(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        vel = state["velocity"].view(self.num_envs, self.n_agents, 3)
        pos_loss = 1.0 - torch.exp(-torch.norm(self.target_pos - pos, dim=-1))
        vel_loss = torch.norm(vel - self.target_vel, dim=-1)
        jerk_loss = torch.norm(self.actions - self.prev_actions, dim=-1)
        collision = self._collision_metrics().float()

        weights = self.cfg.reward_weights
        total_loss = (
            weights.pos * pos_loss
            + weights.vel * vel_loss
            + weights.jerk * jerk_loss
            + weights.collision * collision.unsqueeze(1)
        )
        return total_loss.sum(dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.drone.get_state()["position"].view(self.num_envs, self.n_agents, 3)
        env_half_extent = self.cfg.scene.env_spacing / 2.0
        out_of_bound = (torch.abs(pos) > env_half_extent).any(dim=-1).any(dim=-1)
        ground_collision = (pos[:, :, 2] < self.cfg.termination_height).any(dim=-1)
        collision = self._collision_metrics()
        terminated = ground_collision | out_of_bound | collision
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() == 0:
            return

        n_resets = env_ids_tensor.numel()
        self.target_pos_base[env_ids_tensor] = 0.0

        edge_length = self.cfg.collision_distance * 4
        radius = edge_length / (2 * math.sin(math.pi / self.n_agents))
        base_angles = torch.linspace(0, 2 * math.pi, self.n_agents + 1, device=self.device)[:-1]
        base_angles = base_angles.unsqueeze(0).expand(n_resets, -1)
        random_offsets = torch.rand(n_resets, 1, device=self.device) * (2 * math.pi / self.n_agents)
        angles = base_angles + random_offsets
        self.target_pos_rel[env_ids_tensor] = torch.stack(
            (
                radius * torch.cos(angles),
                radius * torch.sin(angles),
                torch.zeros_like(angles),
            ),
            dim=-1,
        )

        env_half_extent = self.cfg.scene.env_spacing / 2.0
        pos_min = -env_half_extent + RESET_MARGIN
        pos_max = env_half_extent - RESET_MARGIN
        linspace = torch.linspace(0, 1, GRID_POINTS_PER_AXIS, device=self.device).unsqueeze(0)
        axis_values = (pos_max - pos_min) * linspace + pos_min
        xyz = torch.stack(
            (
                axis_values[0].reshape(GRID_POINTS_PER_AXIS, 1, 1).expand(
                    GRID_POINTS_PER_AXIS,
                    GRID_POINTS_PER_AXIS,
                    GRID_POINTS_PER_AXIS,
                ),
                axis_values[0].reshape(1, GRID_POINTS_PER_AXIS, 1).expand(
                    GRID_POINTS_PER_AXIS,
                    GRID_POINTS_PER_AXIS,
                    GRID_POINTS_PER_AXIS,
                ),
                axis_values[0].reshape(1, 1, GRID_POINTS_PER_AXIS).expand(
                    GRID_POINTS_PER_AXIS,
                    GRID_POINTS_PER_AXIS,
                    GRID_POINTS_PER_AXIS,
                ),
            ),
            dim=-1,
        ).reshape(GRID_POINTS_PER_AXIS**3, 3)
        xyz = xyz.unsqueeze(0).expand(n_resets, -1, -1)

        random_idx = torch.stack(
            [
                torch.randperm(GRID_POINTS_PER_AXIS**3, device=self.device)
                for _ in range(n_resets)
            ],
            dim=0,
        )
        random_idx = random_idx[:, : self.n_agents].unsqueeze(-1).expand(-1, -1, 3)
        positions = xyz.gather(dim=1, index=random_idx)
        if is_pointmass_model_type(self.cfg.dynamics.model_type):
            positions[:, :, 2] = torch.clamp(positions[:, :, 2], min=MIN_POINTMASS_RESET_HEIGHT)

        flat_env_ids = (
            env_ids_tensor.unsqueeze(1) * self.n_agents
            + torch.arange(self.n_agents, device=self.device).unsqueeze(0)
        ).view(-1)
        self.drone.reset_states(positions=positions.reshape(-1, 3), env_ids=flat_env_ids)

    def _sample_targets(self, env_ids: torch.Tensor | None = None) -> None:
        """Target formation is initialized during reset."""

    def step(self, action: torch.Tensor) -> EnvStepResult:
        self._pre_physics_step(action)
        for _ in range(self.decimation):
            self._apply_action()

        self.episode_length_buf += 1
        obs_before_reset = self._get_observations()
        state_before_reset = self.drone.get_flat_state().view(self.num_envs, self.n_agents, -1)
        loss_terms = self._get_loss()
        reward = self._get_rewards()
        terminated, truncated = self._get_dones()
        reset_mask = terminated | truncated

        returned_obs = obs_before_reset
        returned_state = state_before_reset
        if reset_mask.any():
            self._reset_idx(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
            returned_obs = self._get_observations()
            returned_state = self.drone.get_flat_state().view(self.num_envs, self.n_agents, -1)

        extras = {
            "reset": reset_mask,
            "terminated": terminated,
            "truncated": truncated,
            "next_obs_before_reset": obs_before_reset["policy"].clone(),
            "next_state_before_reset": state_before_reset.clone(),
            "internal_min_distance": self.internal_min_distance.detach().clone(),
        }
        return returned_obs, returned_state, loss_terms, reward, extras

    def detach_graph(self) -> None:
        self.actions = self.actions.detach()
        self.prev_actions = self.prev_actions.detach()
        self.target_pos_base = self.target_pos_base.detach()
        self.target_pos_rel = self.target_pos_rel.detach()
        self.target_vel = self.target_vel.detach()
        self.drone.detach_graph()


def create_env(cfg: MAPCEnvCfg | None = None, **kwargs) -> MAPCEnv:
    return MAPCEnv(cfg=cfg or MAPCEnvCfg(), **kwargs)
