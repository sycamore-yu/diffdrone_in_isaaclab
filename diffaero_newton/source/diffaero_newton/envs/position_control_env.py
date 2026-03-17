from __future__ import annotations

from collections.abc import Sequence

import torch

from diffaero_newton.common.direct_rl_shim import DirectRLEnv
from diffaero_newton.configs.dynamics_cfg import is_pointmass_model_type
from diffaero_newton.configs.position_control_env_cfg import (
    PositionControlEnvCfg,
    Sim2RealPositionControlEnvCfg,
)
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


class PositionControlEnv(DirectRLEnv):
    """Position-control environment aligned with DiffAero's point-control task."""

    cfg: PositionControlEnvCfg

    def __init__(
        self,
        cfg: PositionControlEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.dynamics.num_envs = self.num_envs
        self.cfg.dynamics.dt = self.physics_dt
        self.drone = create_dynamics(self.cfg.dynamics, device=self.device)
        self.drone.reset_states()

        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._sample_targets()

        action_dim = self.cfg.action_space.shape[0]
        self.actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

    def _setup_scene(self) -> None:
        """Use an empty Newton-only scene."""

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.drone.apply_control(self.actions)
        self.drone.integrate(self.physics_dt)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self.drone.get_state()
        state_flat = torch.cat(
            (
                state["position"],
                state["velocity"],
                state["orientation"],
                state["omega"],
            ),
            dim=1,
        )
        goal_rel = self.target_pos - state["position"]
        return {"policy": torch.cat((state_flat, goal_rel), dim=1)}

    def _get_rewards(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos_dist = torch.norm(self.target_pos - state["position"], dim=-1)
        vel_dist = torch.norm(state["velocity"] - self.target_vel, dim=-1)
        jerk_penalty = torch.norm(self.actions - self.prev_actions, dim=-1)

        pos_reward = torch.exp(-pos_dist)
        attitude_reward = torch.ones_like(pos_dist)
        if self.cfg.dynamics.model_type == "quadrotor":
            attitude_reward = state["orientation"][:, 0]

        weights = self.cfg.reward_weights
        reward = (
            weights.constant
            - weights.vel * vel_dist
            - weights.jerk * jerk_penalty
            - weights.pos * (1 - pos_reward)
            - weights.attitude * (1 - attitude_reward)
        )
        return reward.detach()

    def _get_loss(self) -> torch.Tensor:
        """Return the differentiable training loss for the current state."""

        state = self.drone.get_state()
        pos_dist = torch.norm(self.target_pos - state["position"], dim=-1)
        vel_loss = torch.norm(state["velocity"] - self.target_vel, dim=-1)
        jerk_loss = torch.norm(self.actions - self.prev_actions, dim=-1)
        pos_loss = 1.0 - torch.exp(-pos_dist)

        attitude_loss = torch.zeros_like(pos_dist)
        if self.cfg.dynamics.model_type == "quadrotor":
            attitude_loss = 1.0 - state["orientation"][:, 0].clamp(-1.0, 1.0)

        weights = self.cfg.reward_weights
        return (
            weights.pos * pos_loss
            + weights.vel * vel_loss
            + weights.jerk * jerk_loss
            + weights.attitude * attitude_loss
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.drone.get_state()["position"]
        env_half_extent = self.cfg.scene.env_spacing / 2.0
        out_of_bound = (torch.abs(pos) > env_half_extent).any(dim=-1)
        ground_collision = pos[:, 2] < self.cfg.termination_height
        terminated = ground_collision | out_of_bound
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() == 0:
            return

        env_half_extent = self.cfg.scene.env_spacing / 2.0
        pos_min = -env_half_extent + RESET_MARGIN
        pos_max = env_half_extent - RESET_MARGIN
        positions = torch.rand((env_ids_tensor.numel(), 3), device=self.device) * (pos_max - pos_min) + pos_min
        if is_pointmass_model_type(self.cfg.dynamics.model_type):
            positions[:, 2] = torch.clamp(positions[:, 2], min=MIN_POINTMASS_RESET_HEIGHT)

        self.drone.reset_states(positions=positions, env_ids=env_ids_tensor)
        self._sample_targets(env_ids_tensor)

    def _sample_targets(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.target_pos[env_ids] = 0.0
        self.target_vel[env_ids] = 0.0

    def step(self, action: torch.Tensor) -> EnvStepResult:
        obs, reward, terminated, truncated, extras = super().step(action)
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        loss_terms = self._get_loss()
        state = self.drone.get_flat_state()
        return obs, state, loss_terms, reward, extras

    def detach_graph(self) -> None:
        self.actions = self.actions.detach()
        self.prev_actions = self.prev_actions.detach()
        self.target_pos = self.target_pos.detach()
        self.target_vel = self.target_vel.detach()
        self.drone.detach_graph()


def create_env(cfg: PositionControlEnvCfg | None = None, **kwargs) -> PositionControlEnv:
    return PositionControlEnv(cfg=cfg or PositionControlEnvCfg(), **kwargs)


class Sim2RealPositionControlEnv(PositionControlEnv):
    """Square-target position-control variant inspired by DiffAero's sim-to-real task."""

    cfg: Sim2RealPositionControlEnvCfg

    def __init__(
        self,
        cfg: Sim2RealPositionControlEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self.square_positions: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)
        self.square_positions = torch.tensor(
            [
                [cfg.square_size, -cfg.square_size, 0.0],
                [-cfg.square_size, -cfg.square_size, 0.0],
                [-cfg.square_size, cfg.square_size, 0.0],
                [cfg.square_size, cfg.square_size, 0.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.update_target()

    def update_target(self) -> None:
        assert self.square_positions is not None
        time_in_episode = self.episode_length_buf.float() * self.step_dt
        target_index = torch.floor(time_in_episode / self.cfg.switch_time).long()
        self.target_pos = self.square_positions[target_index % self.square_positions.shape[0]]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.drone.get_state()["position"]
        pos_range = torch.full_like(pos, fill_value=self.cfg.square_size * 2.0)
        out_of_bound = (torch.abs(pos) > pos_range).any(dim=-1)
        truncated = self.episode_length_buf >= self.max_episode_length
        return out_of_bound, truncated

    def _sample_targets(self, env_ids: torch.Tensor | None = None) -> None:
        if self.square_positions is None:
            super()._sample_targets(env_ids)
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.update_target()
        self.target_vel[env_ids] = 0.0

    def step(self, action: torch.Tensor) -> EnvStepResult:
        self.update_target()
        return super().step(action)


def create_sim2real_env(
    cfg: Sim2RealPositionControlEnvCfg | None = None,
    **kwargs,
) -> Sim2RealPositionControlEnv:
    return Sim2RealPositionControlEnv(cfg=cfg or Sim2RealPositionControlEnvCfg(), **kwargs)
