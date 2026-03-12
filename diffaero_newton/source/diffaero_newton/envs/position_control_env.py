from typing import Any, Dict, Optional, Tuple, Sequence
import math

import torch
import numpy as np

from isaaclab.envs import DirectRLEnv
from diffaero_newton.configs.position_control_env_cfg import PositionControlEnvCfg
from diffaero_newton.dynamics.registry import create_dynamics
from diffaero_newton.common.constants import ACTION_DIM


class PositionControlEnv(DirectRLEnv):
    """Position Control Environment mapping diffaero point_control.py to IsaacLab DirectRL."""

    cfg: PositionControlEnvCfg

    def __init__(self, cfg: PositionControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize dynamics
        self.cfg.dynamics.num_envs = self.num_envs
        self.cfg.dynamics.dt = self.physics_dt
        self.drone = create_dynamics(self.cfg.dynamics, device=self.device)
        self.drone.reset_states()

        # Goal targets
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._sample_targets()

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space.shape[0], device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

    def _setup_scene(self):
        # Empty scene for Newton physics
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self):
        self.drone.apply_control(self.actions)
        self.drone.integrate(self.physics_dt)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        state = self.drone.get_state()
        
        state_flat = torch.cat([
            state["position"],
            state["velocity"],
            state["orientation"],
            state["omega"],
        ], dim=1)

        goal_rel = self.target_pos - state["position"]

        obs = torch.cat([
            state_flat,
            goal_rel,
        ], dim=1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos_dist = torch.norm(self.target_pos - state["position"], dim=-1)
        pos_reward = torch.exp(-pos_dist)
        
        # Following reference pc.yaml structure
        vel_dist = torch.norm(state["velocity"] - self.target_vel, dim=-1)

        w = self.cfg.reward_weights
        
        attitude_reward = 0.0
        if self.cfg.dynamics.model_type == "quadrotor":
            # Simplify attitude reward using up vector approx
            attitude_reward = state["orientation"][:, 0]
        
        jerk_penalty = torch.norm(self.actions - self.prev_actions, dim=-1)

        reward = (
            w.constant
            - w.vel * vel_dist
            - w.jerk * jerk_penalty
            - w.pos * (1 - pos_reward)
            - w.attitude * (1 - attitude_reward)
        )
        return reward.detach()

    def _get_loss(self) -> torch.Tensor:
        """Differentiable loss calculation."""
        state = self.drone.get_state()
        pos_dist = torch.norm(self.target_pos - state["position"], dim=-1)
        pos_loss = 1.0 - torch.exp(-pos_dist)
        
        vel_loss = torch.norm(state["velocity"] - self.target_vel, dim=-1)
        jerk_loss = torch.norm(self.actions - self.prev_actions, dim=-1)
        
        attitude_loss = 0.0
        if self.cfg.dynamics.model_type == "quadrotor":
            attitude_loss = 1.0 - state["orientation"][:, 0].clamp(-1.0, 1.0)
            
        w = self.cfg.reward_weights
        total_loss = (
            w.pos * pos_loss + 
            w.vel * vel_loss + 
            w.jerk * jerk_loss + 
            w.attitude * attitude_loss
        )
        return total_loss

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.drone.get_state()
        pos = state["position"]

        # Bound check
        L = self.cfg.scene.env_spacing / 2.0
        out_of_bound = (torch.abs(pos) > L).any(dim=-1)
        
        # Ground
        ground_collision = pos[:, 2] < self.cfg.termination_height
        
        terminated = ground_collision | out_of_bound
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            
            L = self.cfg.scene.env_spacing / 2.0
            p_min, p_max = -L + 0.5, L - 0.5
            p_new = torch.rand((len(env_ids), 3), device=self.device) * (p_max - p_min) + p_min
            if self.cfg.dynamics.model_type == "pointmass":
                p_new[:, 2] = torch.clamp(p_new[:, 2], min=1.0) # Height

            self.drone.reset_states(positions=p_new, env_ids=env_ids_tensor)
            self._sample_targets(env_ids_tensor)

    def _sample_targets(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.target_pos[env_ids] = 0.0  # Origin target for pc
        
    def step(self, action: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, terminated, truncated, extras = super().step(action)
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        
        loss_terms = self._get_loss()
        state = self.drone.get_flat_state()
        
        return obs, state, loss_terms, reward, extras

    def detach_graph(self):
        self.actions = self.actions.detach()
        self.prev_actions = self.prev_actions.detach()
        self.target_pos = self.target_pos.detach()
        self.drone.detach_graph()

def create_env(cfg: Optional[PositionControlEnvCfg] = None, **kwargs) -> PositionControlEnv:
    return PositionControlEnv(cfg=cfg, **kwargs)
