from typing import Any, Dict, Optional, Tuple, Sequence
import math

import torch
import numpy as np

from diffaero_newton.common.isaaclab_compat import DirectRLEnv
from diffaero_newton.configs.mapc_env_cfg import MAPCEnvCfg
from diffaero_newton.configs.dynamics_cfg import is_pointmass_model_type
from diffaero_newton.dynamics.registry import create_dynamics
from diffaero_newton.common.constants import ACTION_DIM

class MAPCEnv(DirectRLEnv):
    """Multi-Agent Position Control Environment."""

    cfg: MAPCEnvCfg

    def __init__(self, cfg: MAPCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.n_agents = self.cfg.n_agents
        
        # Initialize dynamics with num_envs * n_agents instances
        self.cfg.dynamics.num_envs = self.num_envs * self.n_agents
        self.cfg.dynamics.dt = self.physics_dt
        self.drone = create_dynamics(self.cfg.dynamics, device=self.device)
        self.drone.reset_states()

        # Goal targets for each agent [num_envs, n_agents, 3]
        self.target_pos_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos_rel = torch.zeros(self.num_envs, self.n_agents, 3, device=self.device)
        self.target_vel = torch.zeros(self.num_envs, self.n_agents, 3, device=self.device)
        
        self.actions = torch.zeros(self.num_envs, self.n_agents, ACTION_DIM, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self.internal_min_distance = torch.ones(self.num_envs, device=self.device)

        self._sample_targets()

    @property
    def target_pos(self):
        return self.target_pos_base.unsqueeze(1) + self.target_pos_rel

    def _setup_scene(self):
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.view(self.num_envs, self.n_agents, ACTION_DIM).clone()

    def _apply_action(self):
        # Flatten actions for the dynamics engine
        flat_actions = self.actions.view(self.num_envs * self.n_agents, ACTION_DIM)
        self.drone.apply_control(flat_actions)
        self.drone.integrate(self.physics_dt)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        state = self.drone.get_state()
        
        # Reshape to [num_envs, n_agents, ...]
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        vel = state["velocity"].view(self.num_envs, self.n_agents, 3)
        quat = state["orientation"].view(self.num_envs, self.n_agents, 4)
        
        target_vel_all = self.target_vel.reshape(self.num_envs, self.n_agents * 3).unsqueeze(1).expand(-1, self.n_agents, -1)
        
        # Relative positions [num_envs, n_agents, n_agents, 3]
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_pos_all_others = torch.stack(
            [torch.cat([rel_pos[:, i, :i, :], rel_pos[:, i, i+1:, :]], dim=-2) for i in range(self.n_agents)],
            dim=1
        ).reshape(self.num_envs, self.n_agents, -1)
        
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        rel_vel_all_others = torch.stack(
            [torch.cat([rel_vel[:, i, :i, :], rel_vel[:, i, i+1:, :]], dim=-2) for i in range(self.n_agents)],
            dim=1
        ).reshape(self.num_envs, self.n_agents, -1)
        
        related_pos = (self.target_pos.unsqueeze(1) - pos.unsqueeze(2)).reshape(self.num_envs, self.n_agents, -1)
        
        obs = torch.cat([
            target_vel_all,     # 3*n
            quat,               # 4
            vel,                # 3
            rel_pos_all_others, # 3*(n-1)
            rel_vel_all_others, # 3*(n-1)
            related_pos         # 3*n
        ], dim=-1)
        
        # Flatten agent dimension into feature dimension for DirectRLEnv
        # [num_envs, n_agents * (12n + 1)]
        obs_flat = obs.view(self.num_envs, -1)

        return {"policy": obs_flat}
        
    def _collision_metrics(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.norm(rel_pos, dim=-1)
        mask = torch.eye(self.n_agents, device=self.device).bool().unsqueeze(0).expand(self.num_envs, -1, -1)
        dist_masked = dist.clone()
        dist_masked[mask] = float('inf')
        self.internal_min_distance = dist_masked.min(dim=-1)[0].min(dim=-1)[0]
        collision = self.internal_min_distance < self.cfg.collision_distance
        return collision

    def _get_rewards(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        vel = state["velocity"].view(self.num_envs, self.n_agents, 3)
        
        pos_dist = torch.norm(self.target_pos - pos, dim=-1)
        pos_reward = torch.exp(-pos_dist)
        
        vel_dist = torch.norm(vel - self.target_vel, dim=-1)
        
        w = self.cfg.reward_weights
        
        jerk_penalty = torch.norm(self.actions - self.prev_actions, dim=-1)
        
        collision = self._collision_metrics().float()
        
        reward = (
            w.constant
            - w.vel * vel_dist
            - w.jerk * jerk_penalty
            - w.pos * (1 - pos_reward)
            - w.collision * collision.unsqueeze(1)
        )
        return reward.sum(dim=-1).detach()

    def _get_loss(self) -> torch.Tensor:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)
        vel = state["velocity"].view(self.num_envs, self.n_agents, 3)
        
        pos_dist = torch.norm(self.target_pos - pos, dim=-1)
        pos_loss = 1.0 - torch.exp(-pos_dist)
        
        vel_loss = torch.norm(vel - self.target_vel, dim=-1)
        jerk_loss = torch.norm(self.actions - self.prev_actions, dim=-1)
        
        collision = self._collision_metrics().float()
        
        w = self.cfg.reward_weights
        total_loss = (
            w.pos * pos_loss + 
            w.vel * vel_loss + 
            w.jerk * jerk_loss + 
            w.collision * collision.unsqueeze(1)
        )
        return total_loss.sum(dim=-1)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.drone.get_state()
        pos = state["position"].view(self.num_envs, self.n_agents, 3)

        # Bound check
        L = self.cfg.scene.env_spacing / 2.0
        out_of_bound = (torch.abs(pos) > L).any(dim=-1).any(dim=-1)
        
        # Ground
        ground_collision = (pos[:, :, 2] < self.cfg.termination_height).any(dim=-1)
        
        collision = self._collision_metrics()
        
        terminated = ground_collision | out_of_bound | collision
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            n_resets = len(env_ids)
            
            # Init formation and target
            self.target_pos_base[env_ids_tensor] = 0.0
            
            edge_length = self.cfg.collision_distance * 4
            radius = edge_length / (2 * math.sin(math.pi / self.n_agents))
            angles = torch.linspace(0, 2 * math.pi, self.n_agents + 1, device=self.device)[:-1]
            angles = angles.unsqueeze(0).expand(n_resets, -1) + torch.rand(n_resets, 1, device=self.device) * (2 * math.pi / self.n_agents)
            
            self.target_pos_rel[env_ids_tensor] = torch.stack([
                radius * torch.cos(angles),
                radius * torch.sin(angles),
                torch.zeros_like(angles)
            ], dim=-1)
            
            # Init start positions
            N = 5
            L = self.cfg.scene.env_spacing / 2.0
            p_min, p_max = -L + 0.5, L - 0.5
            linspace = torch.linspace(0, 1, N, device=self.device).unsqueeze(0)
            x_vals = y_vals = z_vals = (p_max - p_min) * linspace + p_min
            xyz = torch.stack([
                x_vals[0].reshape(N, 1, 1).expand(N, N, N),
                y_vals[0].reshape(1, N, 1).expand(N, N, N),
                z_vals[0].reshape(1, 1, N).expand(N, N, N)
            ], dim=-1).reshape(N**3, 3).unsqueeze(0).expand(n_resets, -1, -1)
            
            random_idx = torch.stack([torch.randperm(N**3, device=self.device) for _ in range(n_resets)], dim=0)
            random_idx = random_idx[:, :self.n_agents].unsqueeze(-1).expand(-1, -1, 3)
            p_new = xyz.gather(dim=1, index=random_idx)
            
            if is_pointmass_model_type(self.cfg.dynamics.model_type):
                 p_new[:, :, 2] = torch.clamp(p_new[:, :, 2], min=1.0) # Height
                 
            # Note: We need to reset the states flattened to num_envs * n_agents
            flat_env_ids = (env_ids_tensor.unsqueeze(1) * self.n_agents + torch.arange(self.n_agents, device=self.device).unsqueeze(0)).view(-1)
            p_new_flat = p_new.reshape(-1, 3)
            
            self.drone.reset_states(positions=p_new_flat, env_ids=flat_env_ids)

    def _sample_targets(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            n_resets = self.num_envs
        else:
            n_resets = len(env_ids)
        # For mapc, keeping it relatively static is fine. Target formation is set in reset.
        pass

    def step(self, action: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, terminated, truncated, extras = super().step(action)
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        
        loss_terms = self._get_loss()
        
        state_raw = self.drone.get_flat_state()
        state = state_raw.view(self.num_envs, self.n_agents, -1)
        
        return obs, state, loss_terms, reward, extras

    def detach_graph(self):
        self.actions = self.actions.detach()
        self.prev_actions = self.prev_actions.detach()
        self.target_pos_base = self.target_pos_base.detach()
        self.target_pos_rel = self.target_pos_rel.detach()
        self.target_vel = self.target_vel.detach()
        self.drone.detach_graph()

def create_env(cfg: Optional[MAPCEnvCfg] = None, **kwargs) -> MAPCEnv:
    return MAPCEnv(cfg=cfg, **kwargs)
