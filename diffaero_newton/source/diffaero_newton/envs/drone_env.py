"""Drone environment following IsaacLab DirectRLEnv pattern.

This module provides an IsaacLab DirectRLEnv-compatible environment that uses
Newton differentiable physics for quadrotor control instead of generic Omniverse physics.
"""

from typing import Any, Dict, Optional, Tuple, Sequence
import math

import torch
import numpy as np

from isaaclab.envs import DirectRLEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig
from diffaero_newton.tasks.obstacle_manager import ObstacleManager
from diffaero_newton.tasks.reward_terms import compute_risk_loss
from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg
from diffaero_newton.common.constants import ACTION_DIM


class DroneEnv(DirectRLEnv):
    """A gymnasium environment for quadrotor control with differentiable physics.

    Inherits from IsaacLab's DirectRLEnv using Newton as the actual simulation engine.
    """

    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        # We explicitly skip initializing super until we set up Newton, 
        # actually super() will setup omni.sim so we can just let it run.
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize drone dynamics (Newton)
        drone_cfg = DroneConfig(
            num_envs=self.num_envs,
            dt=self.physics_dt,
            requires_grad=False,
        )
        self.drone = Drone(drone_cfg, device=self.device)
        self.drone.reset_states()

        # Goal (random target position)
        self.goal_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._sample_goals_for_envs()

        # Previous action for action rate penalty
        self.prev_actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        
        # We also need self.actions which is already instantiated by DirectRLEnv, but lets ensure dtype
        self.actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)

        # Obstacle manager for obstacle avoidance task
        obstacle_cfg = ObstacleTaskCfg(num_obstacles=self.cfg.num_obstacles)
        self.obstacle_manager = ObstacleManager(
            num_envs=self.num_envs,
            cfg=obstacle_cfg,
            device=self.device
        )

        # Diagnostic buffers
        self.collision_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.nearest_obstacle_dist = torch.zeros(self.num_envs, device=self.device)
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """Setup the scene for the environment."""
        # Called inside super().__init__(). We don't add Omniverse prims since Newton physics 
        # is handled entirely within the Drone class.
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics."""
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self):
        """Apply actions to the simulator."""
        self.drone.apply_control(self.actions)
        self.drone.integrate(self.physics_dt)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations."""
        state = self.drone.get_state()

        # Combine state and goal
        # Position: [pos(3), vel(3), quat(4), omega(3)] = 13
        state_flat = torch.cat([
            state["position"],
            state["velocity"],
            state["orientation"],
            state["omega"],
        ], dim=1)

        # Goal-relative observation
        goal_rel = self.goal_position - state["position"]

        # Nearest obstacle distance
        nearest_dist = self.obstacle_manager.compute_nearest_distances(state["position"])

        # Concatenate: state + goal + prev_action + nearest_obstacle_dist
        obs = torch.cat([
            state_flat,
            goal_rel,
            self.prev_actions,
            nearest_dist.unsqueeze(1),
        ], dim=1)

        # Update diagnostic buffers
        self.nearest_obstacle_dist = nearest_dist.clone()
        self.goal_dist = torch.norm(goal_rel, dim=1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards (detached for RL)."""
        state = self.drone.get_state()

        # Goal tracking reward
        pos = state["position"]
        goal_dist = torch.norm(self.goal_position - pos, dim=1)
        tracking_reward = torch.exp(-goal_dist / 0.5)

        # Obstacle avoidance reward (penalize being close to obstacles)
        nearest_dist = self.obstacle_manager.compute_nearest_distances(pos)
        obstacle_penalty = -0.1 * torch.exp(-nearest_dist / 0.5)

        # Velocity penalty
        vel = state["velocity"]
        vel_penalty = -0.01 * torch.sum(vel ** 2, dim=1)

        # Orientation penalty
        quat = state["orientation"]
        up_reward = quat[:, 0]  # w component

        # Action smoothness
        action_penalty = -0.001 * torch.sum((self.prev_actions - self.actions) ** 2, dim=1)

        # Time penalty
        time_penalty = -0.01

        survival = 0.0

        # Total reward
        reward = (
            tracking_reward
            + obstacle_penalty
            + vel_penalty
            + up_reward * 0.1
            + action_penalty
            + time_penalty
            + survival
        )

        return reward.detach()

    def _get_loss(self) -> torch.Tensor:
        """Compute differentiable per-env loss for actor optimization."""
        state = self.drone.get_state()
        flat_state = self.drone.get_flat_state()

        pos = state["position"]
        goal_dist = torch.norm(self.goal_position - pos, dim=1)
        goal_loss = goal_dist

        risk_loss, _ = compute_risk_loss(flat_state, self.obstacle_manager.obstacles)

        vel_loss = 0.01 * torch.sum(state["velocity"] ** 2, dim=1)
        up_loss = 0.05 * (1.0 - state["orientation"][:, 0].clamp(-1.0, 1.0))

        action_rate_loss = 0.01 * torch.sum((self.prev_actions - self.actions) ** 2, dim=1)

        return goal_loss + risk_loss + vel_loss + up_loss + action_rate_loss

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute done flags."""
        state = self.drone.get_state()
        pos = state["position"]

        # Ground collision
        ground_collision = pos[:, 2] < self.cfg.termination_height

        # Obstacle collision
        obstacle_collision = self.obstacle_manager.check_collisions(pos)

        # Termination: collision or out of bounds
        terminated = ground_collision | obstacle_collision

        # Track collision count
        self.collision_count = obstacle_collision

        # Time out
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices."""
        super()._reset_idx(env_ids)
        
        # Reset drone states for these envs
        if len(env_ids) > 0:
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            positions = torch.zeros(len(env_ids), 3, device=self.device)
            positions[:, 2] = 1.0
            self.drone.reset_states(positions, env_ids=env_ids_tensor)

            self._sample_goals_for_envs(env_ids_tensor)
            self.obstacle_manager.reset(env_ids_tensor)

    def _sample_goals_for_envs(self, env_ids: Optional[torch.Tensor] = None):
        """Sample random goal positions for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        count = len(env_ids)
        # Sample goals in a sphere around origin
        radius = 3.0
        angles = torch.rand(count, 2, device=self.device) * 2 * math.pi
        z_vals = torch.rand(count, device=self.device) * 2 + 1  # z in [1, 3]

        self.goal_position[env_ids, 0] = radius * torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
        self.goal_position[env_ids, 1] = radius * torch.sin(angles[:, 0]) * torch.sin(angles[:, 1])
        self.goal_position[env_ids, 2] = z_vals

    def step(self, action: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute one environment step.

        Overrides DirectRLEnv to repackage the return tuple into the strict DiffAero contract.
        Returns:
            Tuple of (obs, state, loss_terms, reward, extras).
        """
        # Call DirectRLEnv step which handles decimation, obs, reward, donesk, reset
        obs, reward, terminated, truncated, extras = super().step(action)
        
        # Inject our diffaero specific keys into extras if not already there
        extras["episode"] = {
            "r": reward.sum().item() / self.num_envs,
            "l": self.episode_length_buf.float().mean().item(),
        }
        extras["obstacles"] = {
            "nearest_dist": self.nearest_obstacle_dist.mean().item(),
            "goal_dist": self.goal_dist.mean().item(),
            "collisions": self.collision_count.sum().item(),
        }
        extras["terminated"] = terminated
        extras["truncated"] = truncated

        # Compute loss_terms explicitly
        loss_terms = self._get_loss()

        # The explicit strict contract: obs, state, loss_terms, reward, extras
        state = self.drone.get_flat_state()
        
        # Return 5-tuple
        return obs, state, loss_terms, reward, extras

    def detach_graph(self):
        """Detach runtime tensors between training iterations."""
        self.prev_actions = self.prev_actions.detach()
        self.actions = self.actions.detach()
        self.goal_position = self.goal_position.detach()
        self.drone.detach_graph()


def create_env(cfg: Optional[DroneEnvCfg] = None, **kwargs) -> DroneEnv:
    """Create a drone environment."""
    return DroneEnv(cfg=cfg, **kwargs)
