"""Drone environment following IsaacLab DirectRLEnv pattern.

This module provides a gymnasium-compatible environment that uses
Newton differentiable physics for quadrotor control.
"""

from typing import Any, Dict, Optional, Tuple
import math
from dataclasses import field

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig
from diffaero_newton.tasks.obstacle_manager import ObstacleManager
from diffaero_newton.tasks.reward_terms import compute_risk_loss
from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg
from diffaero_newton.common.constants import (
    DEFAULT_DT,
    MAX_EPISODE_LENGTH_S,
    STATE_DIM,
    ACTION_DIM,
    COLLISION_RADIUS,
)


class DroneEnv(gym.Env):
    """A gymnasium environment for quadrotor control with differentiable physics.

    This environment follows the IsaacLab DirectRLEnv pattern.

    Output channels:
        obs: Actor-facing observation (21D: state + goal + prev_action + nearest_obstacle_dist)
        reward: Detached RL signal for accounting
        extras: Diagnostics including obstacle metrics
    """

    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(
        self,
        cfg: Optional[DroneEnvCfg] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the drone environment.

        Args:
            cfg: Environment configuration.
            render_mode: Rendering mode.
        """
        self.cfg = cfg or DroneEnvCfg()
        self.render_mode = render_mode

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Number of environments
        self.num_envs = self.cfg.num_envs

        # Timestep
        self.physics_dt = self.cfg.sim.dt
        self.decimation = self.cfg.decimation
        self.step_dt = self.physics_dt * self.decimation

        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_max = int(self.cfg.episode_length_s / self.step_dt)

        # Initialize drone dynamics
        drone_cfg = DroneConfig(
            num_envs=self.num_envs,
            dt=self.physics_dt,
            requires_grad=False,
        )
        self.drone = Drone(drone_cfg, device=self.device)
        self.drone.reset_states()

        # State buffers
        self.obs_buf = None
        self.reward_buf = None
        self.extras = {}

        # Action buffer
        self.actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)

        # Goal (random target position)
        self.goal_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._sample_goals()

        # Previous action for action rate penalty
        self.prev_actions = torch.zeros_like(self.actions)

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

        # Configure gym spaces
        self._configure_spaces()

    def _configure_spaces(self):
        """Configure action and observation spaces."""
        # Observation: [pos(3), vel(3), quat(4), omega(3), goal(3), prev_action(4), nearest_obs_dist(1)] = 21
        # Added: nearest obstacle distance
        obs_dim = 13 + 3 + 4 + 1  # state + goal + prev_action + nearest_obs_dist

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: 4 motor thrusts normalized [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _sample_goals(self):
        """Sample random goal positions."""
        self._sample_goals_for_envs()

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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Observations and extras.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Reset drone state
        self.drone.reset_states()

        # Reset episode counter
        self.episode_length_buf.zero_()

        # Sample new goals
        self._sample_goals()

        # Reset previous actions
        self.prev_actions.zero_()

        # Get initial observation
        obs = self._get_observations()
        self.extras = {}

        return obs, self.extras

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[Dict, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Execute one environment step.

        Args:
            action: Actions to apply [num_envs, 4].

        Returns:
            Tuple of (observations, (loss, rewards), terminated, truncated, extras).
        """
        action = action.to(self.device)

        prev_actions = self.prev_actions.clone()
        self.prev_actions = action.clone()

        # Apply action through decimation steps
        for _ in range(self.decimation):
            # Convert normalized action to thrust
            thrust_action = action  # Already normalized [0, 1]
            self.drone.apply_control(thrust_action)
            self.drone.integrate(self.physics_dt)

        # Update episode length
        self.episode_length_buf += 1

        # Get observations
        obs = self._get_observations()

        # Compute rewards (detached for RL)
        reward = self._get_rewards(prev_actions)
        loss = self._get_loss(prev_actions)

        # Compute done flags
        terminated, truncated = self._get_dones()
        reset_mask = terminated | truncated
        
        # Save before-reset state and reset flag
        next_obs_before_reset = obs["policy"].clone()

        # Reset terminated environments
        if reset_mask.any():
            self._reset_idx(reset_mask.nonzero(as_tuple=True)[0])
            # Recompute observation after reset for the returning variables
            obs = self._get_observations()

        # Extras with obstacle diagnostics
        self.extras = {
            "episode": {
                "r": self.reward_buf.sum().item() / self.num_envs,
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
            "next_obs_before_reset": next_obs_before_reset
        }

        return obs, (loss, reward), terminated, truncated, self.extras

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations.

        Returns:
            Dictionary with 'policy' observation.
        """
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

    def _get_rewards(self, prev_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rewards (detached for RL).

        Returns:
            Rewards tensor [num_envs].
        """
        state = self.drone.get_state()

        # Goal tracking reward
        pos = state["position"]
        goal_dist = torch.norm(self.goal_position - pos, dim=1)
        tracking_reward = torch.exp(-goal_dist / 0.5)  # Exponential decay

        # Obstacle avoidance reward (penalize being close to obstacles)
        nearest_dist = self.obstacle_manager.compute_nearest_distances(pos)
        obstacle_penalty = -0.1 * torch.exp(-nearest_dist / 0.5)  # Higher penalty when closer

        # Velocity penalty (encourage slower movement)
        vel = state["velocity"]
        vel_penalty = -0.01 * torch.sum(vel ** 2, dim=1)

        # Orientation penalty (encourage upright)
        quat = state["orientation"]
        # Simplified: penalize non-upright (w component should be near 1)
        up_reward = quat[:, 0]  # w component

        # Action smoothness (penalize large changes)
        if prev_actions is None:
            prev_actions = torch.zeros_like(self.prev_actions)
        action_penalty = -0.001 * torch.sum((self.prev_actions - prev_actions) ** 2, dim=1)

        # Time penalty
        time_penalty = -0.01

        # Survival bonus
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

        # Store for extras
        self.reward_buf = reward.detach()

        return reward.detach()

    def _get_loss(self, prev_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute differentiable per-env loss for actor optimization."""
        state = self.drone.get_state()
        flat_state = self.drone.get_flat_state()

        pos = state["position"]
        goal_dist = torch.norm(self.goal_position - pos, dim=1)
        goal_loss = goal_dist

        risk_loss, _ = compute_risk_loss(flat_state, self.obstacle_manager.obstacles)

        vel_loss = 0.01 * torch.sum(state["velocity"] ** 2, dim=1)
        up_loss = 0.05 * (1.0 - state["orientation"][:, 0].clamp(-1.0, 1.0))

        if prev_actions is None:
            prev_actions = torch.zeros_like(self.prev_actions)
        action_rate_loss = 0.01 * torch.sum((self.prev_actions - prev_actions) ** 2, dim=1)

        return goal_loss + risk_loss + vel_loss + up_loss + action_rate_loss

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute done flags.

        Returns:
            Tuple of (terminated, truncated).
        """
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
        truncated = self.episode_length_buf >= self.episode_length_max

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Reset drone states for these envs
        positions = torch.zeros(len(env_ids), 3, device=self.device)
        positions[:, 2] = 1.0
        self.drone.reset_states(positions, env_ids=env_ids)

        # Reset episode length
        self.episode_length_buf[env_ids] = 0

        # Sample new goals for these envs
        self._sample_goals_for_envs(env_ids)

        # Reset obstacles for these envs
        self.obstacle_manager.reset(env_ids)

    def render(self):
        """Render the environment."""
        # TODO: Implement rendering
        pass

    def close(self):
        """Clean up resources."""
        pass

    def detach_graph(self):
        """Detach runtime tensors between training iterations."""
        self.prev_actions = self.prev_actions.detach()
        self.goal_position = self.goal_position.detach()
        self.drone.detach_graph()

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @num_envs.setter
    def num_envs(self, value: int):
        """Set number of environments."""
        self._num_envs = value


def create_env(cfg: Optional[DroneEnvCfg] = None, **kwargs) -> DroneEnv:
    """Create a drone environment.

    Args:
        cfg: Environment configuration.
        **kwargs: Additional arguments.

    Returns:
        Configured DroneEnv instance.
    """
    return DroneEnv(cfg=cfg, **kwargs)
