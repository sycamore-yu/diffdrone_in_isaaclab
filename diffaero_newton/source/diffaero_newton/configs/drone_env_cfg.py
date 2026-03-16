"""Environment configuration using the configclass pattern."""

from dataclasses import dataclass, field
from typing import Tuple

from diffaero_newton.common.constants import (
    DEFAULT_DT,
    DEFAULT_ROLLOUT_HORIZON,
    MAX_EPISODE_LENGTH_S,
    ACTION_DIM,
)
from diffaero_newton.common.isaaclab_compat import (
    DirectRLEnvCfg,
    FeatherstoneSolverCfg,
    InteractiveSceneCfg,
    NewtonCfg,
    SimulationCfg,
    configclass,
)

import numpy as np
import torch
from gymnasium.spaces import Box

@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    """Configuration for the drone environment.

    This config follows the IsaacLab pattern using @configclass for
    automatic config validation and UI generation.
    """
    
    # Environment config
    num_envs: int = 256
    num_obstacles: int = 5
    env_spacing: float = 2.5
    decimation: int = 4
    episode_length_s: float = MAX_EPISODE_LENGTH_S

    num_actions: int = ACTION_DIM
    num_observations: int = 21  # state(13) + goal(3) + prev_act(4) + nearest_obs(1)
    num_states: int = 0
    
    # We need scene for DirectRLEnvCfg contract even if unused directly by Newton
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=256, env_spacing=2.5)
    )

    # Required spaces by DirectRLEnvCfg validation
    observation_space: Box = field(default_factory=lambda: Box(low=-np.inf, high=np.inf, shape=(21,)))
    action_space: Box = field(default_factory=lambda: Box(low=0.0, high=1.0, shape=(ACTION_DIM,)))

    # Custom Drone parameters
    action_scale: float = 1.0
    rollout_horizon: int = DEFAULT_ROLLOUT_HORIZON
    differentiable_dynamics: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_episode_length: int = 1500
    termination_height: float = 0.1

    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=DEFAULT_DT,
            render_interval=4,
            newton_cfg=NewtonCfg(solver_cfg=FeatherstoneSolverCfg()),
        )
    )
    reward_weights: object = field(default_factory=lambda: RewardWeights())
    viewer: object = field(default_factory=lambda: ViewerCfg())
    events: object = field(default_factory=lambda: None)



class RewardWeights:
    """Reward scaling weights."""

    tracking_sigma: float = 0.25
    orientation: float = 0.05
    lin_vel: float = 0.05
    ang_vel: float = 0.0
    action_rate: float = 0.0
    collision: float = 1.0
    time_penalty: float = 0.01
    survival: float = 0.0


@dataclass
class ViewerCfg:
    """Viewer configuration for rendering."""

    ref_env_name: str = "Reference"
    pos: tuple = (0.0, -5.0, 3.0)
    lookat_pos: tuple = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/World/FalseKart/width_offset/Camera"
    resolution: tuple = (640, 480)
