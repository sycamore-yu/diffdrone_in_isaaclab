import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple

from gymnasium.spaces import Box

from diffaero_newton.common.constants import (
    DEFAULT_DT,
    ACTION_DIM,
    MAX_EPISODE_LENGTH_S,
)
from diffaero_newton.common.isaaclab_compat import (
    DirectRLEnvCfg,
    FeatherstoneSolverCfg,
    InteractiveSceneCfg,
    NewtonCfg,
    SimulationCfg,
    configclass,
)
from diffaero_newton.configs.dynamics_cfg import PointMassCfg

@dataclass
class MAPCRewardWeights:
    """Reward scaling weights based on diffaero mapc.yaml."""
    constant: float = 4.0
    vel: float = 1.0       # pointmass
    jerk: float = 0.005    # pointmass
    pos: float = 1.0       # pointmass 
    collision: float = 16.0 # pointmass

@dataclass
class MAPCViewerCfg:
    """Viewer configuration for rendering."""
    ref_env_name: str = "Reference"
    pos: tuple = (0.0, -5.0, 3.0)
    lookat_pos: tuple = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/World/Camera"
    resolution: tuple = (640, 480)

@configclass
class MAPCEnvCfg(DirectRLEnvCfg):
    """Configuration for Multi-Agent Position Control."""
    
    num_envs: int = 256
    n_agents: int = 4
    
    env_spacing: float = 2.5
    decimation: int = 4
    episode_length_s: float = MAX_EPISODE_LENGTH_S

    num_actions: int = 0
    num_observations: int = 0
    num_states: int = 0
    
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=256, env_spacing=2.5)
    )

    observation_space: Box = field(default_factory=lambda: Box(low=-np.inf, high=np.inf, shape=(0,)))
    action_space: Box = field(default_factory=lambda: Box(low=0.0, high=1.0, shape=(0,)))

    # Custom Drone parameters
    action_scale: float = 1.0
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_episode_length: int = 1500
    termination_height: float = 0.1
    collision_distance: float = 0.3
    
    # Task specific
    max_target_vel: float = 5.0
    min_target_vel: float = 5.0

    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=DEFAULT_DT,
            render_interval=4,
            newton_cfg=NewtonCfg(solver_cfg=FeatherstoneSolverCfg()),
        )
    )
    
    reward_weights: object = field(default_factory=lambda: MAPCRewardWeights())
    viewer: object = field(default_factory=lambda: MAPCViewerCfg())
    events: object = field(default_factory=lambda: None)
    dynamics: object = field(default_factory=lambda: PointMassCfg())

    def __post_init__(self):
        # Dynamically set vector sizes based on n_agents
        # Observation per agent: target_vel(3*n) + quat(4) + vel(3) + rel_pos_others(3*(n-1)) + rel_vel_others(3*(n-1)) + rel_targets(3*n)
        # = 3n + 7 + 6n - 6 + 3n = 12n + 1. Whole env obs = n * (12n + 1)
        self.num_actions = ACTION_DIM * self.n_agents
        self.num_observations = self.n_agents * (12 * self.n_agents + 1)
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_actions,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))
