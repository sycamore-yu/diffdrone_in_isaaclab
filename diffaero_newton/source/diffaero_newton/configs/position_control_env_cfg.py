import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple

from gymnasium.spaces import Box

from diffaero_newton.common.isaaclab_compat import (
    DirectRLEnvCfg,
    FeatherstoneSolverCfg,
    InteractiveSceneCfg,
    NewtonCfg,
    SimulationCfg,
    configclass,
)
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
from diffaero_newton.configs.dynamics_cfg import QuadrotorCfg

@dataclass
class PCRewardWeights:
    """Reward scaling weights based on diffaero pc.yaml."""
    constant: float = 1.0
    vel: float = 1.0       # pc.yaml quadrotor vel weight
    jerk: float = 0.001    # pc.yaml quadrotor jerk weight
    pos: float = 3.0       # pc.yaml quadrotor pos weight
    attitude: float = 0.1  # pc.yaml quadrotor attitude weight

@dataclass
class PCViewerCfg:
    """Viewer configuration for rendering."""
    ref_env_name: str = "Reference"
    pos: tuple = (0.0, -5.0, 3.0)
    lookat_pos: tuple = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/World/Camera"
    resolution: tuple = (640, 480)

@configclass
class PositionControlEnvCfg(DirectRLEnvCfg):
    """Configuration for single-agent Position Control."""
    
    num_envs: int = 256
    env_spacing: float = 2.5
    decimation: int = 4
    episode_length_s: float = MAX_EPISODE_LENGTH_S

    num_actions: int = ACTION_DIM
    num_observations: int = 16  # state(13) + goal(3)
    num_states: int = 0
    
    scene: InteractiveSceneCfg = field(default_factory=lambda: InteractiveSceneCfg(num_envs=256, env_spacing=2.5))

    observation_space: Box = field(default_factory=lambda: Box(low=-np.inf, high=np.inf, shape=(16,)))
    action_space: Box = field(default_factory=lambda: Box(low=0.0, high=1.0, shape=(ACTION_DIM,)))

    # Custom Drone parameters
    action_scale: float = 1.0
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_episode_length: int = 1500
    termination_height: float = 0.1
    
    # Task specific
    max_target_vel: float = 10.0
    min_target_vel: float = 5.0

    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=DEFAULT_DT,
            render_interval=4,
            newton_cfg=NewtonCfg(solver_cfg=FeatherstoneSolverCfg()),
        )
    )
    
    reward_weights: object = field(default_factory=lambda: PCRewardWeights())
    viewer: object = field(default_factory=lambda: PCViewerCfg())
    events: object = field(default_factory=lambda: None)
    dynamics: object = field(default_factory=lambda: QuadrotorCfg())


@configclass
class Sim2RealPositionControlEnvCfg(PositionControlEnvCfg):
    """Configuration for the square-wave Sim2Real position control variant."""

    square_size: float = 1.0
    switch_time: float = 1.0
