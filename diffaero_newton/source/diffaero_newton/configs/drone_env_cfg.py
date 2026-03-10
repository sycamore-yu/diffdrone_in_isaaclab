"""Environment configuration using the configclass pattern."""

from dataclasses import dataclass, field
from typing import Tuple

import torch

from diffaero_newton.common.constants import (
    DEFAULT_DT,
    DEFAULT_ROLLOUT_HORIZON,
    MAX_EPISODE_LENGTH_S,
    ACTION_DIM,
    STATE_DIM,
    POS_BOUNDS_LOW,
    POS_BOUNDS_HIGH,
    ACTION_BOUNDS_LOW,
    ACTION_BOUNDS_HIGH,
)


# Use dataclass as configclass (IsaacLab pattern without omni.isaac dependency)
class DroneEnvCfg:
    """Configuration for the drone environment.

    This config follows the IsaacLab pattern using @configclass for
    automatic config validation and UI generation.
    """

    def __init__(self, num_envs: int = 256, episode_length_s: float = MAX_EPISODE_LENGTH_S, **kwargs):
        """Initialize config with optional overrides."""
        self.num_envs = num_envs
        self.episode_length_s = episode_length_s
        self.env_spacing = 2.5
        self.sim = self._default_sim_cfg()
        self.decimation = 4
        self.observation_space = 18
        self.action_space = 4
        self.state_space = 13
        self.rollout_horizon = 10
        self.initial_position = (0.0, 0.0, 1.0)
        self.initial_velocity = (0.0, 0.0, 0.0)
        self.action_scale = 1.0
        self.reward_weights = self._default_reward_weights()
        self.max_episode_length = 1500
        self.termination_height = 0.1
        self.viewer = self._default_viewer_cfg()
        self.events = None

    # Environment dimensions
    num_envs: int = 256
    env_spacing: float = 2.5

    # Simulation parameters
    sim: object = field(default_factory=lambda: DroneEnvCfg._default_sim_cfg())
    decimation: int = 4  # Physics steps per environment step
    episode_length_s: float = MAX_EPISODE_LENGTH_S

    # Observation and action spaces
    observation_space: int = 18  # Base obs + goal + actions
    action_space: int = ACTION_DIM
    state_space: int = STATE_DIM

    # Rollout for SHAC
    rollout_horizon: int = DEFAULT_ROLLOUT_HORIZON

    # Initial state
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Action space bounds
    action_scale: float = 1.0

    # Reward weights
    reward_weights: object = field(default_factory=lambda: DroneEnvCfg._default_reward_weights())

    # Termination conditions
    max_episode_length: int = 1500  # steps
    termination_height: float = 0.1

    # Camera (for rendering)
    viewer: object = field(default_factory=lambda: DroneEnvCfg._default_viewer_cfg())

    # Events
    events: object = field(default_factory=lambda: None)

    @staticmethod
    def _default_sim_cfg():
        """Default simulation configuration."""
        return SimCfg()

    @staticmethod
    def _default_reward_weights():
        """Default reward weights."""
        return RewardWeights()

    @staticmethod
    def _default_viewer_cfg():
        """Default viewer configuration."""
        return ViewerCfg()


@dataclass
class SimCfg:
    """Simulation configuration."""

    dt: float = DEFAULT_DT
    substeps: int = 1
    use_gpu: bool = True
    enable_cuda_kernel: bool = True
    render_interval: int = 4


@dataclass
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
