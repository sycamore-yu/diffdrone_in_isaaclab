"""Physical constants and shape definitions for quadrotor simulation."""

import torch
from typing import Final

# ==============================================================================
# Physical Constants (SI Units)
# ==============================================================================

# Gravity
GRAVITY: Final[float] = 9.81  # m/s^2

# Air density (sea level, 15°C)
AIR_DENSITY: Final[float] = 1.225  # kg/m^3

# Mass properties (typical quadrotor)
QUADROTOR_MASS: Final[float] = 1.0  # kg

# Moment of inertia (assuming symmetric X-frame)
IXX: Final[float] = 0.01  # kg⋅m²
IYY: Final[float] = 0.01  # kg⋅m²
IZZ: Final[float] = 0.02  # kg⋅m²

# Arm length (center to motor)
ARM_LENGTH: Final[float] = 0.2  # m

# Motor/rotor constants
MOTOR_TIME_CONSTANT: Final[float] = 0.05  # s (motor dynamics)
MOTOR_ROLL_MAX: Final[float] = 100.0  # rad/s (max motor speed)
KT: Final[float] = 1.0e-5  # thrust coefficient (N/(rad/s)²)
KD: Final[float] = 1.0e-7  # drag coefficient (N⋅m/(rad/s)²)

# ==============================================================================
# Shape Constants
# ==============================================================================

# State dimensions
STATE_DIM: Final[int] = 13  # [pos(3), quat(4), vel(3), omega(3)]
ACTION_DIM: Final[int] = 4  # 4 motor thrusts

# Observation dimensions
OBS_DIM_BASE: Final[int] = 13  # position, velocity, orientation
OBS_DIM_GOAL: Final[int] = 3  # relative goal position
OBS_DIM_OBSTACLE: Final[int] = 4  # [pos(3), radius(1)] per obstacle
MAX_OBSTACLES: Final[int] = 20

# ==============================================================================
# Simulation Parameters
# ==============================================================================

# Default simulation timestep
DEFAULT_DT: Final[float] = 0.01  # 100 Hz
CONTROL_DT: Final[float] = 0.02  # 50 Hz (control frequency)

# Rollout horizons
DEFAULT_ROLLOUT_HORIZON: Final[int] = 10  # short horizon for SHAC
MAX_EPISODE_LENGTH_S: Final[float] = 30.0  # seconds

# ==============================================================================
# Space Bounds
# ==============================================================================

# Position bounds (meters)
POS_BOUNDS_LOW = torch.tensor([-5.0, -5.0, 0.0])
POS_BOUNDS_HIGH = torch.tensor([5.0, 5.0, 10.0])

# Velocity bounds (m/s)
VEL_BOUNDS_LOW = torch.tensor([-10.0, -10.0, -5.0])
VEL_BOUNDS_HIGH = torch.tensor([10.0, 10.0, 5.0])

# Action bounds (motor thrusts normalized [0, 1])
ACTION_BOUNDS_LOW = torch.zeros(4)
ACTION_BOUNDS_HIGH = torch.ones(4)

# ==============================================================================
# Collision & Safety
# ==============================================================================

# Collision parameters
COLLISION_RADIUS: Final[float] = 0.3  # drone collision radius (m)
OBSTACLE_COLLISION_RADIUS: Final[float] = 0.5  # default obstacle radius

# Risk thresholds
RISK_DISTANCE_CRITICAL: Final[float] = 0.5  # critical proximity (m)
RISK_DISTANCE_WARNING: Final[float] = 1.0  # warning proximity (m)

# Reward scaling
GOAL_REWARD_SCALE: Final[float] = 10.0
COLLISION_PENALTY: Final[float] = 100.0
TIME_PENALTY: Final[float] = 0.1
