"""Configuration package for diffaero_newton.

Avoid eager imports here so lightweight CLI paths such as `train.py --list`
do not pull in IsaacLab modules during package initialization.
"""

__all__ = [
    "drone_env_cfg",
    "dynamics_cfg",
    "mapc_env_cfg",
    "obstacle_env_cfg",
    "obstacle_task_cfg",
    "position_control_env_cfg",
    "racing_env_cfg",
    "sensor_cfg",
    "training_cfg",
]
