"""Sensor configuration for obstacle avoidance observation modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SensorCfg:
    """Base sensor configuration."""
    name: str = "relpos"  # "relpos", "camera", "lidar"
    max_dist: float = 10.0


@dataclass
class CameraSensorCfg(SensorCfg):
    """Camera sensor configuration."""
    name: str = "camera"
    height: int = 32
    width: int = 32
    horizontal_fov: float = 90.0


@dataclass
class LidarSensorCfg(SensorCfg):
    """LiDAR sensor configuration."""
    name: str = "lidar"
    n_rays_vertical: int = 16
    n_rays_horizontal: int = 16
    depression_angle: float = -15.0
    elevation_angle: float = 15.0


@dataclass
class RelposSensorCfg(SensorCfg):
    """Relative position sensor configuration."""
    name: str = "relpos"
    n_obstacles: int = 10
    ceiling: bool = False
    walls: bool = False


def sensor_observation_shape(cfg: SensorCfg) -> Tuple[int, ...]:
    """Return the flattened observation shape emitted by a sensor config."""

    if isinstance(cfg, CameraSensorCfg):
        return (cfg.height * cfg.width,)
    if isinstance(cfg, LidarSensorCfg):
        return (cfg.n_rays_vertical * cfg.n_rays_horizontal,)
    if isinstance(cfg, RelposSensorCfg):
        rows = cfg.n_obstacles + int(cfg.ceiling) + 4 * int(cfg.walls)
        return (rows * 3,)
    raise TypeError(f"Unsupported sensor config type: {type(cfg).__name__}")


def sensor_observation_dim(cfg: SensorCfg) -> int:
    """Return the flattened observation dimension emitted by a sensor config."""

    return sensor_observation_shape(cfg)[0]


def build_sensor_cfg(name: str, num_obstacles: int) -> SensorCfg:
    """Build a supported sensor config by CLI/registry name."""

    name = name.lower()
    if name == "camera":
        return CameraSensorCfg()
    if name == "lidar":
        return LidarSensorCfg()
    if name == "relpos":
        return RelposSensorCfg(n_obstacles=num_obstacles)
    raise ValueError(f"Unknown sensor: {name}. Available: camera, lidar, relpos")
