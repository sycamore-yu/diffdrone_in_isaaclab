"""Sensor configuration for obstacle avoidance observation modes."""

from dataclasses import dataclass


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
