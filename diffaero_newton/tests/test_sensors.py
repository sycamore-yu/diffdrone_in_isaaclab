from dataclasses import dataclass

import pytest
import torch

from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg
from diffaero_newton.envs.sensors import create_sensor
from diffaero_newton.tasks.obstacle_manager import ObstacleManager


pytestmark = pytest.mark.usefixtures("isaaclab_app")


@dataclass
class CameraCfg:
    name: str = "camera"
    height: int = 32
    width: int = 32
    horizontal_fov: float = 90.0
    max_dist: float = 10.0


@dataclass
class LidarCfg:
    name: str = "lidar"
    n_rays_vertical: int = 16
    n_rays_horizontal: int = 16
    depression_angle: float = -15.0
    elevation_angle: float = 15.0
    max_dist: float = 10.0


@dataclass
class RelposCfg:
    name: str = "relpos"
    n_obstacles: int = 10
    ceiling: bool = False
    walls: bool = False


def test_sensors_support_differentiable_queries() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4

    obstacle_cfg = ObstacleTaskCfg()
    obstacle_cfg.num_obstacles = 10
    obstacle_manager = ObstacleManager(num_envs=num_envs, cfg=obstacle_cfg, device=device)

    pos = torch.zeros(num_envs, 3, device=device, requires_grad=True)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    camera = create_sensor(CameraCfg(), num_envs, device)
    camera_depth = camera(obstacle_manager, pos, quat)
    assert camera_depth.shape == (num_envs, 32, 32)
    camera_depth.sum().backward()
    assert pos.grad is not None
    pos.grad.zero_()

    lidar = create_sensor(LidarCfg(), num_envs, device)
    lidar_depth = lidar(obstacle_manager, pos, quat)
    assert lidar_depth.shape == (num_envs, 16, 16)
    lidar_depth.sum().backward()
    assert pos.grad is not None
    pos.grad.zero_()

    relpos = create_sensor(RelposCfg(), num_envs, device)
    rel_pos = relpos(obstacle_manager, pos, quat)
    assert rel_pos.shape == (num_envs, 10, 3)
    rel_pos.sum().backward()
    assert pos.grad is not None
