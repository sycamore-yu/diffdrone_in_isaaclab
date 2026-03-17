import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_launch import launch_app
app = launch_app()

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from diffaero_newton.envs.sensors import create_sensor
from diffaero_newton.tasks.obstacle_manager import ObstacleManager
from dataclasses import dataclass

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

from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg

def test_sensors():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    
    obst_cfg = ObstacleTaskCfg()
    obst_cfg.num_obstacles = 10
    obstacle_manager = ObstacleManager(num_envs=num_envs, cfg=obst_cfg, device=device)
    
    pos = torch.zeros(num_envs, 3, device=device, requires_grad=True)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    
    # 1. Camera Test
    cam_cfg = CameraCfg()
    camera = create_sensor(cam_cfg, num_envs, device)
    
    depth = camera(obstacle_manager, pos, quat)
    assert depth.shape == (num_envs, 32, 32)
    
    loss = depth.sum()
    loss.backward()
    assert pos.grad is not None
    pos.grad.zero_()
    
    # 2. Lidar Test
    lidar_cfg = LidarCfg()
    lidar = create_sensor(lidar_cfg, num_envs, device)
    
    depth_lidar = lidar(obstacle_manager, pos, quat)
    assert depth_lidar.shape == (num_envs, 16, 16)
    
    loss = depth_lidar.sum()
    loss.backward()
    assert pos.grad is not None
    pos.grad.zero_()
    
    try:
        # 3. Relpos Test
        relpos_cfg = RelposCfg()
        relpos = create_sensor(relpos_cfg, num_envs, device)
        
        rel_pos = relpos(obstacle_manager, pos, quat)
        assert rel_pos.shape == (num_envs, 10, 3)
        
        loss = rel_pos.sum()
        loss.backward()
        assert pos.grad is not None
        
        print("All sensor tests passed.")
    finally:
        if app is not None:
            app.close()

if __name__ == "__main__":
    test_sensors()
