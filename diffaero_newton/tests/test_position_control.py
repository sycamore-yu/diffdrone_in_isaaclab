import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_launch import launch_app
app = launch_app()

from diffaero_newton.envs.position_control_env import create_env
from diffaero_newton.configs.position_control_env_cfg import PositionControlEnvCfg
from diffaero_newton.configs.dynamics_cfg import PointMassCfg

def test_position_control_env():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    
    cfg = PositionControlEnvCfg()
    cfg.num_envs = num_envs
    cfg.scene.num_envs = num_envs
    cfg.dynamics = PointMassCfg(num_envs=num_envs, requires_grad=True)
    
    env = create_env(cfg=cfg, device=device)
    env.reset()
    
    actions = torch.ones(num_envs, 4, device=device, requires_grad=True)
    
    try:
        obs, state, loss_terms, reward, extras = env.step(actions)
        
        loss = torch.sum(loss_terms)
        loss.backward()
        
        assert actions.grad is not None
        assert obs["policy"].shape[0] == num_envs
        assert state.shape[0] == num_envs
        
        print("Position control step and gradient flow passed.")
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")
    finally:
        if app is not None:
            app.close()

if __name__ == "__main__":
    test_position_control_env()
