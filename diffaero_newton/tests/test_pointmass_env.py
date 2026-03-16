import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_compat import launch_app
app = launch_app()

from diffaero_newton.envs.drone_env import create_env
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.configs.dynamics_cfg import ContinuousPointMassCfg, DiscretePointMassCfg, PointMassCfg

@pytest.mark.parametrize("cfg_cls", [PointMassCfg, ContinuousPointMassCfg, DiscretePointMassCfg])
def test_pointmass_env_step(cfg_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 2
    
    cfg = DroneEnvCfg()
    cfg.num_envs = num_envs
    cfg.scene.num_envs = num_envs
    cfg.dynamics = cfg_cls(num_envs=num_envs, requires_grad=True)
    
    env = create_env(cfg=cfg, device=device)
    
    # Needs a warm reset
    env.reset()
    
    actions = torch.ones(num_envs, 3, device=device, requires_grad=True)
    
    # DroneEnv currently assumes ACTION_DIM=4 for quadrotor.
    # We must pad it to 4 to pass DroneEnv checks, and PointMass only reads the first 3.
    padded_actions = torch.cat([actions, torch.zeros(num_envs, 1, device=device, requires_grad=True)], dim=-1)
    
    try:
        obs, state, loss_terms, reward, extras = env.step(padded_actions)
        
        # Test differentiability
        loss = torch.sum(loss_terms)
        loss.backward()
        
        assert actions.grad is not None
        print("PointMass env step and gradient backward passed.")
        
    except Exception as e:
        pytest.fail(f"Pointmass env test failed with error: {e}")
    finally:
        app.close()

if __name__ == "__main__":
    test_pointmass_env_step()
