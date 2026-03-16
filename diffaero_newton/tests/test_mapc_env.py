import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_compat import launch_app
app = launch_app()

from diffaero_newton.envs.mapc_env import create_env
from diffaero_newton.configs.mapc_env_cfg import MAPCEnvCfg
from diffaero_newton.configs.dynamics_cfg import PointMassCfg

def test_mapc_env():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    n_agents = 4
    
    cfg = MAPCEnvCfg()
    cfg.num_envs = num_envs
    cfg.n_agents = n_agents
    cfg.scene.num_envs = num_envs
    cfg.dynamics = PointMassCfg(num_envs=num_envs * n_agents, requires_grad=True)
    
    env = create_env(cfg=cfg, device=device)
    env.reset()
    
    actions = torch.ones(num_envs, cfg.num_actions, device=device, requires_grad=True)
    
    try:
        obs, state, loss_terms, reward, extras = env.step(actions)
        
        loss = torch.sum(loss_terms)
        loss.backward()
        
        assert actions.grad is not None
        assert obs["policy"].shape[0] == num_envs
        assert obs["policy"].shape[1] == cfg.num_observations
        assert state.shape[0] == num_envs
        assert state.shape[1] == n_agents
        
        print("MAPC step and gradient flow passed.")
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")
    finally:
        if app is not None:
            app.close()

if __name__ == "__main__":
    test_mapc_env()
