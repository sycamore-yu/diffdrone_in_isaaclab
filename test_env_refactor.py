import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test App")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

if __name__ == "__main__":
    cfg = DroneEnvCfg()
    env = DroneEnv(cfg)
    
    # Check return type of reset
    res = env.reset()
    print("Reset returned:", type(res), len(res))
    
    # Check return type of step
    action = torch.zeros(cfg.num_envs, 4, device=env.device)
    step_res = env.step(action)
    print("Step returned:", type(step_res), len(step_res))
    
    # Ensure correct names
    obs, state, loss_terms, reward, extras = step_res
    print("obs type:", type(obs))
    print("state shape:", state.shape)
    print("loss_terms shape:", loss_terms.shape)
    print("reward shape:", reward.shape)
    
    simulation_app.close()
