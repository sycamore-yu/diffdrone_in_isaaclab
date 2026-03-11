import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test App")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    num_envs: int = 1
    episode_length_s: float = 10.0
    decimation: int = 1
    num_actions: int = 4
    num_observations: int = 1
    num_states: int = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)

class MyEnv(DirectRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
    def _setup_scene(self):
        pass
    def _pre_physics_step(self, actions):
        pass
    def _apply_action(self):
        pass
    def _get_observations(self):
        return {"policy": torch.zeros(1, 1)}
    def _get_rewards(self):
        return torch.zeros(1)
    def _get_dones(self):
        return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    def _reset_idx(self, env_ids):
        pass

if __name__ == "__main__":
    cfg = MyEnvCfg()
    env = MyEnv(cfg)
    print("Success")
    simulation_app.close()
