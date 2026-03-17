import torch

from diffaero_newton.configs.dynamics_cfg import PointMassCfg
from diffaero_newton.configs.position_control_env_cfg import Sim2RealPositionControlEnvCfg
from diffaero_newton.envs.position_control_env import create_sim2real_env


def test_sim2real_position_control_env_updates_square_targets():
    cfg = Sim2RealPositionControlEnvCfg()
    cfg.num_envs = 2
    cfg.scene.num_envs = 2
    cfg.switch_time = 0.05
    cfg.square_size = 1.5
    cfg.dynamics = PointMassCfg(num_envs=2, requires_grad=True)

    env = create_sim2real_env(cfg=cfg, device="cpu")

    try:
        env.reset()
        initial_target = env.target_pos.clone()
        actions = torch.zeros(2, 4, device="cpu", requires_grad=True)

        for _ in range(3):
            obs, state, loss_terms, reward, extras = env.step(actions)

        assert obs["policy"].shape == (2, cfg.num_observations)
        assert state.shape[0] == 2
        assert loss_terms.shape == (2,)
        assert reward.shape == (2,)
        assert "terminated" in extras and "truncated" in extras
        assert not torch.allclose(env.target_pos, initial_target)
    finally:
        env.close()
