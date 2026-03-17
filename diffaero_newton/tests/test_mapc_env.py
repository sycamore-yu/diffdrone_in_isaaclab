import pytest
import torch

from diffaero_newton.configs.dynamics_cfg import PointMassCfg
from diffaero_newton.configs.mapc_env_cfg import MAPCEnvCfg
from diffaero_newton.envs.mapc_env import create_env


pytestmark = pytest.mark.usefixtures("isaaclab_app")


def test_mapc_env_supports_gradient_flow() -> None:
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
        loss_terms.sum().backward()

        assert actions.grad is not None
        assert obs["policy"].shape[0] == num_envs
        assert obs["policy"].shape[1] == cfg.num_observations
        assert state.shape == (num_envs, n_agents, state.shape[-1])
    except Exception as error:
        pytest.fail(f"MAPC environment gradient-flow test failed: {error}")
    finally:
        env.close()
