import pytest
import torch

from diffaero_newton.configs.dynamics_cfg import PointMassCfg
from diffaero_newton.configs.position_control_env_cfg import PositionControlEnvCfg
from diffaero_newton.envs.position_control_env import create_env


pytestmark = pytest.mark.usefixtures("isaaclab_app")


def test_position_control_env_supports_gradient_flow() -> None:
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
        loss_terms.sum().backward()

        assert actions.grad is not None
        assert obs["policy"].shape[0] == num_envs
        assert state.shape[0] == num_envs
    except Exception as error:
        pytest.fail(f"Position control environment gradient-flow test failed: {error}")
    finally:
        env.close()
