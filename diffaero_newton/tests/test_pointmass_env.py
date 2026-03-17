import pytest
import torch

from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.configs.dynamics_cfg import (
    ContinuousPointMassCfg,
    DiscretePointMassCfg,
    PointMassCfg,
)
from diffaero_newton.envs.drone_env import create_env


pytestmark = pytest.mark.usefixtures("isaaclab_app")


@pytest.mark.parametrize("cfg_cls", [PointMassCfg, ContinuousPointMassCfg, DiscretePointMassCfg])
def test_pointmass_env_step_supports_backward(cfg_cls: type[PointMassCfg]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 2

    cfg = DroneEnvCfg()
    cfg.num_envs = num_envs
    cfg.scene.num_envs = num_envs
    cfg.dynamics = cfg_cls(num_envs=num_envs, requires_grad=True)

    env = create_env(cfg=cfg, device=device)
    env.reset()
    actions = torch.ones(num_envs, 3, device=device, requires_grad=True)
    padded_actions = torch.cat(
        [actions, torch.zeros(num_envs, 1, device=device, requires_grad=True)],
        dim=-1,
    )

    try:
        obs, state, loss_terms, reward, extras = env.step(padded_actions)
        loss_terms.sum().backward()
        assert actions.grad is not None
    except Exception as error:
        pytest.fail(f"Point-mass environment backward test failed: {error}")
    finally:
        env.close()
