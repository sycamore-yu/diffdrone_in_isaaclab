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


@pytest.mark.cpu_smoke
def test_pointmass_env_normalized_actions_move_symmetrically_in_x() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DroneEnvCfg()
    cfg.num_envs = 2
    cfg.scene.num_envs = 2
    cfg.decimation = 1
    cfg.dynamics = PointMassCfg(num_envs=2, requires_grad=False, dt=cfg.sim.dt)

    env = create_env(cfg=cfg, device=device)
    env.reset(seed=0)

    try:
        actions = torch.tensor(
            [
                [1.0, 0.5, 0.25, 0.0],
                [0.0, 0.5, 0.25, 0.0],
            ],
            device=device,
        )
        _obs, state, _loss_terms, _reward, extras = env.step(actions)

        assert not extras["reset"].any()
        assert state[0, 7].item() > 0.0
        assert state[1, 7].item() < 0.0

        _obs, state, _loss_terms, _reward, extras = env.step(actions)
        assert not extras["reset"].any()
        assert state[0, 0].item() > 0.0
        assert state[1, 0].item() < 0.0
    finally:
        env.close()
