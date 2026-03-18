"""Tests for SHAC-family training paths, including SHA2C."""

from __future__ import annotations

import pytest
import torch

from diffaero_newton.configs.dynamics_cfg import PointMassCfg
from diffaero_newton.configs.position_control_env_cfg import PositionControlEnvCfg
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.envs.position_control_env import create_env
from diffaero_newton.training.shac import SHA2C, SHA2CAgent


pytestmark = [pytest.mark.usefixtures("isaaclab_app"), pytest.mark.cpu_smoke]


def _make_env(device: str, num_envs: int = 2):
    cfg = PositionControlEnvCfg()
    cfg.num_envs = num_envs
    cfg.scene.num_envs = num_envs
    cfg.dynamics = PointMassCfg(
        num_envs=num_envs,
        requires_grad=True,
        dt=cfg.sim.dt,
    )
    return create_env(cfg=cfg, device=device)


def test_sha2c_agent_action_shape() -> None:
    device = "cpu"
    agent = SHA2CAgent(
        obs_dim=18,
        state_dim=13,
        action_dim=4,
        cfg=TrainingCfg(device=device),
    )
    obs = torch.randn(4, 18, device=device)
    action, log_prob, entropy = agent.get_action(obs)

    assert action.shape == (4, 4)
    assert log_prob.shape == (4, 1)
    assert entropy.shape == (4, 1)


def test_sha2c_training_iteration() -> None:
    device = "cpu"
    env = _make_env(device=device)
    trainer = SHA2C(
        env,
        cfg=TrainingCfg(
            device=device,
            rollout_horizon=2,
            num_iterations=1,
            save_interval=1000,
            enable_tensorboard=False,
        ),
    )

    try:
        obs, _ = env.reset()
        trainer._collect_rollout(obs)
        assert trainer.buffer.actor_loss_graph is not None
        assert trainer.buffer.actor_loss_graph.requires_grad
        assert trainer.buffer.states.shape[-1] == trainer.agent.state_dim

        actor_before = {
            name: param.detach().clone()
            for name, param in trainer.agent.actor.named_parameters()
        }
        metrics = trainer.agent.update(trainer.buffer)
        actor_changed = any(
            not torch.allclose(actor_before[name], param.detach(), atol=1e-8)
            for name, param in trainer.agent.actor.named_parameters()
        )

        assert trainer.buffer.ptr == trainer.cfg.rollout_horizon
        assert actor_changed
        assert set(metrics.keys()) == {
            "actor_loss",
            "critic_loss",
            "entropy",
            "value_mean",
            "advantage_mean",
        }
    finally:
        env.close()


def test_sha2c_collect_rollout_returns_detached_observation() -> None:
    device = "cpu"
    env = _make_env(device=device)
    trainer = SHA2C(
        env,
        cfg=TrainingCfg(
            device=device,
            rollout_horizon=2,
            num_iterations=2,
            save_interval=1000,
            enable_tensorboard=False,
        ),
    )

    try:
        obs, _ = env.reset()
        next_obs = trainer._collect_rollout(obs)
        policy_obs = next_obs["policy"] if isinstance(next_obs, dict) else next_obs
        assert not policy_obs.requires_grad

        next_obs = trainer._collect_rollout(next_obs)
        next_policy_obs = next_obs["policy"] if isinstance(next_obs, dict) else next_obs
        assert not next_policy_obs.requires_grad
    finally:
        env.close()
