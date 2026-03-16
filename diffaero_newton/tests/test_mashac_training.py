"""Tests for the MASHAC multi-agent training path."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_compat import launch_app

app = launch_app()

from diffaero_newton.configs.dynamics_cfg import ContinuousPointMassCfg
from diffaero_newton.configs.mapc_env_cfg import MAPCEnvCfg
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.envs.mapc_env import create_env
from diffaero_newton.training.mashac import MASHAC, MASHACAgent


def _make_env(device: str, num_envs: int = 2, n_agents: int = 3):
    cfg = MAPCEnvCfg()
    cfg.num_envs = num_envs
    cfg.n_agents = n_agents
    cfg.scene.num_envs = num_envs
    cfg.__post_init__()
    cfg.dynamics = ContinuousPointMassCfg(num_envs=num_envs * n_agents, requires_grad=True, dt=cfg.sim.dt)
    return create_env(cfg=cfg, device=device)


def test_mashac_agent_action_shape():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 111
    state_dim = 30
    action_dim = 12

    agent = MASHACAgent(obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim, cfg=TrainingCfg(device=device))
    obs = torch.randn(4, obs_dim, device=device)
    action, log_prob, entropy = agent.get_action(obs)

    assert action.shape == (4, action_dim)
    assert log_prob.shape == (4, 1)
    assert entropy.shape == (4, 1)


def test_mashac_training_iteration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = _make_env(device=device)
    trainer = MASHAC(
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
        if app is not None:
            app.close()
