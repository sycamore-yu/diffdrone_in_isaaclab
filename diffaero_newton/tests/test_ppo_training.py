"""Test PPO and AsymmetricPPO training algorithms."""

import pytest
import torch

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.training.ppo import AsymmetricPPO, PPO


pytestmark = pytest.mark.usefixtures("isaaclab_app")


def test_ppo_updates_parameters() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    n_envs = 8
    l_rollout = 4

    agent = PPO(
        obs_dim=obs_dim,
        action_dim=ACTION_DIM,
        n_envs=n_envs,
        l_rollout=l_rollout,
        n_minibatch=2,
        n_epoch=2,
        device=device,
    )
    initial_params = [parameter.clone() for parameter in agent.agent.parameters()]

    agent.buffer.clear()
    obs = torch.randn(n_envs, obs_dim, device=device)
    for _ in range(l_rollout):
        with torch.no_grad():
            action, info = agent.act(obs)
            reward = -((action - 0.5) ** 2).sum(-1)
            next_done = torch.zeros(n_envs, device=device)
            next_value = agent.agent.get_value(obs)
            agent.buffer.add(
                obs,
                info["sample"],
                info["logprob"],
                reward,
                next_done,
                info["value"],
                next_value,
            )
            obs = torch.randn(n_envs, obs_dim, device=device)

    advantages, target_values = agent.bootstrap()
    for _ in range(agent.n_epoch):
        losses, grad_norms = agent.train_epoch(advantages, target_values)

    assert "actor_loss" in losses
    assert "critic_loss" in losses
    assert "entropy_loss" in losses
    assert any(
        not torch.allclose(previous_param, current_param)
        for previous_param, current_param in zip(initial_params, agent.agent.parameters())
    ), "PPO parameters should change after training"


def test_asymmetric_ppo_updates_parameters() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    state_dim = 30
    n_envs = 8
    l_rollout = 4

    agent = AsymmetricPPO(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=ACTION_DIM,
        n_envs=n_envs,
        l_rollout=l_rollout,
        n_minibatch=2,
        n_epoch=2,
        device=device,
    )
    initial_params = [parameter.clone() for parameter in agent.agent.parameters()]

    agent.buffer.clear()
    obs = torch.randn(n_envs, obs_dim, device=device)
    for _ in range(l_rollout):
        with torch.no_grad():
            action, info = agent.act(obs)
            state = torch.randn(n_envs, state_dim, device=device)
            reward = -((action - 0.5) ** 2).sum(-1)
            next_done = torch.zeros(n_envs, device=device)
            next_value = agent.agent.get_value(state)
            agent.buffer.add(
                obs,
                state,
                info["sample"],
                info["logprob"],
                reward,
                next_done,
                next_value,
                next_value,
            )
            obs = torch.randn(n_envs, obs_dim, device=device)

    advantages, target_values = agent.bootstrap()
    for _ in range(agent.n_epoch):
        losses, grad_norms = agent.train_epoch(advantages, target_values)

    assert "actor_loss" in losses
    assert "critic_loss" in losses
    assert any(
        not torch.allclose(previous_param, current_param)
        for previous_param, current_param in zip(initial_params, agent.agent.parameters())
    ), "APPO parameters should change after training"
