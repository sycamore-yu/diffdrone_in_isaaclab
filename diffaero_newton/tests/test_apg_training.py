"""Test APG and APG_stochastic training algorithms."""

import pytest
import torch

from diffaero_newton.common.constants import ACTION_DIM
from diffaero_newton.training.apg import APG, APGStochastic


pytestmark = pytest.mark.usefixtures("isaaclab_app")


def test_apg_deterministic_updates_parameters() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    agent = APG(obs_dim=obs_dim, action_dim=ACTION_DIM, lr=1e-3, l_rollout=4, device=device)
    initial_params = [parameter.clone() for parameter in agent.actor.parameters()]

    obs = torch.randn(8, obs_dim, device=device, requires_grad=True)
    for _ in range(4):
        action = agent.act(obs)
        loss = ((action - 0.5) ** 2).sum(dim=-1)
        agent.record_loss(loss)
        obs = torch.randn(8, obs_dim, device=device, requires_grad=True)

    metrics = agent.update_actor()
    assert "actor_loss" in metrics
    assert "actor_grad_norm" in metrics
    assert metrics["actor_grad_norm"] > 0
    assert any(
        not torch.allclose(previous_param, current_param)
        for previous_param, current_param in zip(initial_params, agent.actor.parameters())
    ), "Parameters should have changed after update"


def test_apg_stochastic_updates_parameters() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    agent = APGStochastic(
        obs_dim=obs_dim,
        action_dim=ACTION_DIM,
        lr=1e-3,
        l_rollout=4,
        entropy_weight=0.01,
        device=device,
    )
    initial_params = [parameter.clone() for parameter in agent.actor.parameters()]

    obs = torch.randn(8, obs_dim, device=device, requires_grad=True)
    for _ in range(4):
        action, log_prob, entropy = agent.act(obs)
        loss = ((action - 0.5) ** 2).sum(dim=-1)
        agent.record_loss(loss, entropy)
        obs = torch.randn(8, obs_dim, device=device, requires_grad=True)

    metrics = agent.update_actor()
    assert "actor_loss" in metrics
    assert "entropy_loss" in metrics
    assert "actor_grad_norm" in metrics
    assert metrics["actor_grad_norm"] > 0
    assert any(
        not torch.allclose(previous_param, current_param)
        for previous_param, current_param in zip(initial_params, agent.actor.parameters())
    ), "Parameters should have changed after update"
