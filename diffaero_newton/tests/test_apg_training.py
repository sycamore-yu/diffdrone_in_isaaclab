"""Test APG and APG_stochastic training algorithms.

Validates:
  1. APG deterministic actor processes obs → action correctly.
  2. APG accumulates loss and updates parameters (grad ≠ 0).
  3. APGStochastic produces stochastic actions and updates with entropy.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_launch import launch_app
app = launch_app()

import torch

from diffaero_newton.training.apg import APG, APGStochastic
from diffaero_newton.common.constants import ACTION_DIM


def test_apg_deterministic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    agent = APG(obs_dim=obs_dim, action_dim=ACTION_DIM, lr=1e-3, l_rollout=4, device=device)

    # Save initial params
    init_params = [p.clone() for p in agent.actor.parameters()]

    # Simulate a short rollout
    obs = torch.randn(8, obs_dim, device=device, requires_grad=True)
    for _ in range(4):
        action = agent.act(obs)
        # Dummy differentiable loss: action should move toward 0.5
        loss = ((action - 0.5) ** 2).sum(dim=-1)
        agent.record_loss(loss)
        obs = torch.randn(8, obs_dim, device=device, requires_grad=True)

    metrics = agent.update_actor()
    assert "actor_loss" in metrics
    assert "actor_grad_norm" in metrics
    assert metrics["actor_grad_norm"] > 0, "Gradients should be non-zero"

    # Verify params changed
    for p_old, p_new in zip(init_params, agent.actor.parameters()):
        assert not torch.allclose(p_old, p_new), "Parameters should have changed after update"

    print(f"APG deterministic test passed: loss={metrics['actor_loss']:.4f}, grad_norm={metrics['actor_grad_norm']:.4f}")


def test_apg_stochastic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    agent = APGStochastic(obs_dim=obs_dim, action_dim=ACTION_DIM, lr=1e-3,
                          l_rollout=4, entropy_weight=0.01, device=device)

    init_params = [p.clone() for p in agent.actor.parameters()]

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

    for p_old, p_new in zip(init_params, agent.actor.parameters()):
        assert not torch.allclose(p_old, p_new), "Parameters should have changed after update"

    print(f"APG stochastic test passed: loss={metrics['actor_loss']:.4f}, "
          f"entropy_loss={metrics['entropy_loss']:.4f}, grad_norm={metrics['actor_grad_norm']:.4f}")


def main():
    try:
        test_apg_deterministic()
        test_apg_stochastic()
        print("\n=== All APG training tests passed ===")
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
