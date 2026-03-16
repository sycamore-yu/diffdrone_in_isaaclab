"""Test PPO and AsymmetricPPO training algorithms.

Validates:
  1. PPO collects rollout data, computes GAE, and trains with minibatches.
  2. AsymmetricPPO uses privileged state for critic.
  3. Parameters change after training.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
from diffaero_newton.common.isaaclab_compat import launch_app
app = launch_app()

import torch
from diffaero_newton.training.ppo import PPO, AsymmetricPPO
from diffaero_newton.common.constants import ACTION_DIM


def test_ppo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    n_envs = 8
    l_rollout = 4

    agent = PPO(obs_dim=obs_dim, action_dim=ACTION_DIM, n_envs=n_envs,
                l_rollout=l_rollout, n_minibatch=2, n_epoch=2, device=device)

    init_params = [p.clone() for p in agent.agent.parameters()]

    # Simulate rollout
    agent.buffer.clear()
    obs = torch.randn(n_envs, obs_dim, device=device)
    for _ in range(l_rollout):
        with torch.no_grad():
            action, info = agent.act(obs)
            reward = -((action - 0.5) ** 2).sum(-1)
            next_done = torch.zeros(n_envs, device=device)
            next_value = agent.agent.get_value(obs)
            agent.buffer.add(obs, info["sample"], info["logprob"],
                             reward, next_done, info["value"], next_value)
            obs = torch.randn(n_envs, obs_dim, device=device)

    # Bootstrap and train
    advantages, target_values = agent.bootstrap()
    for _ in range(agent.n_epoch):
        losses, grad_norms = agent.train_epoch(advantages, target_values)

    assert "actor_loss" in losses
    assert "critic_loss" in losses
    assert "entropy_loss" in losses

    # Verify params changed
    changed = False
    for p_old, p_new in zip(init_params, agent.agent.parameters()):
        if not torch.allclose(p_old, p_new):
            changed = True
            break
    assert changed, "PPO parameters should change after training"

    print(f"PPO test passed: actor_loss={losses['actor_loss']:.4f}, "
          f"critic_loss={losses['critic_loss']:.4f}")


def test_asymmetric_ppo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 18
    state_dim = 30  # privileged state
    n_envs = 8
    l_rollout = 4

    agent = AsymmetricPPO(obs_dim=obs_dim, state_dim=state_dim,
                          action_dim=ACTION_DIM, n_envs=n_envs,
                          l_rollout=l_rollout, n_minibatch=2, n_epoch=2,
                          device=device)

    init_params = [p.clone() for p in agent.agent.parameters()]

    # Simulate rollout with privileged state
    agent.buffer.clear()
    obs = torch.randn(n_envs, obs_dim, device=device)
    for _ in range(l_rollout):
        with torch.no_grad():
            action, info = agent.act(obs)
            state = torch.randn(n_envs, state_dim, device=device)
            reward = -((action - 0.5) ** 2).sum(-1)
            next_done = torch.zeros(n_envs, device=device)
            next_value = agent.agent.get_value(state)
            agent.buffer.add(obs, state, info["sample"], info["logprob"],
                             reward, next_done, next_value, next_value)
            obs = torch.randn(n_envs, obs_dim, device=device)

    advantages, target_values = agent.bootstrap()
    for _ in range(agent.n_epoch):
        losses, _ = agent.train_epoch(advantages, target_values)

    assert "actor_loss" in losses
    assert "critic_loss" in losses

    changed = False
    for p_old, p_new in zip(init_params, agent.agent.parameters()):
        if not torch.allclose(p_old, p_new):
            changed = True
            break
    assert changed, "APPO parameters should change after training"

    print(f"AsymmetricPPO test passed: actor_loss={losses['actor_loss']:.4f}, "
          f"critic_loss={losses['critic_loss']:.4f}")


def main():
    try:
        test_ppo()
        test_asymmetric_ppo()
        print("\n=== All PPO training tests passed ===")
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
