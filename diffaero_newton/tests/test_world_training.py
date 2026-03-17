"""DreamerV3/world integration smoke tests."""

from __future__ import annotations

import torch

from diffaero_newton.scripts.registry import build_algo, build_env, get_env_state


def test_world_agent_steps_position_control_env():
    env = build_env(
        name="position_control",
        dynamics="pointmass",
        num_envs=2,
        device="cpu",
        differentiable=False,
    )

    try:
        env.reset()
        state = get_env_state(env)
        action_dim = env.action_space.shape[0]
        agent = build_algo(
            "world",
            obs_dim=state.shape[-1],
            action_dim=action_dim,
            device="cpu",
            env=env,
            cfg={
                "state_predictor": {
                    "action_dim": action_dim,
                    "only_state": True,
                    "enable_rec": False,
                    "use_amp": False,
                },
                "replaybuffer": {
                    "max_length": 128,
                    "warmup_length": 1_000,
                    "min_ready_steps": 8,
                    "store_on_gpu": False,
                },
                "world_state_env": {
                    "batch_size": 2,
                    "batch_length": 4,
                    "imagine_length": 4,
                },
            },
        )

        next_state, policy_info, env_info, reward_mean, done_mean = agent.step(state)

        assert next_state.shape == state.shape
        assert next_state.device.type == "cpu"
        assert isinstance(policy_info, dict)
        assert "terminated" in env_info
        assert "truncated" in env_info
        assert isinstance(reward_mean, float)
        assert isinstance(done_mean, float)
        assert torch.isfinite(next_state).all()
    finally:
        env.close()


def test_world_agent_unpacks_gym_style_step_output():
    class DummyEnv:
        device = torch.device("cpu")

    agent = build_algo(
        "world",
        obs_dim=13,
        action_dim=4,
        device="cpu",
        env=DummyEnv(),
        cfg={
            "is_test": True,
            "state_predictor": {
                "action_dim": 4,
                "only_state": True,
                "enable_rec": False,
                "use_amp": False,
            },
        },
    )

    next_obs = torch.randn(2, 13)
    reward = torch.randn(2)
    terminated = torch.tensor([False, True])
    truncated = torch.tensor([False, False])

    _, next_state, parsed_reward, done, extras = agent._unpack_env_step(
        (next_obs, reward, terminated, truncated, {})
    )

    assert torch.equal(next_state, next_obs)
    assert torch.equal(parsed_reward, reward)
    assert torch.equal(done, torch.tensor([0.0, 1.0]))
    assert torch.equal(extras["terminated"], terminated)
    assert torch.equal(extras["truncated"], truncated)
