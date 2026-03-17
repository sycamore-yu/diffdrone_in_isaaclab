"""DreamerV3/world integration smoke tests."""

from __future__ import annotations

import pytest
import torch

from diffaero_newton.scripts.registry import build_algo, build_env, get_env_state


def _run_world_agent_steps_position_control_env(device: str, expect_update: bool = False) -> None:
    env = build_env(
        name="position_control",
        dynamics="pointmass",
        num_envs=2,
        device=device,
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
            device=device,
            env=env,
            cfg={
                "state_predictor": {
                    "action_dim": action_dim,
                    "only_state": True,
                    "enable_rec": False,
                    "use_amp": device.startswith("cuda"),
                },
                "replaybuffer": {
                    "max_length": 128,
                    "warmup_length": 1 if expect_update else 1_000,
                    "min_ready_steps": 2 if expect_update else 8,
                    "store_on_gpu": device.startswith("cuda"),
                },
                "world_state_env": {
                    "batch_size": 2,
                    "batch_length": 2 if expect_update else 4,
                    "imagine_length": 2 if expect_update else 4,
                },
            },
        )

        update_info = {}
        for _ in range(4 if expect_update else 1):
            next_state, policy_info, env_info, reward_mean, done_mean = agent.step(state)
            state = next_state
            if policy_info:
                update_info = policy_info

        assert next_state.shape == state.shape
        assert next_state.device.type == torch.device(device).type
        assert isinstance(policy_info, dict)
        assert "terminated" in env_info
        assert "truncated" in env_info
        assert isinstance(reward_mean, float)
        assert isinstance(done_mean, float)
        assert torch.isfinite(next_state).all()
        if expect_update:
            assert "WorldModel/state_total_loss" in update_info
    finally:
        env.close()


@pytest.mark.cpu_smoke
def test_world_agent_steps_position_control_env_cpu():
    _run_world_agent_steps_position_control_env(device="cpu")


@pytest.mark.gpu_smoke
def test_world_agent_steps_position_control_env_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gpu_smoke world coverage")
    _run_world_agent_steps_position_control_env(device="cuda")


@pytest.mark.cpu_smoke
def test_world_agent_updates_when_replay_is_ready_cpu():
    _run_world_agent_steps_position_control_env(device="cpu", expect_update=True)


@pytest.mark.gpu_smoke
def test_world_agent_updates_when_replay_is_ready_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gpu_smoke world coverage")
    _run_world_agent_steps_position_control_env(device="cuda", expect_update=True)


@pytest.mark.cpu_smoke
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
