"""Contract and regression tests for the racing environment."""

from __future__ import annotations

import pytest
import torch

from diffaero_newton.configs.dynamics_cfg import PointMassCfg
from diffaero_newton.configs.racing_env_cfg import RacingEnvCfg
from diffaero_newton.envs.racing_env import RacingEnv, get_gate_rotmat_w2g
from diffaero_newton.scripts.registry import build_env


def _make_racing_env(device: str = "cpu", num_envs: int = 1) -> RacingEnv:
    cfg = RacingEnvCfg()
    cfg.num_envs = num_envs
    cfg.scene.num_envs = num_envs
    cfg.dynamics = PointMassCfg(num_envs=num_envs, requires_grad=True, dt=cfg.sim.dt)
    return RacingEnv(cfg=cfg, device=device)


@pytest.mark.cpu_smoke
def test_registry_racing_observation_contract_matches_cfg() -> None:
    env = build_env(
        name="racing",
        dynamics="pointmass",
        num_envs=2,
        device="cpu",
        differentiable=True,
    )

    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, env.cfg.num_observations)
        assert env.cfg.num_observations == 10
        assert env.observation_space.shape == (10,)
    finally:
        env.close()


@pytest.mark.cpu_smoke
def test_gate_pass_advances_target_and_emits_reward(monkeypatch) -> None:
    env = _make_racing_env()

    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (1, env.cfg.num_observations)

        gate_idx = env.target_gates[0].item()
        gate_pos = env.gate_pos[gate_idx]
        gate_yaw = env.gate_yaw[gate_idx].unsqueeze(0)
        rotmat = get_gate_rotmat_w2g(gate_yaw).squeeze(0)

        def _world_pos(rel_pos: torch.Tensor) -> torch.Tensor:
            return gate_pos + rotmat.transpose(0, 1) @ rel_pos

        state = env.drone.get_flat_state().detach().clone()
        state[0, :3] = _world_pos(torch.tensor([-0.5, 0.0, 0.0], device=env.device))
        state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        state[0, 7:10] = 0.0
        state[0, 10:13] = 0.0
        env.drone.set_state(state)

        def _teleport_apply_action() -> None:
            stepped = env.drone.get_flat_state().detach().clone()
            stepped[0, :3] = _world_pos(torch.tensor([0.5, 0.0, 0.0], device=env.device))
            env.drone.set_state(stepped)

        monkeypatch.setattr(env, "_apply_action", _teleport_apply_action)

        next_obs, _state, _loss, reward, extras = env.step(torch.zeros(1, 4, device=env.device))

        assert next_obs["policy"].shape == (1, env.cfg.num_observations)
        assert reward[0].item() > env.racing_cfg.reward_constant
        assert extras["racing"]["gate_passed"] == 1
        assert extras["racing"]["gate_collisions"] == 0
        assert env.n_passed_gates[0].item() == 1
        assert env.target_gates[0].item() == 1
        assert not extras["terminated"][0]
    finally:
        env.close()
