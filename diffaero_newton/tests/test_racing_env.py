"""Contract and regression tests for the racing environment."""

from __future__ import annotations

import pytest
import torch

from diffaero_newton.configs.dynamics_cfg import PointMassCfg
from diffaero_newton.configs.racing_env_cfg import RacingEnvCfg
from diffaero_newton.envs.racing_env import RacingEnv, get_gate_rotmat_w2g
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.scripts.registry import build_env
from diffaero_newton.training.shac import SHAC


def _make_racing_env(device: str = "cpu", num_envs: int = 1) -> RacingEnv:
    cfg = RacingEnvCfg()
    cfg.num_envs = num_envs
    cfg.scene.num_envs = num_envs
    cfg.dynamics = PointMassCfg(num_envs=num_envs, requires_grad=True, dt=cfg.sim.dt)
    return RacingEnv(cfg=cfg, device=device)


def _set_gate_frame_state(
    env: RacingEnv,
    *,
    env_id: int,
    gate_idx: int,
    rel_pos_gate: torch.Tensor,
    vel_gate: torch.Tensor | None = None,
) -> None:
    gate_pos = env.gate_pos[gate_idx]
    gate_yaw = env.gate_yaw[gate_idx].unsqueeze(0)
    rotmat = get_gate_rotmat_w2g(gate_yaw).squeeze(0)

    state = env.drone.get_flat_state().detach().clone()
    state[env_id, :3] = gate_pos + rotmat.transpose(0, 1) @ rel_pos_gate.to(device=env.device)
    state[env_id, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[env_id, 7:10] = 0.0 if vel_gate is None else rotmat.transpose(0, 1) @ vel_gate.to(device=env.device)
    state[env_id, 10:13] = 0.0
    env.drone.set_state(state)


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
def test_racing_reset_starts_one_meter_before_target_gate() -> None:
    env = _make_racing_env(num_envs=8)

    try:
        env.reset(seed=0)
        positions = env.drone.get_state()["position"]
        gate_pos = env.gate_pos[env.target_gates]
        gate_yaw = env.gate_yaw[env.target_gates]
        forward = torch.stack(
            (
                torch.cos(gate_yaw),
                torch.sin(gate_yaw),
                torch.zeros_like(gate_yaw),
            ),
            dim=-1,
        )

        assert torch.allclose(positions, gate_pos - forward, atol=1.0e-5)
        assert env.target_gates.unique().numel() > 1
    finally:
        env.close()


@pytest.mark.cpu_smoke
def test_gate_pass_advances_target_and_emits_reward(monkeypatch) -> None:
    env = _make_racing_env()

    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (1, env.cfg.num_observations)

        env.target_gates[0] = 1
        env.n_passed_gates[0] = 0
        gate_idx = 1
        gate_pos = env.gate_pos[gate_idx]
        gate_yaw = env.gate_yaw[gate_idx].unsqueeze(0)
        rotmat = get_gate_rotmat_w2g(gate_yaw).squeeze(0)

        def _world_pos(rel_pos: torch.Tensor) -> torch.Tensor:
            return gate_pos + rotmat.transpose(0, 1) @ rel_pos

        _set_gate_frame_state(
            env,
            env_id=0,
            gate_idx=gate_idx,
            rel_pos_gate=torch.tensor([-0.5, 0.0, 0.0], device=env.device),
        )

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
        assert env.target_gates[0].item() == 2
        assert not extras["terminated"][0]
    finally:
        env.close()


@pytest.mark.cpu_smoke
def test_racing_progress_reward_uses_public_velocity_shaping_and_reward_progress() -> None:
    env = _make_racing_env()
    env.racing_cfg.use_vel_track = False
    env.racing_cfg.reward_progress = 3.0
    env.racing_cfg.progress_loss_weight = 0.0

    try:
        env.reset(seed=0)
        env.target_gates[0] = 1
        env.actions.zero_()
        env.prev_actions.zero_()
        _set_gate_frame_state(
            env,
            env_id=0,
            gate_idx=1,
            rel_pos_gate=torch.tensor([0.5, 0.0, 0.0], device=env.device),
            vel_gate=torch.tensor([0.4, 0.3, -0.2], device=env.device),
        )

        reward, loss = env._compute_step_terms(prev_gate_dist=torch.tensor([1.5], device=env.device))
        gate_dist = torch.tensor([0.5], device=env.device)
        progress_loss = gate_dist - torch.tensor([1.5], device=env.device)
        pos_loss = 1.0 - torch.exp(-gate_dist)
        vel_loss = torch.tensor(
            [
                (env.racing_cfg.min_target_vel - 0.4) ** 2 + 0.3 ** 2 + (-0.2) ** 2
            ],
            device=env.device,
        )
        expected_reward = (
            env.racing_cfg.reward_constant
            - env.racing_cfg.reward_progress * progress_loss
            - env.racing_cfg.pos_loss_weight * pos_loss
            - env.racing_cfg.vel_loss_weight * vel_loss
        )
        expected_loss = (
            env.racing_cfg.pos_loss_weight * pos_loss
            + env.racing_cfg.vel_loss_weight * vel_loss
        )

        assert torch.allclose(reward, expected_reward, atol=1.0e-6)
        assert torch.allclose(loss, expected_loss + progress_loss * 0.0, atol=1.0e-6)
    finally:
        env.close()


@pytest.mark.cpu_smoke
def test_racing_shac_training_smoke_emits_racing_metrics(tmp_path) -> None:
    env = _make_racing_env(device="cpu", num_envs=4)
    trainer = SHAC(
        env,
        cfg=TrainingCfg(
            device="cpu",
            rollout_horizon=4,
            num_iterations=2,
            log_interval=1,
            save_interval=1000,
            save_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "runs"),
            enable_tensorboard=True,
        ),
    )

    try:
        trainer.train()

        assert "racing" in trainer.last_extras
        assert {"gate_passed", "gate_collisions", "target_gate_mean", "passed_gate_mean"} <= set(
            trainer.last_extras["racing"]
        )
        assert list((tmp_path / "runs").rglob("events.out.tfevents*"))
    finally:
        env.close()
