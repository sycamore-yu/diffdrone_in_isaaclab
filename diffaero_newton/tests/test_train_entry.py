"""Smoke tests for the unified training entry and registry wiring."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import types
from pathlib import Path

import torch
from gymnasium.spaces import Box
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "diffaero_newton/source/diffaero_newton/scripts/train.py"


@pytest.mark.runtime_preflight
def test_registry_points_to_real_modules():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.scripts.registry import ALGO_REGISTRY, DYNAMICS_REGISTRY, ENV_REGISTRY

    assert ENV_REGISTRY["position_control"] == "diffaero_newton.envs.position_control_env.PositionControlEnv"
    assert ENV_REGISTRY["sim2real_position_control"] == "diffaero_newton.envs.position_control_env.Sim2RealPositionControlEnv"
    assert ENV_REGISTRY["mapc"] == "diffaero_newton.envs.mapc_env.MAPCEnv"
    assert ALGO_REGISTRY["mashac"] == "diffaero_newton.training.mashac.MASHAC"
    assert ALGO_REGISTRY["sha2c"] == "diffaero_newton.training.shac.SHA2C"
    assert DYNAMICS_REGISTRY["pointmass"] == "diffaero_newton.configs.dynamics_cfg.PointMassCfg"
    assert ALGO_REGISTRY["world"] == "diffaero_newton.training.dreamerv3.World_Agent"
    assert DYNAMICS_REGISTRY["continuous_pointmass"] == "diffaero_newton.configs.dynamics_cfg.ContinuousPointMassCfg"
    assert DYNAMICS_REGISTRY["discrete_pointmass"] == "diffaero_newton.configs.dynamics_cfg.DiscretePointMassCfg"


@pytest.mark.runtime_preflight
def test_train_list_runs_without_pythonpath_hack():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Algorithms:" in result.stdout
    assert "sha2c" in result.stdout
    assert "position_control" in result.stdout
    assert "sim2real_position_control" in result.stdout


@pytest.mark.runtime_preflight
def test_train_dry_run_writes_resolved_world_config(tmp_path):
    config_path = tmp_path / "resolved-world-config.json"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--algo",
            "world",
            "--env",
            "obstacle_avoidance",
            "--dynamics",
            "pointmass",
            "--sensor",
            "camera",
            "--n_envs",
            "1",
            "--device",
            "cpu",
            "--dry-run",
            "--config-out",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert config_path.exists()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["args"]["algo"] == "world"
    assert payload["args"]["env"] == "obstacle_avoidance"
    assert payload["world_cfg"]["state_predictor"]["use_perception"] is True
    assert payload["world_cfg"]["state_predictor"]["only_state"] is False
    assert payload["env_cfg"]["sensor_cfg"]["name"] == "camera"


@pytest.mark.runtime_preflight
def test_isaaclab_launch_exports_launch_app():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.common.isaaclab_launch import launch_app

    assert callable(launch_app)


@pytest.mark.runtime_preflight
def test_package_main_uses_same_device_for_env_and_trainer(monkeypatch):
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    import diffaero_newton.__main__ as package_main
    from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
    from diffaero_newton.configs.training_cfg import TrainingCfg

    captured: dict[str, object] = {}

    class FakeApp:
        def close(self):
            captured["app_closed"] = True

    class FakeEnv:
        def __init__(self, cfg, device=None):
            captured["env_device"] = device
            captured["env_cfg_device"] = cfg.device
            self.num_envs = cfg.num_envs

        def close(self):
            captured["env_closed"] = True

    class FakeSHAC:
        def __init__(self, env, cfg):
            captured["trainer_device"] = cfg.device
            self.env = env

        def train(self):
            captured["trained"] = True

    fake_env_module = types.ModuleType("diffaero_newton.envs.drone_env")
    fake_env_module.DroneEnv = FakeEnv
    fake_cfg_module = types.ModuleType("diffaero_newton.configs.drone_env_cfg")
    fake_cfg_module.DroneEnvCfg = DroneEnvCfg
    fake_train_cfg_module = types.ModuleType("diffaero_newton.configs.training_cfg")
    fake_train_cfg_module.TrainingCfg = TrainingCfg
    fake_shac_module = types.ModuleType("diffaero_newton.training.shac")
    fake_shac_module.SHAC = FakeSHAC

    monkeypatch.setitem(sys.modules, "diffaero_newton.envs.drone_env", fake_env_module)
    monkeypatch.setitem(sys.modules, "diffaero_newton.configs.drone_env_cfg", fake_cfg_module)
    monkeypatch.setitem(sys.modules, "diffaero_newton.configs.training_cfg", fake_train_cfg_module)
    monkeypatch.setitem(sys.modules, "diffaero_newton.training.shac", fake_shac_module)
    monkeypatch.setattr(package_main, "launch_app", lambda args=None: FakeApp())
    monkeypatch.setattr(package_main, "parse_args", lambda: types.SimpleNamespace(
        num_envs=4,
        episode_length_s=5.0,
        num_iterations=1,
        rollout_horizon=2,
        actor_lr=3e-4,
        critic_lr=1e-3,
        log_interval=1,
        save_interval=10,
        save_dir="checkpoints",
        log_dir="runs/test",
        no_tensorboard=True,
        device="cuda",
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    package_main.main()

    assert captured["env_device"] == "cpu"
    assert captured["env_cfg_device"] == "cpu"
    assert captured["trainer_device"] == "cpu"
    assert captured["trained"] is True
    assert captured["env_closed"] is True
    assert captured["app_closed"] is True


@pytest.mark.runtime_preflight
def test_direct_rl_shim_honors_decimation():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.common.direct_rl_shim import DirectRLEnv, SimulationCfg

    class DummyCfg:
        num_envs = 2
        decimation = 4
        episode_length_s = 1.0
        sim = SimulationCfg(dt=1.0 / 120.0)
        observation_space = Box(low=-1.0, high=1.0, shape=(1,))
        action_space = Box(low=-1.0, high=1.0, shape=(1,))

    class DummyEnv(DirectRLEnv):
        def __init__(self):
            self.apply_calls = 0
            super().__init__(DummyCfg(), device="cpu")

        def _pre_physics_step(self, action):
            self.last_action = action

        def _get_observations(self):
            return torch.zeros(self.num_envs, 1, device=self.device)

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            return (
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )

        def _apply_action(self):
            self.apply_calls += 1

    env = DummyEnv()
    env.step(torch.zeros(env.num_envs, 1))

    assert env.apply_calls == env.decimation


@pytest.mark.cpu_smoke
def test_apg_iteration_returns_detached_observation_for_next_iteration():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    import diffaero_newton.scripts.train as train_entry
    from diffaero_newton.scripts.registry import build_algo, build_env, get_policy_obs

    env = build_env(
        name="position_control",
        dynamics="pointmass",
        num_envs=2,
        device="cpu",
        differentiable=True,
    )

    try:
        obs, _ = env.reset()
        agent = build_algo(
            "apg",
            obs_dim=get_policy_obs(obs).shape[-1],
            action_dim=env.action_space.shape[0],
            device="cpu",
            lr=1.0e-3,
            l_rollout=2,
        )

        obs, metrics = train_entry._run_apg_iteration(agent, env, obs, l_rollout=2)
        assert "actor_loss" in metrics
        assert get_policy_obs(obs).grad_fn is None

        obs, metrics = train_entry._run_apg_iteration(agent, env, obs, l_rollout=2)
        assert "actor_loss" in metrics
        assert get_policy_obs(obs).grad_fn is None
    finally:
        env.close()


@pytest.mark.cpu_smoke
def test_build_env_applies_quadrotor_body_rate_overrides_end_to_end():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.scripts.registry import build_env

    env = build_env(
        name="position_control",
        dynamics="quadrotor",
        num_envs=1,
        device="cpu",
        differentiable=True,
        dynamics_overrides={
            "control_mode": "body_rate",
            "drag_coeff_xy": 0.05,
            "drag_coeff_z": 0.1,
            "k_angvel": (5.0, 4.0, 3.0),
            "max_body_rates": (3.0, 3.0, 1.5),
            "thrust_ratio": 0.85,
            "torque_ratio": 0.75,
        },
    )

    try:
        obs, _ = env.reset()
        state_before = env.drone.get_flat_state().clone()
        action = torch.tensor([[0.65, 1.0, 0.0, 0.5]], device=env.device)
        next_obs, state_after, loss_terms, reward, _extras = env.step(action)

        assert env.cfg.dynamics.control_mode == "body_rate"
        assert env.drone.config.control_mode == "body_rate"
        assert env.drone.config.max_body_rates == (3.0, 3.0, 1.5)
        assert env.drone.config.thrust_ratio == pytest.approx(0.85)
        assert env.drone.config.torque_ratio == pytest.approx(0.75)
        assert obs["policy"].shape[-1] == next_obs["policy"].shape[-1]
        assert torch.isfinite(loss_terms).all()
        assert torch.isfinite(reward).all()
        assert not torch.allclose(state_before, state_after)
    finally:
        env.close()
