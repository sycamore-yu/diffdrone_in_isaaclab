"""Smoke tests for the unified training entry and registry wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from gymnasium.spaces import Box


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "diffaero_newton/source/diffaero_newton/scripts/train.py"


def test_registry_points_to_real_modules():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.scripts.registry import ALGO_REGISTRY, DYNAMICS_REGISTRY, ENV_REGISTRY

    assert ENV_REGISTRY["position_control"] == "diffaero_newton.envs.position_control_env.PositionControlEnv"
    assert ENV_REGISTRY["sim2real_position_control"] == "diffaero_newton.envs.position_control_env.Sim2RealPositionControlEnv"
    assert ENV_REGISTRY["mapc"] == "diffaero_newton.envs.mapc_env.MAPCEnv"
    assert ALGO_REGISTRY["mashac"] == "diffaero_newton.training.mashac.MASHAC"
    assert DYNAMICS_REGISTRY["pointmass"] == "diffaero_newton.configs.dynamics_cfg.PointMassCfg"
    assert ALGO_REGISTRY["world"] == "diffaero_newton.training.dreamerv3.World_Agent"
    assert DYNAMICS_REGISTRY["continuous_pointmass"] == "diffaero_newton.configs.dynamics_cfg.ContinuousPointMassCfg"
    assert DYNAMICS_REGISTRY["discrete_pointmass"] == "diffaero_newton.configs.dynamics_cfg.DiscretePointMassCfg"


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
    assert "position_control" in result.stdout
    assert "sim2real_position_control" in result.stdout


def test_isaaclab_launch_exports_launch_app():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.common.isaaclab_launch import launch_app

    assert callable(launch_app)


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
