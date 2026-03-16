"""Smoke tests for the unified training entry and registry wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "diffaero_newton/source/diffaero_newton/scripts/train.py"


def test_registry_points_to_real_modules():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.scripts.registry import ALGO_REGISTRY, DYNAMICS_REGISTRY, ENV_REGISTRY

    assert ENV_REGISTRY["position_control"] == "diffaero_newton.envs.position_control_env.PositionControlEnv"
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


def test_isaaclab_compat_exports_launch_app():
    sys.path.insert(0, str(REPO_ROOT / "diffaero_newton/source"))

    from diffaero_newton.common.isaaclab_compat import launch_app

    assert callable(launch_app)
