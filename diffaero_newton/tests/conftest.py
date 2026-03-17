"""Pytest configuration for diffaero_newton tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SOURCE_PATH = PROJECT_ROOT / "source"

if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))


@pytest.fixture(scope="session")
def isaaclab_app():
    """Launch the IsaacLab runtime once for tests that need it."""

    from diffaero_newton.common.isaaclab_launch import launch_app

    app = launch_app()
    yield app
    if app is not None:
        app.close()
