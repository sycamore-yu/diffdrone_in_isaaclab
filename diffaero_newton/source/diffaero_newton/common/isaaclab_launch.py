"""Runtime launcher helpers for real IsaacLab execution."""

from __future__ import annotations

from typing import Any


def launch_app(headless: bool = True, **kwargs) -> Any | None:
    """Launch IsaacLab using the real AppLauncher.

    Returns the underlying SimulationApp handle when Omniverse mode is required.
    In IsaacLab standalone headless mode, AppLauncher intentionally returns ``None``.
    """

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless, **kwargs)
    return app_launcher.app
