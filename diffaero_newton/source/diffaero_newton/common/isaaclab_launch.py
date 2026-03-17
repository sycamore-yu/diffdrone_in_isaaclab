"""Runtime launcher helpers for real IsaacLab execution."""

from __future__ import annotations

from typing import Any

from isaaclab.app import AppLauncher


def add_app_launcher_args(parser) -> None:
    """Register IsaacLab launcher CLI flags on an argparse parser."""

    AppLauncher.add_app_launcher_args(parser)


def launch_app(args: Any | None = None, headless: bool = True, **kwargs) -> Any | None:
    """Launch IsaacLab using the real AppLauncher.

    Returns the underlying SimulationApp handle when Omniverse mode is required.
    In IsaacLab standalone headless mode, AppLauncher intentionally returns ``None``.
    """

    if args is not None:
        app_launcher = AppLauncher(args)
    else:
        app_launcher = AppLauncher(headless=headless, **kwargs)
    return app_launcher.app
