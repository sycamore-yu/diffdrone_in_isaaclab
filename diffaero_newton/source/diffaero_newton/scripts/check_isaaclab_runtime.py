"""Launch-layer preflight checks for IsaacLab availability on the host environment.

This script intentionally validates only the real IsaacSim/IsaacLab launcher path used
by this repository's entrypoints. It does not claim that the full IsaacLab env/sim
stack is importable or stable on the current machine.
"""

from __future__ import annotations

import json
from typing import Any


def _check_import(name: str):
    __import__(name)


def _check_applauncher():
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app
    if app is not None:
        app.close()


def run_checks() -> list[dict[str, Any]]:
    checks = [
        ("import_isaacsim", lambda: _check_import("isaacsim")),
        ("import_isaaclab_app", lambda: _check_import("isaaclab.app")),
        ("launch_applauncher_headless", _check_applauncher),
    ]

    results = []
    for name, fn in checks:
        try:
            fn()
            results.append({"check": name, "ok": True})
        except Exception as exc:  # keep failures inspectable without swallowing them silently
            results.append({"check": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return results


def main():
    results = run_checks()
    payload = {
        "scope": "launch_preflight",
        "validates": [
            "isaacsim import",
            "isaaclab.app import",
            "headless AppLauncher startup",
        ],
        "does_not_validate": [
            "full isaaclab.envs / isaaclab.sim stack importability",
            "task-specific environment construction",
            "GPU training correctness",
        ],
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if not all(item["ok"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
