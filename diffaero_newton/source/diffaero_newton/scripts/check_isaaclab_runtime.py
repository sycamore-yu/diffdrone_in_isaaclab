"""Preflight checks for IsaacLab runtime availability on the host environment."""

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
    print(json.dumps(results, indent=2))
    if not all(item["ok"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
