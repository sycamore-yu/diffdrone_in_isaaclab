---
title: Quality Rules
readMode: required
priority: high
category: execution
scope: project
dimension: specs
keywords: [execution, quality, pytest, runtime_preflight, cpu_smoke, gpu_smoke]
---

# Quality Rules

- Use the pytest marker taxonomy in `pytest.ini` as the repo-wide contract: `runtime_preflight`, `cpu_smoke`, and `gpu_smoke`.
- Any change to registry, CLI, script entry, or runtime launch wiring must add or update `runtime_preflight` coverage.
- Any change to environment, observation, state, or reset contracts must add or update `cpu_smoke` coverage.
- Any change to differentiable rollout, Newton/Warp stepping, or gradient-carrying training paths must add or update `gpu_smoke` coverage.
- Every new top-level capability must be exercisable through `diffaero_newton/source/diffaero_newton/scripts/train.py`.
- No duplicate config fields or public legacy aliases in config objects.
- No new `sys.path` hacks in tests unless the test is explicitly validating script-entry behavior.
- Before claiming a migration step complete, run the relevant `isaaclab-newton` command path and record whether it passes, fails, or is blocked in `docs/progress.md`.
