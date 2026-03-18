---
title: Architecture Constraints
readMode: optional
priority: high
category: planning
scope: project
dimension: specs
keywords: [planning, architecture, runtime, registry, differentiable-rl]
---

# Architecture Constraints

- Keep the runtime split explicit: real IsaacLab launch belongs in `common/isaaclab_launch.py`, while the Newton-only headless contract belongs in `common/direct_rl_shim.py`.
- Do not add new broad fallback layers that silently mask IsaacLab or Newton runtime failures.
- Treat `scripts/registry.py` as the only supported public assembly point for environment, dynamics, and algorithm wiring.
- Keep solver, model, and state ownership centralized inside the dynamics layer. Do not scatter Newton lifecycle construction across tasks or trainers.
- Keep Warp and Newton kernels numeric-only. Reset logic, truncation, curriculum, and logging stay in Python orchestration.
- Keep differentiable `loss_terms` separate from detached RL-facing `reward`.
- Keep obstacle geometry queries reusable by both observations and risk loss. Do not bury geometry access inside one sensor-only path.
- Batch over environments by default. Avoid per-environment Python loops in hot rollout, sensing, or reset paths.
- Keep docs roles explicit: `docs/development.md` is target architecture, `docs/progress.md` is current implementation truth, and `bd` is the task tracker.
