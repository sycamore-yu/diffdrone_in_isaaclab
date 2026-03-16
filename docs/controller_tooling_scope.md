# DiffAero Control Semantics And Tooling Scope

This note records the implementation decision for the remaining DiffAero parity gap around quadrotor control semantics and the old Hydra-heavy tooling stack.

## Current Mainline Reality

Current `main` has two different levels of maturity:

- The environment/task/training path is usable through the unified `diffaero_newton/scripts/train.py` entry.
- The quadrotor action semantics and experiment tooling are still materially simpler than the reference DiffAero stack.

The main gaps are:

- quadrotor control currently treats the action as normalized per-motor thrust and feeds it directly into the Newton-backed dynamics
- there is no explicit controller abstraction for body-rate or attitude-target control
- the repository no longer has a Hydra-first train/test/export workflow
- Optuna, WandB, and deploy/export surfaces are not part of the mainline workflow

## Decision

### 1. Control Semantics

Do not silently replace the existing motor-thrust action path in the current environments.

Reason:

- the existing environments, tests, and rollout code already assume direct normalized motor-thrust actions
- replacing that contract in-place would invalidate current smoke coverage and make regression attribution hard
- DiffAero parity requires an explicit controller layer, not an implicit reinterpretation of the same action tensor

Adopt this target migration shape instead:

- keep `motor_thrust` as the current default control mode
- introduce an explicit controller module for rate-controller semantics
- make future controller-backed semantics opt-in via config, for example `control_mode=bodyrate` or equivalent
- only route tasks to the controller path once dedicated tests and smoke commands exist

What is in scope next:

- add a controller abstraction between policy action and motor thrust
- support at least one controller-backed quadrotor mode that mirrors DiffAero's rate-oriented semantics closely enough for task parity work
- preserve current `motor_thrust` behavior as a stable baseline

What is out of scope for this task:

- rewriting all existing tasks to the controller path immediately
- changing current action semantics in-place without an opt-in mode

### 2. Hydra And Experiment Tooling

Do not restore the full DiffAero Hydra/Optuna/WandB/export stack as a prerequisite for the main training path.

Reason:

- the unified entry is already the only validated workflow on `main`
- the algorithm and environment parity gaps are still larger than the tooling gap
- restoring the whole tooling surface now would create a large amount of glue around unstable runtime contracts

Adopt this migration order instead:

- keep `scripts/train.py` as the stable execution path
- add a thin Hydra-compatible wrapper only after the env/algo contracts it drives are stable
- split experiment services into separate follow-up work:
  - Hydra config wrapper and sweep entry
  - logging backends such as WandB
  - search backends such as Optuna
  - export/deploy surfaces

What is in scope next:

- define a minimal structured-config wrapper over the current registry-backed train entry
- make sure any future Hydra layer is a front-end to the current mainline API, not a second training stack

What is explicitly deferred:

- Optuna tuning integration
- WandB logging integration
- export/deploy tooling

## Recommended Follow-Up Execution

The next implementation work should land in this order:

1. Introduce explicit quadrotor control-mode semantics with `motor_thrust` preserved as the baseline.
2. Add a thin Hydra wrapper that delegates to the current unified registry-backed train entry.
3. Add optional experiment services only after the first two layers are stable.

## Acceptance Mapping For i8x.16

This task is considered complete when:

- the migration decision is explicit and reviewable
- controller semantics and tooling are split into concrete follow-up issues
- `docs/progress.md` reflects that controller/tooling parity is now a scoped follow-up rather than an ambiguous gap
