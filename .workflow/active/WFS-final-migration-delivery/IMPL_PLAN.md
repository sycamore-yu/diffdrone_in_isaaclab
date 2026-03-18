# Implementation Plan

## 1. Requirements Summary

Deliver the remaining DiffAero -> Newton/IsaacLab migration work from a dedicated worktree and land it in one PR. The delivery bar is not "most features exist"; it is "the remaining scope is either implemented with executable proof or explicitly de-scoped in docs with tests and rationale."

Mandatory execution loop for every implementation slice:

1. Write code for one bounded capability.
2. Run the relevant `isaaclab-newton` tests / commands.
3. Review against `ai-first-engineering` principles with regression focus.
4. Update `docs/progress.md` and `docs/regression_matrix.md` when implementation truth changes.
5. Commit only after the slice has executable evidence.

## 2. Architecture Decisions

- Use `scripts/registry.py` and `scripts/train.py` as the only supported public assembly points.
- Keep runtime split explicit: IsaacLab runtime launch in `common/isaaclab_launch.py`; Newton-only RL shim in `common/direct_rl_shim.py`.
- Keep differentiable rollout ownership in the dynamics / training layers; do not bury task semantics inside kernels.
- Use absolute root reference paths for `reference/diffaero`, `reference/newton`, and `reference/IsaacLab` because the new worktree's relative `reference/` directories are empty placeholders.
- Treat docs as first-class delivery artifacts:
  - `docs/development.md`: target architecture
  - `docs/progress.md`: actual implementation truth
  - `docs/regression_matrix.md`: concise validation and parity matrix

## 3. Task Breakdown

### IMPL-1: Re-baseline docs and validation gates

Scope:
- Remove stale contradictions left in `docs/progress.md`.
- Align `docs/regression_matrix.md` with the verified merged-main state.
- Reconfirm the quadrotor unified-entry fix and current gate commands.

Acceptance:
- `test_train_entry.py` passes in `isaaclab-newton`.
- The quadrotor body-rate unified-entry smoke passes in `isaaclab-newton`.
- Docs no longer claim the same path is both fixed and blocked.

### IMPL-2: Migrate SHA2C into the unified entry

Scope:
- Port the missing `SHA2C` algorithm from the DiffAero reference into the Newton/IsaacLab stack.
- Wire it through `scripts/registry.py` and `scripts/train.py`.
- Add targeted tests covering construction and one-step training smoke.

Acceptance:
- `--algo sha2c` is available in the public CLI and registry.
- At least one `isaaclab-newton` smoke command runs through `scripts/train.py --algo sha2c`.
- Docs mark `SHA2C` as implemented only after the CLI path and test coverage pass.

### IMPL-3: Close quadrotor semantic parity gap

Scope:
- Compare current `body_rate` / `motor_thrust` behavior against the DiffAero reference.
- Decide whether the remaining gap is solvable in this branch or should be explicitly de-scoped.
- If solvable, implement the missing controller semantics and regression tests without reintroducing the already-fixed registry regression.

Acceptance:
- Quadrotor control semantics are either implemented end-to-end or explicitly limited in docs with precise language.
- `test_train_entry.py`, `test_drone_dynamics.py`, and unified-entry quadrotor smoke still pass.
- No regression in `position_control` / `obstacle_avoidance` quadrotor paths.

### IMPL-4: Expand or explicitly narrow obstacle/sensor parity

Scope:
- Compare current obstacle/sensor stack with the DiffAero reference semantics.
- Decide what can be implemented in this branch:
  - mixed geometry
  - IMU state
  - mount/randomization
  - richer obstacle/state contracts
- Implement the feasible subset and explicitly de-scope the rest with tests/docs.

Acceptance:
- `test_obstacle_training.py` remains green.
- New or updated tests prove the chosen obstacle/sensor contract.
- `docs/progress.md` clearly distinguishes implemented behavior from deferred parity.

### IMPL-5: Expand DreamerV3/world beyond the current state-only limit or formally de-scope it

Scope:
- Audit the existing perception plumbing in `training/dreamerv3`.
- Decide whether perception-backed world training can be made real on this branch.
- If yes, wire at least one supported path end-to-end.
- If no, document the exact scope limit and keep tests aligned with the real contract.

Acceptance:
- `test_world_training.py` passes.
- Public CLI behavior for `--algo world` matches the documented scope.
- No stale docs claim state-only while code silently expects perception, or vice versa.

### IMPL-6: Deliver minimal tooling parity

Scope:
- Define the smallest supportable tooling surface for this branch:
  - export
  - WandB / Optuna hooks
  - Hydra-like train wrapper or an explicit statement that it is deferred
- Prefer a minimal but real interface over placeholders.

Acceptance:
- A user can discover the supported tooling from docs and the repository entrypoints.
- Any implemented tooling path has at least one command-level smoke or unit test.
- Any non-implemented tooling path is explicitly marked deferred instead of implied.

### IMPL-7: Final delivery gate and PR prep

Scope:
- Run the relevant runtime_preflight / cpu_smoke / gpu_smoke subsets for touched surfaces.
- Perform an `ai-first-engineering` review pass over the diff with regression focus.
- Update docs to final truth and prepare one PR only after the branch reaches the delivery bar.

Acceptance:
- Final docs and tests agree.
- Review notes explicitly cover behavior regressions, failure handling, rollout safety, and contract drift.
- Exactly one PR is opened from `feature/final-migration-delivery`.

## 4. Implementation Strategy

Sequential with hard gate closures:

1. Re-baseline docs and current validation truth first.
2. Add missing algorithm surface (`SHA2C`) because it is the clearest binary gap.
3. Address quadrotor parity next because it is a central correctness and regression risk area.
4. Resolve obstacle/sensor and world/perception scope with explicit evidence rather than vague future work.
5. Finish tooling parity only after core algorithm / env / dynamics truth is stable.
6. Run final quality gates and PR prep as a separate closing slice.

No task may skip docs or review. If a capability is intentionally deferred, that deferment still requires tests/docs alignment.

## 5. Risk Assessment

- Highest risk: trying to "finish everything" by widening scope without maintaining regression gates.
- Second risk: docs drift, especially on quadrotor and world scope, because merged `main` still contains stale contradictory text.
- Third risk: reference mismatch, since the worktree-local `reference/` directories are empty and absolute root references must be used deliberately.

## 6. Per-Task Closure Contract

For every IMPL task:

- Code:
  - change only the files required for that slice
  - preserve stable public contracts where possible
- Validation:
  - run the relevant `isaaclab-newton` tests or commands
  - record pass/fail/block explicitly
- AI-first review:
  - check for behavior regressions
  - check interface contract drift
  - check rollout / training safety
  - request extra tests if changed behavior lacks evidence
- Docs:
  - update `docs/progress.md`
  - update `docs/regression_matrix.md` if validation truth changes
- Commit:
  - one commit per closed slice
  - do not advance to the next slice if the current one lacks executable evidence
