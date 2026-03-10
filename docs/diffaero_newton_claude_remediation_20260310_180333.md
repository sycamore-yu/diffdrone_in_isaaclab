# `diffaero_newton/` Claude Remediation Plan

Timestamp: `2026-03-10 18:03:33 +0800`

This document is the execution brief for the next Claude update cycle.

Scope:

- include `Phase 1`, `Phase 2`, `Phase 4`, `Phase 5`
- exclude `Phase 3`

`Phase 3` is intentionally split out for Gemini and documented separately in:

- `docs/diffaero_newton_phase3_gemini_prep_20260310_180333.md`

## Verified Current State

Validated in `isaaclab-newton`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-newton
export PYTHONPATH=diffaero_newton/source

pytest -q diffaero_newton/tests/test_obstacle_training.py
python diffaero_newton/run_training.py --num_envs 4 --num_iterations 1 --save_interval 1000
```

Current status:

- `pytest` passes with `15 passed`
- the training CLI completes a 1-iteration smoke run

This is only a runtime baseline. It does not mean the implementation is semantically aligned with `reference/diffaero/`.

## What Must Be Fixed

### 1. Obstacle task integration needs cleanup and hardening

Current reality:

- `ObstacleManager` is now wired into `DroneEnv`
- nearest-obstacle distance is included in policy observation
- obstacle collision affects termination
- obstacle proximity affects reward
- obstacle diagnostics are emitted in `extras`

Remaining problems:

- obstacle reward logic is embedded directly inside `DroneEnv._get_rewards()` instead of being cleanly routed through task helpers
- obstacle diagnostics and reset behavior need stronger consistency guarantees
- the code path still mixes “environment bookkeeping” and “task semantics” too tightly

Required code work:

- refactor obstacle-related reward and diagnostics into reusable task-side helpers where practical
- make obstacle reset semantics explicit for partial resets
- ensure obstacle metrics in `extras` are always defined after `reset()` and `step()`
- keep policy observation, reward, done, and diagnostics numerically consistent for the same state

Files to update:

- `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
- `diffaero_newton/source/diffaero_newton/tasks/obstacle_manager.py`
- `diffaero_newton/source/diffaero_newton/tasks/reward_terms.py`
- `diffaero_newton/source/diffaero_newton/tasks/observations.py`

Acceptance criteria:

- changing only obstacle distance changes obstacle penalty in the expected direction
- obstacle collision triggers termination deterministically
- partial env reset reinitializes obstacle state only for the requested env ids
- `extras["obstacles"]` is available after every `step()`

### 2. Dynamics implementation is not yet safe to call “complete”

Confirmed issues:

- `Drone.__init__()` initializes quaternion identity inconsistently with `reset_states()`
- `RK4` position integration is wrong: position is updated using acceleration terms rather than velocity integration
- rollout helpers still discard quaternion, linear velocity, and angular velocity when restoring state
- the implementation still does not match important `reference/diffaero` semantics such as full-state rollout fidelity

Required code work:

- unify quaternion convention across constructor, reset, observation, and integrator code paths
- fix `_rk4_step()` so position is integrated from velocity, not from acceleration
- update rollout helpers to restore and evolve the full 13D state
- add tests that exercise nontrivial attitude and angular-rate evolution
- if the implementation remains PyTorch-only, keep the docs explicit about that fact

Files to update:

- `diffaero_newton/source/diffaero_newton/dynamics/drone_dynamics.py`
- `diffaero_newton/source/diffaero_newton/dynamics/rollout.py`
- `diffaero_newton/tests/test_obstacle_training.py`

Acceptance criteria:

- Euler and RK4 both preserve valid quaternion normalization
- RK4 no longer uses acceleration as a surrogate for position derivative
- `rollout_onestep()` and `rollout_horizon()` preserve all 13 state dimensions
- tests cover at least one case where asymmetric thrust changes attitude or angular velocity

### 3. Docs and public interfaces are still inconsistent

Confirmed issues:

- `DroneEnvCfg.__init__()` sets `observation_space = 20`
- `DroneEnv` actually exposes a 21D observation
- `docs/api.md` still documents a 20D observation
- `README.md` still overstates “short-horizon risk loss”
- `DroneEnv` docstrings still describe `state` and `loss_terms` outputs that are not actually returned

Required code/doc work:

- make config, runtime, and docs agree on the true observation dimension
- remove or clearly mark planned-but-not-implemented output channels
- describe the training loop as PPO-style / SHAC-style only, not true differentiable SHAC
- describe the dynamics as PyTorch differentiable quadrotor dynamics, not true Newton-backed dynamics

Files to update:

- `diffaero_newton/source/diffaero_newton/configs/drone_env_cfg.py`
- `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
- `diffaero_newton/README.md`
- `diffaero_newton/docs/api.md`
- `diffaero_newton/docs/quickstart.md`

Acceptance criteria:

- config examples, runtime observation shape, and API docs all report the same dimension
- no doc claims `loss_terms` or critic `state` outputs unless the env really returns them
- no doc claims true Newton integration unless runtime code actually uses Newton

### 4. Git hygiene is improved but not complete

Confirmed issues:

- at least one `.pyc` file is still tracked:
  - `diffaero_newton/tests/__pycache__/conftest.cpython-311-pytest-9.0.2.pyc`
- generated checkpoints must not be committed
- final remediation should leave a reviewable tree without bytecode artifacts

Required code/repo work:

- remove tracked `.pyc` and `__pycache__` entries from version control
- keep `.gitignore` aligned with actual generated artifacts
- do not commit `checkpoints/`
- before final handoff, verify only intentional source/doc changes remain

Files to update:

- `.gitignore`
- git index for tracked bytecode artifacts

Acceptance criteria:

- `git ls-files '*__pycache__*' '*.pyc'` returns nothing
- `git status --short` contains only intentional changes

## Recommended Execution Order

1. Fix dynamics correctness first.
2. Then clean obstacle/task wiring.
3. Then reconcile config/docs/runtime interfaces.
4. Finish with git hygiene cleanup.

Reason:

- dynamics bugs affect both tests and future training behavior
- doc cleanup is easier once runtime behavior is stable
- git cleanup should be the last pass before commit

## Required Verification

Run these after the changes:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-newton
export PYTHONPATH=diffaero_newton/source

pytest -q diffaero_newton/tests/test_obstacle_training.py
python diffaero_newton/run_training.py --num_envs 4 --num_iterations 1 --save_interval 1000
git ls-files '*__pycache__*' '*.pyc'
git status --short
```

Add or extend tests for:

- quaternion initialization consistency
- RK4 position integration correctness
- rollout full-state preservation
- obstacle collision termination
- obstacle-distance reward monotonicity

## Explicit Non-Goals For This Claude Pass

Do not claim `Phase 3` is fixed here.

Do not rewrite the training algorithm to reference SHAC in this pass.

Do not restore Newton/Warp marketing language unless runtime code truly depends on it.
