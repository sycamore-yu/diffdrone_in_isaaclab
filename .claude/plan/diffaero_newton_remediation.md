# Implementation Plan: diffaero_newton Remediation

## Overview

Based on the review in `docs/diffaero_newton_review_and_remediation.md`, this plan addresses 5 phases of remediation for the `diffaero_newton` project.

## Current Status

- Tests pass (11/11)
- Basic training runs
- **BUT**: Semantic implementation doesn't match reference DiffAero

---

## Phase 1: Wire Obstacle Task Semantics into DroneEnv

### Goal
Turn `DroneEnv` from a goal-tracking env into an actual obstacle-avoidance task.

### Required Changes

1. **Integrate ObstacleManager into DroneEnv**
   - File: `diffaero_newton/source/diffaero_newton/envs/drone_env.py`
   - Add `ObstacleManager` instantiation in `__init__`
   - Store obstacle manager as `self.obstacle_manager`

2. **Add Obstacle-Aware Observations**
   - Modify `_get_observations()` to include:
     - Nearest obstacle distance
     - Obstacle positions in observation space
   - Update observation dimension (currently 20 → needs expansion)

3. **Integrate Collision into Termination**
   - Modify `_get_dones()` to check:
     - Ground collision (existing)
     - Obstacle collision via `obstacle_manager.check_collisions()`

4. **Add Obstacle Rewards**
   - Modify `_get_rewards()`:
     - Add collision penalty
     - Add progress reward toward goal
     - Add distance-to-obstacle penalty

5. **Expose Diagnostics in Extras**
   - Add collision count, nearest distance, goal distance to `extras`

### Key Files
| File | Operation | Description |
|------|-----------|-------------|
| `envs/drone_env.py` | Modify | Add obstacle integration |
| `configs/drone_env_cfg.py` | Modify | Add obstacle config |

### Acceptance Criteria
- [ ] Environment reward changes when obstacle proximity changes
- [ ] Collisions with obstacles trigger termination
- [ ] `extras` exposes collision diagnostics

---

## Phase 2: Upgrade Dynamics Toward DiffAero Quadrotor Semantics

### Goal
Replace simplified integrator with true quadrotor model matching `reference/diffaero/dynamics/quadrotor.py`.

### Required Changes

1. **Add Full State Integration**
   - File: `diffaero_newton/source/diffaero_newton/dynamics/drone_dynamics.py`
   - Implement quaternion derivative in `integrate()`
   - Add angular velocity dynamics with inertia

2. **Implement Control Allocation**
   - Map 4 motor inputs to thrust + torque
   - Add `_tau_thrust_matrix` (reference lines 72-81)
   - Implement `RateController` pattern

3. **Add Physics Components**
   - Inertia tensor (J_xy, J_z)
   - Drag coefficients (D_xy, D_z)
   - Gravity compensation

4. **Add Solver Options**
   - Support Euler and RK4 integrators
   - Add `n_substeps` configuration

### Key Files
| File | Operation | Description |
|------|-----------|-------------|
| `dynamics/drone_dynamics.py` | Rewrite | Full quadrotor dynamics |
| `dynamics/rollout.py` | Modify | Support full state rollout |

### Acceptance Criteria
- [ ] State includes position, quaternion, linear velocity, angular velocity
- [ ] Rollout preserves full state without silent discarding
- [ ] Unit tests cover orientation/angular-rate updates

---

## Phase 3: Align Training Semantics with Reference SHAC

### Goal
Move from PPO-style to true SHAC with differentiable loss accumulation.

### Required Changes

1. **Modify Environment Output Contract**
   - File: `envs/drone_env.py`
   - Add `loss` to step return (differentiable risk term)
   - Add `next_obs_before_reset` or `next_state_before_reset` to extras

2. **Rewrite SHAC Agent**
   - File: `diffaero_newton/source/diffaero_newton/training/shac.py`
   - Replace PPO clipped objective with differentiable loss accumulation
   - Implement `record_loss()` method (reference lines 109-126)
   - Accumulate `cumulated_loss` across horizon

3. **Update Buffer**
   - File: `diffaero_newton/source/diffaero_newton/training/buffer.py`
   - Add `loss` storage alongside `rewards`
   - Add `next_obs_before_reset` storage

4. **Implement Terminal Bootstrap**
   - Use environment-provided `next_obs_before_reset` for bootstrap
   - Implement reference `bootstrap_gae()` pattern (lines 94-107)

### Key Files
| File | Operation | Description |
|------|-----------|-------------|
| `envs/drone_env.py` | Modify | Add loss output |
| `training/shac.py` | Rewrite | True SHAC implementation |
| `training/buffer.py` | Modify | Add loss storage |

### Acceptance Criteria
- [ ] Actor update uses horizon loss accumulation
- [ ] Training explains mapping to reference SHAC
- [ ] Tests cover horizon accumulation edge cases

---

## Phase 4: Bring Interfaces and Docs into Alignment

### Goal
Ensure docs describe actual implementation and match code behavior.

### Required Changes

1. **Update README.md**
   - Clarify current backend is PyTorch-only
   - Remove Newton/Warp claims until actually used
   - Document current limitations

2. **Update docs/api.md**
   - Fix observation dimensions to match actual code
   - Document actual vs claimed capabilities

3. **Update docs/quickstart.md**
   - Ensure commands run in `isaaclab-newton`
   - Fix any outdated instructions

4. **Update CLAUDE.md**
   - Document simplified dynamics approach
   - Clarify training is PPO-style (temporarily)

### Key Files
| File | Operation | Description |
|------|-----------|-------------|
| `README.md` | Modify | Fix claims |
| `docs/api.md` | Modify | Fix dimensions |
| `docs/quickstart.md` | Modify | Fix commands |
| `CLAUDE.md` | Modify | Document limitations |

### Acceptance Criteria
- [ ] No doc claims conflict with code behavior
- [ ] Quickstart commands work in `isaaclab-newton`

---

## Phase 5: Clean Git Hygiene Before Final Landing

### Goal
Leave reviewable tree with no generated junk.

### Required Changes

1. **Add .gitignore**
   - Add if missing
   - Include: `__pycache__`, `*.pyc`, `checkpoints/`, `.pytest_cache/`

2. **Clean Tracked Files**
   - Remove any tracked `.pyc` files
   - Remove `__pycache__` from git

3. **Verify Clean State**
   - `git status` shows only intended changes

### Key Files
| File | Operation | Description |
|------|-----------|-------------|
| `.gitignore` | Create/Modify | Add ignore rules |

### Acceptance Criteria
- [ ] `git status` shows only source/doc changes
- [ ] No Python bytecode tracked

---

## Test Commands

All testing in `isaaclab-newton` conda environment:

```bash
conda activate isaaclab-newton
export PYTHONPATH=diffaero_newton/source

# Phase 1-3 tests
pytest -q diffaero_newton/tests/test_obstacle_training.py
python diffaero_newton/run_training.py --num_envs 4 --num_iterations 1 --save_interval 1000
```

---

## Implementation Order

1. Phase 1 (Obstacle Integration) - 2-3 hours
2. Phase 2 (Dynamics Upgrade) - 4-5 hours
3. Phase 3 (SHAC Alignment) - 3-4 hours
4. Phase 4 (Docs) - 1-2 hours
5. Phase 5 (Git Hygiene) - 0.5 hours

**Total Estimated: 11-15 hours**
