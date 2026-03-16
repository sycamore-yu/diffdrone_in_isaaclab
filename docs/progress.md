# Newton + IsaacLab DiffAero Implementation Status

This document serves as the ground truth for the current actual migration status of the `diffaero_newton` module.

## Overall Status
- **Target Setup**: `isaaclab-newton` conda environment.
- **Goal**: Migrate DiffAero capabilities (Dynamics, Algorithms, Environments, Sensors) to Newton + IsaacLab.
- **Mainline Reality**: The unified training entry on `main` is usable for `apg`, `apg_sto`, `ppo`, `appo`, and `shac`, but DiffAero parity is still incomplete in dynamics semantics, world-model integration, and tooling.

## Capability Matrix Status Tracker

### Algorithms
- [x] **SHAC**: Implemented (short-horizon actor-critic with Newton differentiable physics)
- [x] **APG / APG_sto**: Implemented (deterministic APG + stochastic APG with entropy regularisation)
- [x] **PPO / Asymmetric PPO**: Implemented (clipped surrogate with GAE, privilege state critic)
- [ ] **SHA2C**: Not migrated on main
- [ ] **MASHAC**: Not migrated on main
- [ ] **DreamerV3 / world**: Partially present under `training/dreamerv3`, but not wired into `scripts/registry.py` or the unified training entry; treat as not landed capability on `main`

### Dynamics
- [x] **PointMass**: Implemented with explicit `pointmass` (backward-compatible alias), `continuous_pointmass`, and `discrete_pointmass` model options. Low-level differentiable propagation now passes `test_pointmass_dynamics.py`, and both model variants are wired into the unified training entry.
- [x] **Quadrotor**: Implemented with Newton + Warp autograd bridge. Low-level action-to-state backprop works on `main`, and obstacle-task differentiable loss tests pass.
- [ ] **DiffAero Quadrotor Semantics**: Not yet matched 1:1. Current mainline quadrotor uses direct normalized motor thrust input and does not yet reproduce DiffAero's rate-controller-oriented semantics or full aerodynamic model details.

### Tasks/Environments
- [x] **Position Control**: Implemented (Single-agent target position tracking)
- [x] **Multi-Agent Position Control**: Implemented (Multi-agent with collision rewards and proper shape flattening)
- [x] **Obstacle Avoidance**: Implemented (ObstacleAvoidanceEnv with multi-modal sensor integration)
- [x] **Racing**: Implemented (figure-8 gate track with gate passing detection)
- [x] **Sim2RealPositionControl**: Implemented as `sim2real_position_control` with square-target switching derived from DiffAero's reference environment; validated on the unified entry with point-mass dynamics.

### Sensors
- [x] **Relative Position (relpos)**: Implemented (sorted nearest-obstacle relative positions)
- [x] **Camera**: Implemented (depth-map ray-casting, configurable FOV & resolution)
- [x] **Lidar**: Implemented (360° ray-casting with vertical/horizontal coverage)

## Gaps Relative to Reference DiffAero
- Missing algorithms on main: `SHA2C`, `MASHAC`, and a registry-backed `world` / DreamerV3 path.
- Missing dynamics parity: full DiffAero-like quadrotor control semantics, and clearer frame/control abstractions from `reference/diffaero/dynamics`.
- Missing environment parity: no major gap remains for the DiffAero sim-to-real square-target position-control variant, but broader deployment/tooling parity is still missing.
- Missing tooling parity: Hydra-based train/test/export workflow, sweep tooling, Optuna / WandB integration, and export/deploy utilities.

## Validation & Commands
Currently verified on `main`:
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q` (Unified training entry / registry smoke)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q` (PPO / APPO training checks)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_position_control.py -q` (Position control environment smoke)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_sim2real_position_control.py -q` (Sim2Real square-target position control smoke)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q` (Obstacle env + SHAC integration, including differentiable-loss path)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_dynamics.py -q` (Low-level point-mass differentiability for alias/continuous/discrete variants)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q` (Point-mass environment propagation smoke for alias/continuous/discrete variants)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --list` (Unified training entry import/registry smoke test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics pointmass --max_iter 3 --log_interval 1 --n_envs 8 --l_rollout 4` (Env-backed unified training entry smoke test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics continuous_pointmass --max_iter 1 --l_rollout 1 --n_envs 2 --device cpu --log_interval 1` (Continuous point-mass unified entry smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics discrete_pointmass --max_iter 1 --l_rollout 1 --n_envs 2 --device cpu --log_interval 1` (Discrete point-mass unified entry smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env sim2real_position_control --dynamics pointmass --max_iter 1 --l_rollout 2 --n_envs 4 --device cpu --log_interval 1` (Sim2Real unified-entry smoke test)
