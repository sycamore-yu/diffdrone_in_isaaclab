# Newton + IsaacLab DiffAero Implementation Status

This document serves as the ground truth for the current actual migration status of the `diffaero_newton` module.

## Overall Status
- **Target Setup**: `isaaclab-newton` conda environment.
- **Goal**: Migrate DiffAero capabilities (Dynamics, Algorithms, Environments, Sensors) to Newton + IsaacLab.
- **Mainline Reality**: The unified training entry on `main` is usable for `apg`, `apg_sto`, `ppo`, `appo`, `shac`, `mashac`, and `world`, and the environment registry now exposes explicit obstacle/racing observation contracts. Recent fixes restored missing point-mass control semantics needed by racing, migrated SHA2C into the unified entry, aligned the quadrotor body-rate controller with DiffAero's normalized-action-to-physical-command semantics, and fixed a detached-observation bug in multi-iteration APG training, but broader parity is still incomplete in world-model breadth, env/sensor realism, and tooling. The env-backed quadrotor unified-entry path has been fixed: `dynamics/registry.py` no longer passes unsupported `action_frame` into `DroneConfig`, and the quadrotor body-rate runtime-preflight path now passes tests.
- **Runtime Note**: Mainline now separates real IsaacLab launch (`common/isaaclab_launch.py`) from the project's Newton-only DirectRL shim (`common/direct_rl_shim.py`). The old broad runtime fallback in `isaaclab_compat.py` has been reduced to a legacy import bridge, and the package `__main__` entry now reuses the same launch helper with a single resolved device path shared by both environment and trainer.

## Capability Matrix Status Tracker

### Algorithms
- [x] **SHAC**: Implemented (short-horizon actor-critic with Newton differentiable physics)
- [x] **APG / APG_sto**: Implemented (deterministic APG + stochastic APG with entropy regularisation)
- [x] **PPO / Asymmetric PPO**: Implemented (clipped surrogate with GAE, privilege state critic)
- [x] **SHA2C**: Implemented and wired into the unified training entry as `--algo sha2c`. Current validated path is the privileged-state `position_control` point-mass rollout on CPU via `test_shac_training.py` and a direct `train.py --algo sha2c` smoke command.
- [x] **MASHAC**: Implemented for `mapc` with shared actor + centralized critic, wired into the unified training entry as `--algo mashac`
- [x] **DreamerV3 / world**: Wired into `scripts/registry.py` and the unified training entry as `--algo world`. Validation now covers both the original state-only `position_control` rollout and a perception-enabled `obstacle_avoidance` camera path on CPU, with unified-entry smoke for both; broader task parity is still incomplete beyond those surfaces.

### Dynamics
- [x] **PointMass**: Implemented with explicit `pointmass` (backward-compatible alias), `continuous_pointmass`, and `discrete_pointmass` model options. Low-level differentiable propagation now passes `test_pointmass_dynamics.py`, and both model variants are wired into the unified training entry.
- [x] **PointMass action_frame**: Added `action_frame` config option ('world'/'local') to `DiscretePointMass`, with quaternion-to-rotation-matrix conversion for local frame transformation.
- [x] **Quadrotor**: Implemented with Newton + Warp autograd bridge. Low-level action-to-state backprop works on `main`, and obstacle-task differentiable loss tests pass.
- [x] **Quadrotor body-rate controller semantics**: The unified quadrotor path now consumes DiffAero-style controller fields end-to-end. `control_mode`, `k_angvel`, `min_body_rates`, `max_body_rates`, `min_normed_thrust`, `max_normed_thrust`, `torque_ratio`, `thrust_ratio`, and `compensate_gravity` now flow from config/registry/CLI into the Newton rate controller, and regression coverage validates the resulting thrust/torque equations.
- [ ] **DiffAero Quadrotor Parity**: Partially matched. The reference body-rate controller semantics are now aligned on the unified entry, but full aerodynamic/randomization breadth and the extra Newton-only `motor_thrust` mode still differ from the reference repository.

### Tasks/Environments
- [x] **Position Control**: Implemented (Single-agent target position tracking)
- [x] **Multi-Agent Position Control**: Implemented (Multi-agent with collision rewards and proper shape flattening)
- [x] **Obstacle Avoidance**: Implemented at coarse capability level (ObstacleAvoidanceEnv with multi-modal sensor integration). Camera/LiDAR obstacle observations now consume the drone's current orientation end-to-end instead of assuming an identity sensor pose, but the task is still narrower than the reference implementation on obstacle geometry, reset/randomization depth, and richer perception/state contracts.
- [x] **Racing**: Implemented and trainer-validated for the point-mass path. Figure-8 gate geometry, gate-frame observations, gate pass/collision detection, target-gate advancement, gate-frame point-mass action handling, reset semantics aligned to the reference task, reward/loss wiring, and dedicated racing smoke coverage now exist on `main`. The current reward path no longer depends on a hidden per-episode target speed when `use_vel_track=False`, and `reward_progress` is again distinct from loss weighting. Manual CUDA validation with APG continues to produce non-zero gate passes in short runs after those fixes.
- [x] **Sim2RealPositionControl**: Implemented as `sim2real_position_control` with square-target switching derived from DiffAero's reference environment; validated on the unified entry with point-mass dynamics.

### Sensors
- [x] **Relative Position (relpos)**: Implemented (sorted nearest-obstacle relative positions), but still simplified relative to the reference obstacle/state contract.
- [x] **Camera**: Implemented (depth-map ray-casting, configurable FOV & resolution). Obstacle-env integration now uses the live drone quaternion, but the stack is still simplified relative to the reference mount/randomization stack.
- [x] **Lidar**: Implemented (360° ray-casting with vertical/horizontal coverage). Obstacle-env integration now uses the live drone quaternion, but the stack is still simplified relative to the reference mount/randomization stack.

## Gaps Relative to Reference DiffAero
- Known regression on main: fixed on merged `main` as of 2026-03-18. The env-backed quadrotor unified-entry construction path no longer passes unsupported `action_frame` into `DroneConfig`.
- Missing dynamics parity: the DiffAero quadrotor controller path now matches the reference command-scaling semantics, but full aerodynamic/randomization breadth and the extra Newton-only `motor_thrust` mode still differ; point-mass local/world-frame parity is also still narrower than the reference implementation.
- Missing environment parity: richer sim-to-real/deployment workflows are still absent, obstacle avoidance remains narrower than the reference task on geometry/randomization/state semantics, and racing validation currently covers the point-mass APG path rather than every algorithm/backend combination.
- Missing sensor parity: current relpos/camera/lidar paths now respect the live drone orientation in obstacle observations, but they still assume simplified spherical obstacles with no separate mount/randomization model; reference IMU support and richer sensor/randomization stacks are absent.
- Missing world-model parity: the perception-enabled obstacle/camera path now runs on CPU, but broader task coverage, stronger convergence validation, and any Hydra-style experiment workflow are still absent.
- Missing tooling parity: the unified entry now writes a resolved `run_config.json` snapshot and supports `--dry-run` / `--print-config` / `--config-out` for reproducible config inspection, but Hydra-based train/test/export workflow, sweep tooling, Optuna / WandB integration, and policy export/deploy utilities are still absent.

## Validation & Commands
Passing checks currently verified on `main`:

### Runtime / CPU smoke
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py -q` (DreamerV3/world CPU and CUDA smoke, including replay-ready world-model update coverage and Gym-style tuple unpacking)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q` (Obstacle env contract smoke on explicit CPU, sensor-aware observation contract checks, reset-triggering differentiable regression, and CUDA-only SHAC update checks when available)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q` (PPO / APPO training checks)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_mashac_training.py -q` (MASHAC multi-agent rollout/update smoke on `mapc`, now assigned to the `cpu_smoke` tier)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_racing_env.py -q` (Racing observation contract smoke and gate-pass progression/reward regression)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_drone_dynamics.py -q` (Quadrotor body-rate controller, drag model, and differentiable drone dynamics regression coverage)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_shac_training.py -q` (SHAC-family CPU smoke including the new SHA2C agent/trainer rollout/update path)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q` (Runtime-preflight coverage for the unified training entry, including the quadrotor body-rate configuration path restored on merged `main`)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q` (Point-mass env propagation/backward smoke plus normalized `x`-action symmetry regression coverage)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --list` (Unified training entry import/registry smoke test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo world --env position_control --dynamics pointmass --max_iter 1 --l_rollout 4 --n_envs 2 --device cpu --log_interval 1 --world_warmup_steps 4 --world_min_ready_steps 2 --world_batch_size 2 --world_batch_length 2 --world_imagine_length 2` (DreamerV3/world unified-entry CPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo world --env obstacle_avoidance --dynamics pointmass --sensor camera --max_iter 1 --l_rollout 4 --n_envs 2 --device cpu --log_interval 1 --world_warmup_steps 4 --world_min_ready_steps 2 --world_batch_size 2 --world_batch_length 2 --world_imagine_length 2` (DreamerV3/world unified-entry CPU smoke on the perception-enabled obstacle/camera path)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo mashac --env mapc --dynamics continuous_pointmass --max_iter 1 --l_rollout 2 --n_envs 2 --device cpu --log_interval 1` (Unified-entry MASHAC CPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo sha2c --env position_control --dynamics pointmass --max_iter 1 --l_rollout 2 --n_envs 2 --device cpu --log_interval 1` (Unified-entry SHA2C CPU smoke on privileged-state position control)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics quadrotor --quadrotor-control-mode body_rate --quadrotor-max-body-rates 3.0 3.0 1.5 --quadrotor-k-angvel 5.0 4.0 3.0 --quadrotor-thrust-ratio 0.85 --quadrotor-torque-ratio 0.75 --quadrotor-drag-coeff-xy 0.05 --quadrotor-drag-coeff-z 0.1 --max_iter 1 --l_rollout 1 --n_envs 1 --device cpu --log_interval 1` (Unified-entry quadrotor body-rate smoke covering DiffAero-style controller scaling and the merged factory fix)

### GPU-validated differentiable paths
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_position_control.py -q` (Position control env smoke on `cuda` when available)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_dynamics.py -q` (Low-level point-mass differentiability on `cuda` when available; CPU fallback remains for non-CUDA hosts)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q` (Point-mass environment propagation smoke on `cuda` when available)
- `conda run -n isaaclab-newton python -c "..."` driving `build_env('racing', 'pointmass')` plus APG on CUDA for short multi-iteration runs (manual racing trainer validation; recent runs still produce non-zero gate passes, and the unified APG loop now explicitly refreshes detached observations between iterations)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py -q` (DreamerV3/world GPU smoke on CUDA-capable hosts, including replay-ready world-model update coverage)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q` (Includes explicit `gpu_smoke` obstacle differentiable-loss, standalone SHAC update, and TensorBoard training checks when CUDA is available)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics pointmass --max_iter 3 --log_interval 1 --n_envs 8 --l_rollout 4` (Env-backed unified training entry smoke on default accelerator)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo world --env position_control --dynamics pointmass --max_iter 1 --l_rollout 4 --n_envs 2 --device cuda --log_interval 1 --world_warmup_steps 4 --world_min_ready_steps 2 --world_batch_size 2 --world_batch_length 2 --world_imagine_length 2` (DreamerV3/world unified-entry smoke on CUDA-capable hosts)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics continuous_pointmass --max_iter 1 --l_rollout 1 --n_envs 2 --device cuda --log_interval 1` (Continuous point-mass unified entry GPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics discrete_pointmass --max_iter 1 --l_rollout 1 --n_envs 2 --device cuda --log_interval 1` (Discrete point-mass unified entry GPU smoke)

### Known failing / blocked
- No currently confirmed blocking regression is recorded on merged `main` for the migrated surfaces summarized here.
- Remaining gaps are parity gaps rather than hard failures: fuller quadrotor aerodynamics/randomization parity, richer obstacle/sensor realism, broader DreamerV3 task coverage and convergence validation, and fuller Hydra/export/sweep tooling.

### Remaining validation gaps
- Racing is now validated on the differentiable point-mass APG path, but SHAC/PPO-style convergence coverage is still lighter than the point-mass APG validation path.
- DreamerV3/world now has one validated perception-backed path (`obstacle_avoidance` with `camera` on CPU), but broader task coverage and stronger convergence validation are still unvalidated.
- Pytest markers now distinguish `cpu_smoke`, `gpu_smoke`, and `runtime_preflight`, but the full suite has not yet been reorganized around dedicated CI jobs for those layers.

## Delivery Gates Matrix

| Capability | runtime_preflight | cpu_smoke | gpu_smoke | unified-entry |
|------------|-------------------|-----------|-----------|---------------|
| apg | - | test_apg.py | test_apg.py (CUDA) | train.py --algo apg |
| ppo | - | test_ppo_training.py | - | train.py --algo ppo |
| shac | - | test_shac_training.py | - | train.py --algo shac |
| sha2c | - | test_shac_training.py | - | train.py --algo sha2c |
| mashac | - | test_mashac_training.py | - | train.py --algo mashac |
| world | - | test_world_training.py | test_world_training.py (CUDA, state-only position_control) | train.py --algo world |
| pointmass | - | test_pointmass_env.py | test_pointmass_env.py (CUDA) | - |
| quadrotor | test_train_entry.py | test_drone_dynamics.py | - | train.py --algo apg --env position_control --dynamics quadrotor |
| obstacle_avoidance | - | test_obstacle_training.py | test_obstacle_training.py (CUDA) | - |
| racing | - | test_racing_env.py | - | - |

### Running Gates

```bash
# runtime_preflight (fastest)
pytest -m runtime_preflight -q

# cpu_smoke
pytest -m cpu_smoke -q

# gpu_smoke (requires CUDA)
pytest -m gpu_smoke -q

# unified-entry (manual verification)
conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo <algo> --env <env> --dynamics <dyn> --max_iter 1
```

## Obstacle/Sensor Scope (i8x.19)

Current supported:
- relpos sensor: sorted nearest-obstacle relative positions
- camera sensor: depth-map ray-casting
- lidar sensor: 360° ray-casting
- spherical obstacles only

NOT migrated (deferred):
- Mixed geometry (cube/wall/ceiling)
- IMU state
- Sensor mount/randomization
- Richer obstacle/state semantics

These features are deferred pending future implementation. Current obstacle/sensor capabilities are sufficient for coarse obstacle avoidance tasks.
