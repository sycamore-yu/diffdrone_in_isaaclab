# Newton + IsaacLab DiffAero Implementation Status

This document serves as the ground truth for the current actual migration status of the `diffaero_newton` module.

## Overall Status
- **Target Setup**: `isaaclab-newton` conda environment.
- **Goal**: Migrate DiffAero capabilities (Dynamics, Algorithms, Environments, Sensors) to Newton + IsaacLab.
- **Mainline Reality**: The unified training entry on `main` is usable for `apg`, `apg_sto`, `ppo`, `appo`, `shac`, `mashac`, and `world`, and the environment registry now exposes explicit obstacle/racing observation contracts. Recent fixes restored missing point-mass control semantics needed by racing, added an opt-in DiffAero-style quadrotor rate-controller path with normalized `[0, 1]` body-rate commands, and fixed a detached-observation bug in multi-iteration APG training, but broader parity is still incomplete in algorithms, world-model breadth, env/sensor realism, and tooling. The env-backed quadrotor unified-entry path has been fixed: `dynamics/registry.py` no longer passes unsupported `action_frame` into `DroneConfig`, and the quadrotor body-rate runtime-preflight path now passes tests.
- **Runtime Note**: Mainline now separates real IsaacLab launch (`common/isaaclab_launch.py`) from the project's Newton-only DirectRL shim (`common/direct_rl_shim.py`). The old broad runtime fallback in `isaaclab_compat.py` has been reduced to a legacy import bridge, and the package `__main__` entry now reuses the same launch helper with a single resolved device path shared by both environment and trainer.

## Capability Matrix Status Tracker

### Algorithms
- [x] **SHAC**: Implemented (short-horizon actor-critic with Newton differentiable physics)
- [x] **APG / APG_sto**: Implemented (deterministic APG + stochastic APG with entropy regularisation)
- [x] **PPO / Asymmetric PPO**: Implemented (clipped surrogate with GAE, privilege state critic)
- [ ] **SHA2C**: Not migrated on main
- [x] **MASHAC**: Implemented for `mapc` with shared actor + centralized critic, wired into the unified training entry as `--algo mashac`
- [x] **DreamerV3 / world**: Wired into `scripts/registry.py` and the unified training entry as `--algo world`. Current validated path is a state-only world-model rollout on `position_control`; it now has both CPU contract smoke and a CUDA-backed smoke gate on CUDA-capable hosts, while broader task parity remains incomplete.

### Dynamics
- [x] **PointMass**: Implemented with explicit `pointmass` (backward-compatible alias), `continuous_pointmass`, and `discrete_pointmass` model options. Low-level differentiable propagation now passes `test_pointmass_dynamics.py`, and both model variants are wired into the unified training entry.
- [x] **PointMass action_frame**: Added `action_frame` config option ('world'/'local') to `DiscretePointMass`, with quaternion-to-rotation-matrix conversion for local frame transformation.
- [x] **Quadrotor**: Implemented with Newton + Warp autograd bridge. Low-level action-to-state backprop works on `main`, and obstacle-task differentiable loss tests pass.
- [ ] **Quadrotor control_mode**: Partially wired. `control_mode`, `k_angvel`, `max_body_rates`, `torque_ratio`, and `thrust_ratio` exist on the config surface, and the merged `main` branch now constructs env-backed quadrotor runs through the unified factory again. Remaining work is semantic parity: the full DiffAero controller path is still not consumed end-to-end.
- [ ] **DiffAero Quadrotor Semantics**: Not yet matched 1:1. Current mainline quadrotor uses direct normalized motor thrust input and does not yet reproduce DiffAero's rate-controller-oriented semantics or full aerodynamic model details.

### Tasks/Environments
- [x] **Position Control**: Implemented (Single-agent target position tracking)
- [x] **Multi-Agent Position Control**: Implemented (Multi-agent with collision rewards and proper shape flattening)
- [x] **Obstacle Avoidance**: Implemented at coarse capability level (ObstacleAvoidanceEnv with multi-modal sensor integration), but still narrower than the reference task on obstacle geometry, reset/randomization depth, and richer perception/state contracts.
- [x] **Racing**: Implemented and trainer-validated for the point-mass path. Figure-8 gate geometry, gate-frame observations, gate pass/collision detection, target-gate advancement, gate-frame point-mass action handling, reset semantics aligned to the reference task, reward/loss wiring, and dedicated racing smoke coverage now exist on `main`. The current reward path no longer depends on a hidden per-episode target speed when `use_vel_track=False`, and `reward_progress` is again distinct from loss weighting. Manual CUDA validation with APG continues to produce non-zero gate passes in short runs after those fixes.
- [x] **Sim2RealPositionControl**: Implemented as `sim2real_position_control` with square-target switching derived from DiffAero's reference environment; validated on the unified entry with point-mass dynamics.

### Sensors
- [x] **Relative Position (relpos)**: Implemented (sorted nearest-obstacle relative positions), but still simplified relative to the reference obstacle/state contract.
- [x] **Camera**: Implemented (depth-map ray-casting, configurable FOV & resolution), but still simplified relative to the reference mount/randomization stack.
- [x] **Lidar**: Implemented (360° ray-casting with vertical/horizontal coverage), but still simplified relative to the reference mount/randomization stack.

## Gaps Relative to Reference DiffAero

### Sensor Parity Gaps
- **IMU**: `reference/diffaero/utils/sensor.py:360-486` 完整 IMU 类（含 acc/gyro drift, noise, mounting error）；`diffaero_newton/envs/sensors.py` 完全缺失此类
- **Mixed geometry**: `sensors.py` 含 `raydist3d_cube`/`raydist3d_ground_plane` 代码，但 `ObstacleManager` 仅提供 sphere 数据，cube/ground_plane 路径从未触发
- **RelPos walls/ceiling**: `RelativePositionSensor` 有公式支持，但实现仅返回零填充

### Network/Agents Gaps
- `reference/diffaero/network/agents.py` 中所有 RNN-based actors 未移植：
  - `StochasticActorCriticV`, `StochasticAsymmetricActorCriticV`, `RPLActorCritic`, `StochasticActorCriticQ`
- `diffaero_newton` 所有 Actor/Critic 为简单 feed-forward，无 `build_network` 函数

### ObstacleManager Gaps
- 仅存储 `[x,y,z,radius]`，无 `p_cubes/lwh_cubes/rpy_cubes`
- 无 `z_ground_plane` 概念
- 无 walls/ceiling 实现

### Tooling Gaps
- `reference/utils/exporter.py` (8.4KB) 未移植
- `reference/utils/render.py` (46KB) 未移植
- `reference/utils/logger.py` (7.3KB) 未移植
- `reference/utils/math.py` (8.5KB) 未移植

### Algorithm Gaps
- (All major algorithms from reference/diffaero 已迁移)

## Validation & Commands
Passing checks currently verified on `main`:

### Runtime / CPU smoke
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py -q` (DreamerV3/world CPU and CUDA smoke, including replay-ready world-model update coverage and Gym-style tuple unpacking)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q` (Obstacle env contract smoke on explicit CPU, sensor-aware observation contract checks, reset-triggering differentiable regression, and CUDA-only SHAC update checks when available)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q` (PPO / APPO training checks)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_mashac_training.py -q` (MASHAC multi-agent rollout/update smoke on `mapc`, now assigned to the `cpu_smoke` tier)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_racing_env.py -q` (Racing observation contract smoke and gate-pass progression/reward regression)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_drone_dynamics.py -q` (Quadrotor body-rate controller, drag model, and differentiable drone dynamics regression coverage)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q` (Runtime-preflight coverage for the unified training entry, including the quadrotor body-rate configuration path restored on merged `main`)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q` (Point-mass env propagation/backward smoke plus normalized `x`-action symmetry regression coverage)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --list` (Unified training entry import/registry smoke test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo world --env position_control --dynamics pointmass --max_iter 1 --l_rollout 4 --n_envs 2 --device cpu --log_interval 1 --world_warmup_steps 4 --world_min_ready_steps 2 --world_batch_size 2 --world_batch_length 2 --world_imagine_length 2` (DreamerV3/world unified-entry CPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo mashac --env mapc --dynamics continuous_pointmass --max_iter 1 --l_rollout 2 --n_envs 2 --device cpu --log_interval 1` (Unified-entry MASHAC CPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics quadrotor --quadrotor-control-mode body_rate --quadrotor-max-body-rates 3.0 3.0 1.5 --quadrotor-k-angvel 5.0 4.0 3.0 --quadrotor-drag-coeff-xy 0.05 --quadrotor-drag-coeff-z 0.1 --max_iter 1 --l_rollout 1 --n_envs 1 --device cpu --log_interval 1` (Unified-entry quadrotor body-rate smoke after the merged factory fix)

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
- Remaining gaps are parity gaps rather than hard failures: `SHA2C`, full DiffAero quadrotor semantics, richer obstacle/sensor realism, perception-enabled DreamerV3 coverage, and Hydra/export/sweep tooling.

### Remaining validation gaps
- Racing is now validated on the differentiable point-mass APG path, but SHAC/PPO-style convergence coverage is still lighter than the point-mass APG validation path.
- DreamerV3/world remains validated only on the state-only `position_control` path; perception-backed variants and broader task coverage are still unvalidated.
- Pytest markers now distinguish `cpu_smoke`, `gpu_smoke`, and `runtime_preflight`, but the full suite has not yet been reorganized around dedicated CI jobs for those layers.

## Delivery Gates Matrix

| Capability | runtime_preflight | cpu_smoke | gpu_smoke | unified-entry |
|------------|-------------------|-----------|-----------|---------------|
| apg | - | test_apg.py | test_apg.py (CUDA) | train.py --algo apg |
| ppo | - | test_ppo_training.py | - | train.py --algo ppo |
| shac | - | test_shac_training.py | - | train.py --algo shac |
| mashac | - | test_mashac_training.py | - | train.py --algo mashac |
| world | - | test_world_training.py | test_world_training.py (CUDA) | train.py --algo world |
| pointmass | - | test_pointmass_env.py | test_pointmass_env.py (CUDA) | - |
| quadrotor | test_train_entry.py | - | - | - |
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
