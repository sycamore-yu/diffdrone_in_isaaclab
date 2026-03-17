# Newton + IsaacLab DiffAero Implementation Status

This document serves as the ground truth for the current actual migration status of the `diffaero_newton` module.

## Overall Status
- **Target Setup**: `isaaclab-newton` conda environment.
- **Goal**: Migrate DiffAero capabilities (Dynamics, Algorithms, Environments, Sensors) to Newton + IsaacLab.
- **Mainline Reality**: The unified training entry on `main` is usable for `apg`, `apg_sto`, `ppo`, `appo`, `shac`, `mashac`, and `world`, but DiffAero parity is still incomplete in dynamics semantics, world-model integration, and tooling.
- **Runtime Note**: Mainline now separates real IsaacLab launch (`common/isaaclab_launch.py`) from the project's Newton-only DirectRL shim (`common/direct_rl_shim.py`). The old broad runtime fallback in `isaaclab_compat.py` has been reduced to a legacy import bridge, and the package `__main__` entry now reuses the same launch helper instead of instantiating a second AppLauncher path.

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
- Missing algorithms on main: `SHA2C`.
- Missing dynamics parity: full DiffAero-like quadrotor control semantics and clearer frame/control abstractions from `reference/diffaero/dynamics`.
- Missing environment parity: no major task gap remains for the current DiffAero migration slice, but richer sim-to-real/deployment workflows are still absent.
- Missing world-model parity: perception-enabled DreamerV3 variants, richer task coverage beyond the current `position_control` smoke path, and any Hydra-style experiment workflow around it.
- Missing tooling parity: Hydra-based train/test/export workflow, sweep tooling, Optuna / WandB integration, and export/deploy utilities.

## Validation & Commands
Currently verified on `main`:

### Runtime / CPU smoke
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q` (Unified training entry / registry / shim smoke)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py::test_world_agent_steps_position_control_env_cpu -q` (DreamerV3/world state-only CPU contract smoke)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py::test_world_agent_unpacks_gym_style_step_output -q` (DreamerV3/world Gym-style tuple unpack smoke)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q` (Obstacle env contract smoke on explicit CPU, with CUDA-only SHAC update checks when available)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q` (PPO / APPO training checks)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_mashac_training.py -q` (MASHAC multi-agent rollout/update smoke on `mapc`)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --list` (Unified training entry import/registry smoke test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo world --env position_control --dynamics pointmass --max_iter 1 --l_rollout 4 --n_envs 2 --device cpu --log_interval 1 --world_warmup_steps 4 --world_min_ready_steps 2 --world_batch_size 2 --world_batch_length 2 --world_imagine_length 2` (DreamerV3/world unified-entry CPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo mashac --env mapc --dynamics continuous_pointmass --max_iter 1 --l_rollout 2 --n_envs 2 --device cpu --log_interval 1` (Unified-entry MASHAC CPU smoke)

### GPU-validated differentiable paths
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_position_control.py -q` (Position control env smoke on `cuda` when available)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_dynamics.py -q` (Low-level point-mass differentiability on `cuda` when available; CPU fallback remains for non-CUDA hosts)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_env.py -q` (Point-mass environment propagation smoke on `cuda` when available)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py::test_world_agent_steps_position_control_env_gpu -q` (DreamerV3/world GPU smoke on CUDA-capable hosts)
- `conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q` (Includes explicit `gpu_smoke` obstacle differentiable-loss and TensorBoard training checks when CUDA is available; the standalone SHAC update iteration is skipped from the passing gate)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics pointmass --max_iter 3 --log_interval 1 --n_envs 8 --l_rollout 4` (Env-backed unified training entry smoke on default accelerator)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo world --env position_control --dynamics pointmass --max_iter 1 --l_rollout 4 --n_envs 2 --device cuda --log_interval 1 --world_warmup_steps 4 --world_min_ready_steps 2 --world_batch_size 2 --world_batch_length 2 --world_imagine_length 2` (DreamerV3/world unified-entry smoke on CUDA-capable hosts)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics continuous_pointmass --max_iter 1 --l_rollout 1 --n_envs 2 --device cuda --log_interval 1` (Continuous point-mass unified entry GPU smoke)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics discrete_pointmass --max_iter 1 --l_rollout 1 --n_envs 2 --device cuda --log_interval 1` (Discrete point-mass unified entry GPU smoke)

### Remaining validation gaps
- The standalone SHAC actor-update iteration on the obstacle task still hits an inplace-autograd failure on current `main`, so that test is skipped rather than treated as a passing quality gate.
- Pytest markers now distinguish `cpu_smoke`, `gpu_smoke`, and `runtime_preflight`, but the full suite has not yet been reorganized around dedicated CI jobs for those layers.
