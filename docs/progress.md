# Newton + IsaacLab DiffAero Implementation Status

This document serves as the ground truth for the current actual migration status of the `diffaero_newton` module.

## Overall Status
- **Target Setup**: `isaaclab-newton` conda environment.
- **Goal**: Migrate DiffAero capabilities (Dynamics, Algorithms, Environments, Sensors) to Newton + IsaacLab.

## Capability Matrix Status Tracker

### Algorithms
- [x] **SHAC**: Implemented (short-horizon actor-critic with Newton differentiable physics)
- [x] **APG / APG_sto**: Implemented (deterministic APG + stochastic APG with entropy regularisation)
- [x] **PPO / Asymmetric PPO**: Implemented (clipped surrogate with GAE, privilege state critic)
- [ ] **DreamerV3**: Deferred (world-model stack, separate session)

### Dynamics
- [x] **PointMass**: Implemented (Newton backend, fully compatible with continuous forward/backward propagation)
- [ ] **Quadrotor**: Pending (Basic rollout implemented, needs full unification)

### Tasks/Environments
- [x] **Position Control**: Implemented (Single-agent target position tracking)
- [x] **Multi-Agent Position Control**: Implemented (Multi-agent with collision rewards and proper shape flattening)
- [x] **Obstacle Avoidance**: Implemented (ObstacleAvoidanceEnv with multi-modal sensor integration)
- [x] **Racing**: Implemented (figure-8 gate track with gate passing detection)

### Sensors
- [x] **Relative Position (relpos)**: Implemented (sorted nearest-obstacle relative positions)
- [x] **Camera**: Implemented (depth-map ray-casting, configurable FOV & resolution)
- [x] **Lidar**: Implemented (360° ray-casting with vertical/horizontal coverage)

## Validation & Commands
Currently working validation scripts:
- `python test_direct_env.py` (Basic DirectRLEnv test)
- `python test_drone_simple.py` (Newton quadrotor simple step test)
- `python test_pointmass_dynamics.py` (PointMass differentiable test)
- `python test_pointmass_env.py` (PointMass environment propagation test)
- `python test_position_control.py` (Single-agent Position Control test)
- `python test_mapc_env.py` (Multi-Agent Position Control test)
- `python test_sensors.py` (Camera/LiDAR/RelPos sensor gradient flow test)
- `python test_apg_training.py` (APG + APGStochastic training loop test)
- `python test_ppo_training.py` (PPO + AsymmetricPPO training loop test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --list` (Unified training entry import/registry smoke test)
- `conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --algo apg --env position_control --dynamics pointmass --max_iter 3 --log_interval 1 --n_envs 8 --l_rollout 4` (Env-backed unified training entry smoke test)
