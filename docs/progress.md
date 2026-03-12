# Newton + IsaacLab DiffAero Implementation Status

This document serves as the ground truth for the current actual migration status of the `diffaero_newton` module.

## Overall Status
- **Target Setup**: `isaaclab-newton` conda environment.
- **Goal**: Migrate DiffAero capabilities (Dynamics, Algorithms, Environments, Sensors) to Newton + IsaacLab.

## Capability Matrix Status Tracker

### Algorithms
- [ ] **SHAC**: Pending
- [ ] **APG / APG_sto**: Pending
- [ ] **PPO / Asymmetric PPO**: Pending
- [ ] **DreamerV3**: Pending

### Dynamics
- [x] **PointMass**: Implemented (Newton backend, fully compatible with continuous forward/backward propagation)
- [ ] **Quadrotor**: Pending (Basic rollout implemented, needs full unification)

### Tasks/Environments
- [x] **Position Control**: Implemented (Single-agent target position tracking)
- [x] **Multi-Agent Position Control**: Implemented (Multi-agent with collision rewards and proper shape flattening)
- [ ] **Obstacle Avoidance**: Pending
- [ ] **Racing**: Pending

### Sensors
- [ ] **Relative Position (relpos)**: Pending
- [ ] **Camera**: Pending
- [ ] **Lidar**: Pending

## Validation & Commands
Currently working validation scripts:
- `python test_direct_env.py` (Basic DirectRLEnv test)
- `python test_drone_simple.py` (Newton quadrotor simple step test)
- `python test_pointmass_dynamics.py` (PointMass differentiable test)
- `python test_pointmass_env.py` (PointMass environment propagation test)
- `python test_position_control.py` (Single-agent Position Control test)
- `python test_mapc_env.py` (Multi-Agent Position Control test)
