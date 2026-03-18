# Regression Validation Matrix

This document is a compact supplement to `docs/progress.md`.

Legend:

- `✅` verified by dedicated tests or explicit unified-entry smoke
- `⚠` implemented but still narrower than reference parity
- `❌` currently blocked by a known regression
- `—` not yet validated

## Validation Matrix

| Algorithm | Dynamics | Position | Sim2Real | MAPC | Obstacle | Racing |
|-----------|----------|:---:|:---:|:---:|:---:|:---:|
| **SHAC** | PointMass | — | — | — | ✅ | ✅ |
| **APG** | PointMass | ✅ | — | — | — | ✅ |
| **APG_sto** | PointMass | ✅ | — | — | — | — |
| **PPO / APPO** | PointMass | ✅ | — | — | — | — |
| **MASHAC** | Continuous PointMass | — | — | ✅ | — | — |
| **World** | PointMass | ✅ state-only | — | — | — | — |
| **APG** | Quadrotor | ❌ | — | — | — | — |

## Validation Commands

```bash
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_train_entry.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_apg_training.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_ppo_training.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_world_training.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_mashac_training.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_obstacle_training.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_racing_env.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_pointmass_dynamics.py -q
conda run -n isaaclab-newton pytest diffaero_newton/tests/test_drone_dynamics.py -q
```

## Notes

- `Sim2RealPositionControl` has dedicated env tests, but no explicit end-to-end training validation is recorded here yet.
- `ObstacleAvoidance` and the sensor stack are capability-complete at a coarse level, but still narrower than the reference implementation on mixed geometry, IMU support, and sensor/randomization realism.
- The quadrotor row is currently blocked by the `DroneConfig` / `action_frame` factory mismatch confirmed in `test_train_entry.py` on 2026-03-18.
