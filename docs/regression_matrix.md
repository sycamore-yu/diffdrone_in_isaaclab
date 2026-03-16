# Regression Validation Matrix

This document tracks which algorithm × environment × dynamics combinations have been validated.

## Validation Matrix

| Algorithm | Dynamics | Env: Position | Env: MAPC | Env: Obstacle | Env: Racing |
|-----------|----------|:---:|:---:|:---:|:---:|
| **SHAC** | PointMass | ✅ | ✅ | — | — |
| **APG** | PointMass | ✅ | — | — | — |
| **APG_sto** | PointMass | ✅ | — | — | — |
| **PPO** | PointMass | ✅ | — | — | — |
| **Asym PPO** | PointMass | ✅ | — | — | — |

> **Legend**: ✅ = test passes, — = not yet validated, ❌ = known failure

## Validation Commands

```bash
# All tests (run from .wt/gemini)
conda run -n isaaclab-newton python diffaero_newton/tests/test_sensors.py
conda run -n isaaclab-newton python diffaero_newton/tests/test_apg_training.py
conda run -n isaaclab-newton python diffaero_newton/tests/test_ppo_training.py
conda run -n isaaclab-newton python diffaero_newton/tests/test_obstacle_training.py

# Unified training entry (smoke test)
conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py --list
conda run -n isaaclab-newton python diffaero_newton/source/diffaero_newton/scripts/train.py \
  --algo apg --env position_control --dynamics pointmass \
  --max_iter 3 --log_interval 1 --n_envs 8 --l_rollout 4
```

## Not Yet Migrated

- **DreamerV3** (i8x.8): World-model training stack — complex, deferred
- **MASHAC** (i8x.9): Multi-agent SHAC — depends on multi-agent infra, deferred
- **Quadrotor dynamics**: Basic rollout works, full unification pending
