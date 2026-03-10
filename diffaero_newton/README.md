# DiffAero Newton

Differentiable quadrotor dynamics and reinforcement learning on Newton + IsaacLab.

## Overview

This project reproduces DiffAero's key training capabilities using:
- **PyTorch-based differentiable dynamics** - Simplified quadrotor model
- **IsaacLab-style environment interface** - Gymnasium-compatible API

> **Note**: The current implementation uses a pure PyTorch quadrotor dynamics model. Full Newton/Warp integration is planned for future releases.

### Features

- Differentiable quadrotor dynamics with full attitude dynamics
- Short-horizon risk loss for safe navigation
- SHAC-style policy training loop (PPO-style implementation)
- Obstacle avoidance environment with collision detection

## Installation

```bash
# Activate conda environment
conda activate isaaclab-newton

# Add to your PYTHONPATH
export PYTHONPATH=/path/to/diffaero_newton/source:$PYTHONPATH

# Install dependencies (if needed)
pip install torch numpy gymnasium
```

## Quick Start

```python
from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.training.shac import SHAC

# Create environment
cfg = DroneEnvCfg(num_envs=256)
env = DroneEnv(cfg=cfg)

# Create trainer
trainer = SHAC(env)
trainer.train()
```

## CLI Usage

```bash
# Run training
python run_training.py --num_envs 256 --num_iterations 10000
```

## Testing

```bash
# Run all tests
pytest -q diffaero_newton/tests/test_obstacle_training.py

# Run with specific test
pytest -q diffaero_newton/tests/test_obstacle_training.py::TestQuadrotorDynamics
```

## Project Structure

```
diffaero_newton/
├── source/
│   └── diffaero_newton/
│       ├── common/          # Constants, types
│       ├── configs/         # Configuration classes
│       ├── dynamics/        # Quadrotor dynamics (PyTorch)
│       ├── envs/            # IsaacLab-style environments
│       ├── tasks/           # Obstacle tasks, rewards
│       └── training/        # SHAC algorithm
├── docs/
│   ├── api.md
│   └── quickstart.md
└── run_training.py
```

## Architecture

### Four-Layer Design

1. **Dynamics Layer** (`dynamics/`) - PyTorch quadrotor model with quaternion-based attitude dynamics
2. **Environment Layer** (`envs/`) - Gymnasium environments with IsaacLab-style interface
3. **Task Layer** (`tasks/`) - Obstacle generation, observations, risk terms
4. **Training Layer** (`training/`) - SHAC-style training loops

### Key Components

- `Drone` - Differentiable quadrotor dynamics with full physics
- `DroneEnv` - Gymnasium-compatible environment with obstacle avoidance
- `ObstacleManager` - Obstacle spawning and collision detection
- `SHAC` - Short-horizon actor-critic algorithm

### Dynamics Model

The current dynamics model includes:
- Position, velocity integration
- Quaternion-based attitude representation
- Angular velocity dynamics with inertia
- Control allocation (4 motors → thrust + torque)
- Euler/RK4 integration options

## Reference Sources

- Newton drone example: `reference/newton/newton/examples/diffsim/example_diffsim_drone.py`
- IsaacLab DirectRLEnv: `reference/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`
- DiffAero obstacle task: `reference/diffaero/env/obstacle_avoidance.py`
- DiffAero SHAC: `reference/diffaero/algo/SHAC.py`

## Current Limitations

- Dynamics are PyTorch-based (not true Newton/Warp)
- Training uses PPO-style clipped objective (not true differentiable SHAC)
- Full Newton integration planned for future

## License

Apache 2.0
