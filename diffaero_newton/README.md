# DiffAero Newton

Differentiable quadrotor dynamics and reinforcement learning on Newton + IsaacLab.

## Overview

This project reproduces DiffAero's key training capabilities using:
- **Newton** - Differentiable physics engine
- **IsaacLab** - Scalable vectorized task orchestration

### Features

- Differentiable quadrotor dynamics rollout
- Short-horizon risk loss for safe navigation
- SHAC-style policy training loop
- Obstacle avoidance environment

## Installation

```bash
# Add to your PYTHONPATH
export PYTHONPATH=/path/to/diffaero_newton/source:$PYTHONPATH

# Install dependencies
pip install torch numpy gymnasium warp newton isaaclab
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

## Project Structure

```
diffaero_newton/
├── source/
│   └── diffaero_newton/
│       ├── common/          # Constants, types
│       ├── configs/         # Configuration classes
│       ├── dynamics/        # Newton model, rollout
│       ├── envs/            # IsaacLab environments
│       ├── tasks/           # Obstacle tasks, rewards
│       └── training/        # SHAC algorithm
├── docs/
│   ├── api.md
│   └── quickstart.md
└── run_training.py
```

## Architecture

### Four-Layer Design

1. **Dynamics Layer** (`dynamics/`) - Newton model, control interfaces, differentiable rollout
2. **Environment Layer** (`envs/`) - IsaacLab environments, DirectRLEnv-style
3. **Task Layer** (`tasks/`) - Obstacle generation, observations, risk terms
4. **Training Layer** (`training/`) - SHAC-style training loops

### Key Components

- `Drone` - Differentiable quadrotor dynamics
- `DroneEnv` - Gymnasium-compatible environment
- `ObstacleManager` - Obstacle spawning and collision detection
- `SHAC` - Short-horizon actor-critic algorithm

## Reference Sources

- Newton drone example: `reference/newton/newton/examples/diffsim/example_diffsim_drone.py`
- IsaacLab DirectRLEnv: `reference/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`
- DiffAero obstacle task: `reference/diffaero/env/obstacle_avoidance.py`

## License

Apache 2.0
