# Quickstart Guide

## Installation

1. **Clone or copy the project**

2. **Set up dependencies**
   ```bash
   pip install torch numpy gymnasium warp newton isaaclab
   ```

3. **Add to PYTHONPATH**
   ```bash
   export PYTHONPATH=/path/to/diffaero_newton/source:$PYTHONPATH
   ```

## Basic Usage

### Creating an Environment

```python
import torch
from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg

# Create environment configuration
cfg = DroneEnvCfg(
    num_envs=256,      # Number of parallel environments
    episode_length_s=30.0,
)

# Create environment
env = DroneEnv(cfg=cfg)

# Reset and step
obs, extras = env.reset()
action = torch.zeros(256, 4)  # Zero thrust
obs, reward, terminated, truncated, extras = env.step(action)
```

### Training with SHAC

```python
import torch
from diffaero_newton.envs.drone_env import DroneEnv
from diffaero_newton.configs.drone_env_cfg import DroneEnvCfg
from diffaero_newton.configs.training_cfg import TrainingCfg
from diffaero_newton.training.shac import SHAC

# Create environment
env_cfg = DroneEnvCfg(num_envs=256)
env = DroneEnv(cfg=env_cfg)

# Create training configuration
train_cfg = TrainingCfg(
    num_iterations=10000,
    rollout_horizon=10,
    actor_lr=3e-4,
    critic_lr=1e-3,
)

# Create trainer and train
trainer = SHAC(env, cfg=train_cfg)
trainer.train()
```

### Using the Drone Dynamics Directly

```python
import torch
from diffaero_newton.dynamics.drone_dynamics import Drone, DroneConfig

# Create drone configuration
config = DroneConfig(
    num_envs=1,
    dt=0.01,
    requires_grad=False,
)

# Create drone
drone = Drone(config)

# Reset and apply control
drone.reset_states()
thrust = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # 50% thrust on all motors
drone.apply_control(thrust)
drone.integrate(0.01)

# Get state
state = drone.get_flat_state()
```

### Obstacle Avoidance

```python
import torch
from diffaero_newton.tasks.obstacle_manager import ObstacleManager
from diffaero_newton.configs.obstacle_task_cfg import ObstacleTaskCfg

# Create obstacle manager
obs_cfg = ObstacleTaskCfg(num_obstacles=5)
manager = ObstacleManager(num_envs=256, cfg=obs_cfg)

# Check for collisions
positions = torch.randn(256, 3)
collisions = manager.check_collisions(positions)

# Get distances to obstacles
distances = manager.compute_nearest_distances(positions)
```

## CLI Usage

```bash
# Basic training
python diffaero_newton/run_training.py

# Custom parameters
python diffaero_newton/run_training.py \
    --num_envs 512 \
    --num_iterations 50000 \
    --actor_lr 1e-4
```

## Configuration

### Environment Configuration

Key parameters in `DroneEnvCfg`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_envs | 256 | Parallel environments |
| episode_length_s | 30.0 | Max episode length |
| decimation | 4 | Physics steps per step |
| rollout_horizon | 10 | SHAC horizon |

### Training Configuration

Key parameters in `TrainingCfg`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| rollout_horizon | 10 | Short horizon length |
| gamma | 0.99 | Discount factor |
| lam | 0.95 | GAE lambda |
| actor_lr | 3e-4 | Actor learning rate |
| critic_lr | 1e-3 | Critic learning rate |
| clip_epsilon | 0.2 | PPO clip parameter |

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              Training Layer                     │
│  ┌─────────────────────────────────────────┐   │
│  │           SHAC Algorithm                │   │
│  │  - Actor-Critic Networks               │   │
│  │  - GAE Advantage Estimation            │   │
│  │  - PPO-style Updates                   │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              Environment Layer                  │
│  ┌─────────────────────────────────────────┐   │
│  │           DroneEnv (Gymnasium)          │   │
│  │  - Step, Reset, Reward                 │   │
│  │  - Observation Building                │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              Dynamics Layer                     │
│  ┌─────────────────────────────────────────┐   │
│  │           Drone (Newton)                │   │
│  │  - State Management                    │   │
│  │  - Control Application                 │   │
│  │  - Integration                         │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Next Steps

- See [API Documentation](api.md) for detailed reference
- Check `reference/` folder for source implementations
- Modify configs for your specific task
